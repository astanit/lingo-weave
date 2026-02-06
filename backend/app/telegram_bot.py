"""
Telegram bot for LingoWeave: document upload -> payment (or admin skip) -> translation -> send result.
"""
import asyncio
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Throttle progress edits to avoid Telegram rate limits (max once per N seconds)
PROGRESS_EDIT_THROTTLE_SECONDS = 4

# Global queue: only 2 books processed at a time
GLOBAL_BOOK_SEMAPHORE = asyncio.Semaphore(2)
# Parallel processing within a book (e.g. chapters)
CHAPTER_SEMAPHORE = asyncio.Semaphore(15)
_running_books = 0
_queue_waiting = 0

# Trial: first 2000 chars or first chapter, translated with Gemini
TRIAL_MODEL = "google/gemini-2.0-flash-001"
SAMPLE_CHAR_LIMIT = 2000

import ebooklib
from ebooklib import epub
from lxml import etree

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LabeledPrice,
    Message,
    PreCheckoutQuery,
    SuccessfulPayment,
)

# Paths aligned with backend/app/main.py (BASE_DIR = backend/)
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
ALLOWED_EXTENSIONS = (".epub", ".txt", ".fb2")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

ADMIN_USERNAME = "qaskar"
STARS_CURRENCY = "XTR"

# Max file size: 20 MB
FILE_SIZE_LIMIT_BYTES = 20 * 1024 * 1024

# Tier: (model_id,) — amount is computed from word_count via calculate_price
# Standard: 2 Stars per 1000 words (min 50). Premium: 10 Stars per 1000 words (min 50).
TIERS = {
    "gemini": ("google/gemini-2.0-flash-001",),
    "gpt4omini": ("openai/gpt-4o-mini",),
    "gpt4o": ("openai/gpt-4o",),
    "claude": ("anthropic/claude-3.5-sonnet",),
}
STARS_PER_1K_STANDARD = 2
STARS_PER_1K_PREMIUM = 10
MIN_STARS = 50
FALLBACK_MODEL = "openai/gpt-4o-mini"


def calculate_price(word_count: int, model_tier: str) -> int:
    """Price in Telegram Stars. Base unit 1000 words. Standard (Gemini/Mini): 2/1k; Premium (GPT-4o/Claude): 10/1k. Min 50 Stars."""
    if word_count <= 0:
        word_count = 25000
    units = max(1, (word_count + 999) // 1000)
    if model_tier in ("gemini", "gpt4omini"):
        return max(MIN_STARS, units * STARS_PER_1K_STANDARD)
    return max(MIN_STARS, units * STARS_PER_1K_PREMIUM)

# Pending model choice: choice_id -> { "file_id", "chat_id", "user_id", "file_name", "is_admin" }
_pending_choice: Dict[str, Dict[str, Any]] = {}
# Pending payments: payload_id -> { "file_id", "chat_id", "user_id", "file_name", "model_id" }
_pending: Dict[str, Dict[str, Any]] = {}

# Admin: chat_id set when admin first interacts (for error notifications)
_admin_chat_id: Optional[int] = None
# Active translation tasks count (for /status)
_active_translations: int = 0
# Recent failed jobs for /status: list of { "user_id", "username", "file_name", "error", "paid" }
_failed_jobs: List[Dict[str, Any]] = []
_FAILED_JOBS_MAX = 50

router = Router()


def _is_admin(username: Optional[str]) -> bool:
    if not username:
        return False
    return username.strip().lower().lstrip("@") == ADMIN_USERNAME


def _get_extension(filename: Optional[str]) -> str:
    if not filename:
        return ""
    lower = filename.lower()
    for ext in ALLOWED_EXTENSIONS:
        if lower.endswith(ext):
            return ext
    return ""


def _count_words_in_text(text: str) -> int:
    """Count words (sequences of letters/numbers)."""
    return len(re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text))


def _count_words_in_file(input_path: str, ext: str) -> int:
    """Extract text from file and return total word count."""
    try:
        if ext == ".txt":
            with open(input_path, "r", encoding="utf-8", errors="replace") as f:
                return _count_words_in_text(f.read())
        if ext == ".epub":
            book = epub.read_epub(input_path)
            parts = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                if not isinstance(item, epub.EpubHtml):
                    continue
                raw = item.get_content()
                if raw:
                    try:
                        parts.append(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        pass
            return _count_words_in_text(re.sub(r"<[^>]+>", " ", " ".join(parts)))
        if ext == ".fb2":
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(input_path, parser)
            root = tree.getroot()
            ns = "http://www.gribuser.ru/xml/fictionbook/2.0"
            parts = []
            for p in root.iter(f"{{{ns}}}p"):
                text = (p.text or "") + "".join((e.text or "") + (e.tail or "") for e in p)
                parts.append(text)
            return _count_words_in_text(" ".join(parts))
    except Exception as e:
        logger.exception("Word count failed: %s", e)
    return 0


def _create_sample_file(input_path: str, ext: str) -> Optional[Path]:
    """Create a sample file: first 2000 chars (TXT/FB2) or first chapter (EPUB). Returns path or None."""
    try:
        sample_id = uuid.uuid4().hex
        if ext == ".txt":
            with open(input_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(SAMPLE_CHAR_LIMIT + 500)
            if not text.strip():
                return None
            sample_path = UPLOADS_DIR / f"sample_{sample_id}.txt"
            sample_path.write_text(text[:SAMPLE_CHAR_LIMIT], encoding="utf-8")
            return sample_path
        if ext == ".epub":
            book = epub.read_epub(input_path)
            items = [
                i for i in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
                if getattr(i, "get_type", lambda: ebooklib.ITEM_DOCUMENT)() == ebooklib.ITEM_DOCUMENT
                and isinstance(i, epub.EpubHtml)
            ]
            if not items:
                return None
            first = items[0]
            content = first.get_content()
            if not content:
                return None
            new_book = epub.EpubBook()
            ch = epub.EpubHtml(
                title="Sample",
                file_name="sample.xhtml",
                content=content,
            )
            new_book.add_item(ch)
            new_book.spine = ["nav", ch]
            sample_path = UPLOADS_DIR / f"sample_{sample_id}.epub"
            epub.write_epub(str(sample_path), new_book)
            return sample_path
        if ext == ".fb2":
            parser = etree.XMLParser(recover=True)
            tree = etree.parse(input_path, parser)
            root = tree.getroot()
            ns = "http://www.gribuser.ru/xml/fictionbook/2.0"
            parts = []
            n = 0
            for p in root.iter(f"{{{ns}}}p"):
                text = (p.text or "") + "".join((e.text or "") + (e.tail or "") for e in p)
                parts.append(text)
                n += len(text)
                if n >= SAMPLE_CHAR_LIMIT:
                    break
            if not parts:
                return None
            sample_path = UPLOADS_DIR / f"sample_{sample_id}.txt"
            sample_path.write_text("\n\n".join(parts)[:SAMPLE_CHAR_LIMIT], encoding="utf-8")
            return sample_path
    except Exception as e:
        logger.exception("Create sample failed: %s", e)
    return None


async def _run_trial_then_upsell(
    bot: Bot,
    chat_id: int,
    full_path: Path,
    ext: str,
    file_id: str,
    file_name: str,
    user_id: int,
):
    """Create sample, translate with Gemini, send snippet, then show 4-model upsell with dynamic pricing."""
    word_count = _count_words_in_file(str(full_path), ext)
    sample_path = _create_sample_file(str(full_path), ext)
    if not sample_path or not sample_path.exists():
        await bot.send_message(chat_id, "Не удалось подготовить пробный фрагмент. Отправьте файл ещё раз или выберите модель ниже.")
        choice_id = uuid.uuid4().hex
        _pending_choice[choice_id] = {
            "file_id": file_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "file_name": file_name,
            "is_admin": False,
            "word_count": word_count,
        }
        msg = f"В вашей книге {word_count} слов. Выберите модель для полного перевода:" if word_count > 0 else "Чтобы перевести всю книгу целиком, выберите качество:"
        await bot.send_message(chat_id, msg, reply_markup=_model_choice_keyboard(choice_id, word_count or 0))
        return
    sample_ext = sample_path.suffix.lower()
    if sample_ext not in ALLOWED_EXTENSIONS:
        sample_ext = ".txt"
    result = await _run_translation(
        str(sample_path), sample_ext, None, chat_id, bot, model_id=TRIAL_MODEL
    )
    try:
        os.remove(sample_path)
    except OSError:
        pass
    choice_id = uuid.uuid4().hex
    if not result:
        await bot.send_message(chat_id, "Пробный фрагмент не удалось перевести. Выберите модель для полной книги:")
    else:
        output_path, _, _ = result
        if output_path and os.path.exists(output_path):
            result_filename = "SAMPLE_LingoWeave" + (".txt" if sample_ext == ".txt" else ext)
            doc = FSInputFile(output_path, filename=result_filename)
            await bot.send_document(chat_id, doc)
    upsell_msg = (
        f"В вашей книге {word_count} слов. Выберите модель для полного перевода:"
        if word_count > 0 else "Выберите модель для полного перевода:"
    )
    if result:
        upsell_msg = "👆 Это пример перевода вашей книги. " + upsell_msg
    await bot.send_message(chat_id, upsell_msg, reply_markup=_model_choice_keyboard(choice_id, word_count))
    _pending_choice[choice_id] = {
        "file_id": file_id,
        "chat_id": chat_id,
        "user_id": user_id,
        "file_name": file_name,
        "is_admin": False,
        "word_count": word_count,
    }
    try:
        os.remove(full_path)
    except OSError:
        pass


def _progress_bar(percent: int, length: int = 10) -> str:
    """Returns e.g. [▓▓░░░░░░░░] for 20%."""
    filled = round((percent / 100.0) * length)
    filled = min(max(0, filled), length)
    return "[" + "▓" * filled + "░" * (length - filled) + "]"


async def _notify_admin(bot: Bot, text: str) -> None:
    """Send a message to the admin if we know their chat_id."""
    global _admin_chat_id
    if _admin_chat_id is not None:
        try:
            await bot.send_message(_admin_chat_id, text)
        except Exception as e:
            logger.warning("Failed to notify admin: %s", e)


def _books_ahead_text(n: int) -> str:
    """Russian: '0 книг', '1 книга', '2 книги', '5 книг'."""
    if n == 0:
        return "0 книг"
    if n == 1:
        return "1 книга"
    if 2 <= n <= 4:
        return f"{n} книги"
    return f"{n} книг"


async def _run_with_semaphore(
    bot: Bot,
    chat_id: int,
    input_path: str,
    ext: str,
    result_filename: str,
    model_id: Optional[str] = None,
    *,
    paid: bool = False,
    user_username: Optional[str] = None,
    user_id: Optional[int] = None,
    file_name: Optional[str] = None,
):
    """Acquire global book semaphore (wait in queue), then run translation. Only 2 books at a time."""
    global _running_books, _queue_waiting
    _queue_waiting += 1
    n_ahead = _running_books + _queue_waiting - 1
    try:
        await bot.send_message(
            chat_id,
            f"Вы в очереди. Перед вами {_books_ahead_text(n_ahead)}. "
            "Как только освободится место, перевод начнется автоматически.",
        )
    except Exception as e:
        logger.warning("Queue message failed: %s", e)
    await GLOBAL_BOOK_SEMAPHORE.acquire()
    try:
        _queue_waiting -= 1
        _running_books += 1
        await _do_translation_flow(
            bot, chat_id, input_path, ext, result_filename, model_id=model_id,
            paid=paid, user_username=user_username, user_id=user_id, file_name=file_name,
        )
    finally:
        _running_books -= 1
        GLOBAL_BOOK_SEMAPHORE.release()


async def _run_translation(
    input_path: str,
    ext: str,
    progress_callback: Optional[Any],
    chat_id: int,
    bot: Bot,
    model_id: Optional[str] = None,
) -> Optional[tuple]:
    """Run the appropriate weaver. Returns (output_path, failed_count, total_chapters) or None."""
    from app.services.epub_weave import WeaveOptions, run_weave_epub_async
    from app.services.txt_weave import run_weave_txt_async
    from app.services.fb2_weave import run_weave_fb2_async

    opts = WeaveOptions(bold_translations=True)
    outputs_dir = str(OUTPUTS_DIR)

    async def _progress(current: int, total: int):
        if progress_callback:
            await progress_callback(current, total)

    try:
        if ext == ".epub":
            _, output_path, failed_count, total = await run_weave_epub_async(
                input_path, outputs_dir, options=opts, progress_callback=_progress, model_id=model_id
            )
            return (output_path, failed_count, total)
        elif ext == ".txt":
            _, output_path = await run_weave_txt_async(
                input_path, outputs_dir, options=opts, progress_callback=_progress, model_id=model_id
            )
            return (output_path, 0, 0)
        elif ext == ".fb2":
            _, output_path = await run_weave_fb2_async(
                input_path, outputs_dir, options=opts, progress_callback=_progress, model_id=model_id
            )
            return (output_path, 0, 0)
        else:
            return None
    except Exception as e:
        logger.exception("Translation failed: %s", e)
        return None


async def _download_document(bot: Bot, file_id: str, dest_path: Path) -> bool:
    try:
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, dest_path)
        return True
    except Exception as e:
        logger.exception("Download failed: %s", e)
        return False


def _feedback_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🆘 Сообщить о проблеме", callback_data="feedback:problem")],
        [InlineKeyboardButton(text="⭐ Всё отлично!", callback_data="feedback:ok")],
    ])


async def _do_translation_flow(
    bot: Bot,
    chat_id: int,
    input_path: str,
    ext: str,
    result_filename: str,
    model_id: Optional[str] = None,
    *,
    paid: bool = False,
    user_username: Optional[str] = None,
    user_id: Optional[int] = None,
    file_name: Optional[str] = None,
):
    global _active_translations, _failed_jobs
    _active_translations += 1
    status_msg = None
    try:
        status_msg = await bot.send_message(chat_id, "Обработка... " + _progress_bar(0) + " 0%")
        last_percent = [0]
        last_edit_time = [0.0]

        async def progress_callback(finished_chapters: int, total_chapters: int):
            if total_chapters <= 0:
                return
            progress = int((finished_chapters / total_chapters) * 100)
            progress = min(100, progress)
            now = time.monotonic()
            throttle_ok = (
                last_edit_time[0] == 0
                or (now - last_edit_time[0]) >= PROGRESS_EDIT_THROTTLE_SECONDS
            )
            is_final = progress >= 100
            if not (throttle_ok or is_final):
                return
            if progress <= last_percent[0] and not is_final:
                return
            last_percent[0] = progress
            last_edit_time[0] = now
            bar = _progress_bar(progress)
            text = f"Обработка... {bar} {progress}%"
            try:
                await bot.edit_message_text(
                    text,
                    chat_id=chat_id,
                    message_id=status_msg.message_id,
                )
            except Exception as e:
                msg = str(e).lower()
                if "message is not modified" not in msg and "same content" not in msg:
                    logger.debug("Progress edit failed: %s", e)

        result = await _run_translation(
            input_path, ext, progress_callback, chat_id, bot, model_id=model_id
        )
        if not result:
            raise RuntimeError("Translation produced no output")
        output_path, failed_count, total_chapters = result
        if not output_path or not os.path.exists(output_path):
            raise RuntimeError("Translation produced no output")

        if total_chapters > 0 and failed_count / total_chapters > 0.1:
            await _notify_admin(
                bot,
                f"⚠️ Книга завершена с большим числом сбоев: {failed_count}/{total_chapters} глав заменены оригиналом "
                f"(>{10}%). Файл: {file_name or result_filename}",
            )

        # Final status before sending file
        try:
            await bot.edit_message_text(
                "✅ Обработка завершена! Подготавливаю файл к отправке...",
                chat_id=chat_id,
                message_id=status_msg.message_id,
            )
        except Exception as e:
            if "message is not modified" not in str(e).lower():
                logger.debug("Final progress edit failed: %s", e)
        try:
            await status_msg.delete()
        except Exception:
            pass

        doc = FSInputFile(output_path, filename=result_filename)
        await bot.send_document(chat_id, doc)
        await bot.send_message(
            chat_id,
            "Ваш учебник готов! Проверьте, всё ли открывается и устраивает ли вас качество.",
            reply_markup=_feedback_keyboard(),
        )
        try:
            os.remove(input_path)
        except OSError:
            pass
    except Exception as e:
        err_text = str(e)
        logger.exception("Translation flow failed: %s", e)
        _failed_jobs.append({
            "user_id": user_id,
            "username": user_username,
            "file_name": file_name or result_filename,
            "error": err_text,
            "paid": paid,
        })
        if len(_failed_jobs) > _FAILED_JOBS_MAX:
            _failed_jobs = _failed_jobs[-_FAILED_JOBS_MAX:]

        if status_msg:
            try:
                await status_msg.delete()
            except Exception:
                pass

        await bot.send_message(
            chat_id,
            "К сожалению, произошла техническая ошибка. Я уже уведомил администратора, он поможет вам в ближайшее время. @qaskar",
        )
        username_display = f"@{user_username}" if user_username else f"ID {user_id}"
        if paid:
            await _notify_admin(
                bot,
                f"⚠️ ОШИБКА ОПЛАТЫ: Пользователь {username_display} оплатил перевод, но процесс прервался. "
                f"Ошибка: {err_text}. Нужно помочь вручную.",
            )
        else:
            await _notify_admin(
                bot,
                f"⚠️ Ошибка перевода (без оплаты): {username_display}. Ошибка: {err_text}",
            )
        await _notify_admin(
            bot,
            f"📋 User ID: {user_id}, файл: {file_name or result_filename}",
        )
    finally:
        _active_translations = max(0, _active_translations - 1)


def _model_choice_keyboard(choice_id: str, word_count: Optional[int] = None) -> InlineKeyboardMarkup:
    """word_count=None for admin (0 звёзд); else show dynamic prices per 1k words."""
    if word_count is None:
        stars = "0 звёзд"
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⚡ Gemini 2.0 Flash (Fastest) — " + stars, callback_data=f"tier:{choice_id}:gemini")],
            [InlineKeyboardButton(text="🍏 GPT-4o Mini (Balanced) — " + stars, callback_data=f"tier:{choice_id}:gpt4omini")],
            [InlineKeyboardButton(text="🤖 GPT-4o (Powerful) — " + stars, callback_data=f"tier:{choice_id}:gpt4o")],
            [InlineKeyboardButton(text="💎 Claude 3.5 Sonnet (ULTRA PREMIUM) — " + stars, callback_data=f"tier:{choice_id}:claude")],
        ])
    price_std = calculate_price(word_count, "gemini")
    price_premium = calculate_price(word_count, "gpt4o")
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"⚡ Gemini 2.0 Flash (Fastest): {price_std} ⭐", callback_data=f"tier:{choice_id}:gemini")],
        [InlineKeyboardButton(text=f"🍏 GPT-4o Mini (Balanced): {price_std} ⭐", callback_data=f"tier:{choice_id}:gpt4omini")],
        [InlineKeyboardButton(text=f"🤖 GPT-4o (Powerful): {price_premium} ⭐", callback_data=f"tier:{choice_id}:gpt4o")],
        [InlineKeyboardButton(text=f"💎 Claude 3.5 Sonnet (ULTRA PREMIUM): {price_premium} ⭐", callback_data=f"tier:{choice_id}:claude")],
    ])


@router.message(F.document)
async def on_document(message: Message, bot: Bot):
    if not message.document or not message.from_user:
        return
    filename = message.document.file_name
    ext = _get_extension(filename)
    if not ext:
        await message.answer(
            "Отправьте файл в формате EPUB, TXT или FB2."
        )
        return

    file_size = getattr(message.document, "file_size", None) or 0
    if file_size > FILE_SIZE_LIMIT_BYTES:
        await message.answer(
            "⚠️ Файл слишком большой. Для стабильной работы я принимаю книги до 20 МБ. "
            "Пожалуйста, попробуйте сжать файл или выбрать другой.",
        )
        return

    username = message.from_user.username
    chat_id = message.chat.id
    file_id = message.document.file_id
    is_admin = _is_admin(username)
    if is_admin:
        global _admin_chat_id
        _admin_chat_id = chat_id

    if is_admin:
        choice_id = uuid.uuid4().hex
        _pending_choice[choice_id] = {
            "file_id": file_id,
            "chat_id": chat_id,
            "user_id": message.from_user.id,
            "file_name": filename or ("file" + ext),
            "is_admin": True,
        }
        await message.answer("Привет, админ! Выбери модель — оплата не требуется.")
        await message.answer(
            "Выберите качество перевода (для вас бесплатно):",
            reply_markup=_model_choice_keyboard(choice_id, word_count=None),
        )
        return

    await message.answer("Книга получена! Готовлю бесплатный пробный фрагмент... ⏳")
    upload_id = uuid.uuid4().hex
    dest = UPLOADS_DIR / f"{upload_id}{ext}"
    ok = await _download_document(bot, file_id, dest)
    if not ok:
        await message.answer("Не удалось скачать файл. Отправьте его снова.")
        return
    asyncio.create_task(
        _run_trial_then_upsell(
            bot, chat_id, dest, ext, file_id,
            filename or ("file" + ext), message.from_user.id,
        )
    )


@router.callback_query(F.data.startswith("tier:"))
async def on_model_choice(callback: CallbackQuery, bot: Bot):
    parts = callback.data.split(":", 2)
    if len(parts) != 3:
        await callback.answer("Ошибка. Отправьте файл заново.", show_alert=True)
        return
    _, choice_id, tier = parts
    if tier not in TIERS:
        await callback.answer("Неизвестный вариант.", show_alert=True)
        return
    data = _pending_choice.pop(choice_id, None)
    if not data:
        await callback.answer("Сессия истекла. Отправьте файл заново.", show_alert=True)
        return

    model_id = TIERS[tier][0]
    amount = 0 if data["is_admin"] else calculate_price(data.get("word_count") or 0, tier)
    chat_id = data["chat_id"]
    file_id = data["file_id"]
    file_name = data["file_name"]
    is_admin = data["is_admin"]
    ext = _get_extension(file_name)
    if not ext:
        await callback.answer("Неверный формат файла.", show_alert=True)
        return

    await callback.answer()
    if is_admin:
        global _admin_chat_id
        _admin_chat_id = callback.message.chat.id if callback.message else chat_id
        await bot.send_message(chat_id, "Начинаю перевод без оплаты...")
        upload_id = uuid.uuid4().hex
        dest = UPLOADS_DIR / f"{upload_id}{ext}"
        ok = await _download_document(bot, file_id, dest)
        if not ok:
            await bot.send_message(chat_id, "Не удалось скачать файл.")
            return
        result_name = "lingoweave" + ext
        asyncio.create_task(
            _run_with_semaphore(
                bot, chat_id, str(dest), ext, result_name, model_id=model_id,
                paid=False, user_username=callback.from_user.username, user_id=callback.from_user.id, file_name=file_name,
            )
        )
        return

    payload_id = uuid.uuid4().hex
    _pending[payload_id] = {
        "file_id": file_id,
        "chat_id": chat_id,
        "user_id": data["user_id"],
        "file_name": file_name,
        "model_id": model_id,
    }
    await bot.send_invoice(
        chat_id=chat_id,
        title="LingoWeave — перевод книги",
        description="Diglot Weave: перевод с прогрессией и глоссарием по главам.",
        payload=payload_id,
        currency=STARS_CURRENCY,
        prices=[LabeledPrice(label="Перевод книги", amount=amount)],
    )


@router.pre_checkout_query()
async def on_pre_checkout(query: PreCheckoutQuery, bot: Bot):
    await query.answer(ok=True)


@router.message(F.successful_payment)
async def on_successful_payment(message: Message, bot: Bot):
    if not message.payment or not message.from_user:
        return
    payload = message.payment.invoice_payload
    data = _pending.pop(payload, None)
    if not data:
        await message.answer("Сессия истекла. Отправьте файл заново.")
        return

    chat_id = data["chat_id"]
    file_id = data["file_id"]
    file_name = data["file_name"]
    ext = _get_extension(file_name)
    if not ext:
        await message.answer("Неверный формат файла.")
        return

    upload_id = uuid.uuid4().hex
    dest = UPLOADS_DIR / f"{upload_id}{ext}"
    ok = await _download_document(bot, file_id, dest)
    if not ok:
        await message.answer(
            "К сожалению, не удалось скачать файл. Я уведомил администратора — он свяжется с вами. @qaskar",
        )
        _failed_jobs.append({
            "user_id": message.from_user.id,
            "username": message.from_user.username,
            "file_name": file_name,
            "error": "Download failed after payment",
            "paid": True,
        })
        if len(_failed_jobs) > _FAILED_JOBS_MAX:
            _failed_jobs[:] = _failed_jobs[-_FAILED_JOBS_MAX:]
        await _notify_admin(
            bot,
            f"⚠️ ОШИБКА ОПЛАТЫ: Пользователь @{message.from_user.username or message.from_user.id} оплатил перевод, "
            f"но не удалось скачать файл. User ID: {message.from_user.id}, файл: {file_name}. Нужно помочь вручную.",
        )
        return

    model_id = data.get("model_id")
    result_name = "lingoweave" + ext
    asyncio.create_task(
        _run_with_semaphore(
            bot, chat_id, str(dest), ext, result_name, model_id=model_id,
            paid=True, user_username=message.from_user.username, user_id=message.from_user.id, file_name=file_name,
        )
    )
    await message.answer("Оплата прошла. Начинаю перевод...")


@router.message(F.text == "/status")
async def on_status(message: Message, bot: Bot):
    if not message.from_user or not _is_admin(message.from_user.username):
        return
    global _admin_chat_id
    _admin_chat_id = message.chat.id
    failed_count = len(_failed_jobs)
    failed_blurb = ""
    if _failed_jobs:
        failed_blurb = "\n\nПоследние сбои:\n" + "\n".join(
            f"• {j.get('username') or j.get('user_id')} — {j.get('file_name', '?')}: {j.get('error', '')[:80]}"
            for j in _failed_jobs[-5:]
        )
    await message.answer(
        f"📊 Активных переводов: {_active_translations}\n"
        f"❌ Сбоев (всего): {failed_count}"
        + failed_blurb,
    )


@router.callback_query(F.data == "feedback:ok")
async def on_feedback_ok(callback: CallbackQuery, bot: Bot):
    await callback.answer()
    await callback.message.answer("Спасибо! Рады, что вам понравилось 😊")


@router.callback_query(F.data == "feedback:problem")
async def on_feedback_problem(callback: CallbackQuery, bot: Bot):
    await callback.answer()
    await callback.message.answer(
        "Напишите @qaskar в личные сообщения с описанием проблемы и приложите скриншот или файл, если нужно.",
    )


async def run_telegram_bot():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.info("TELEGRAM_BOT_TOKEN not set; Telegram bot disabled.")
        return
    bot = Bot(token=token)
    dp = Dispatcher()
    dp.include_router(router)
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
