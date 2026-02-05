"""
Telegram bot for LingoWeave: document upload -> payment (or admin skip) -> translation -> send result.
"""
import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Tier: (model_id, amount in Stars) — slugs per OpenRouter docs
MODEL_FAST = "google/gemini-2.0-flash-001"
MODEL_PREMIUM = "anthropic/claude-3.5-sonnet"
FALLBACK_MODEL = "openai/gpt-4o-mini"
PRICE_FAST = 100
PRICE_PREMIUM = 200
TIERS = {"fast": (MODEL_FAST, PRICE_FAST), "premium": (MODEL_PREMIUM, PRICE_PREMIUM)}

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


async def _run_translation(
    input_path: str,
    ext: str,
    progress_callback: Optional[Any],
    chat_id: int,
    bot: Bot,
    model_id: Optional[str] = None,
) -> Optional[str]:
    """Run the appropriate weaver; progress_callback(current, total) if provided. Returns output path or None."""
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
            _, output_path = await run_weave_epub_async(
                input_path, outputs_dir, options=opts, progress_callback=_progress, model_id=model_id
            )
        elif ext == ".txt":
            _, output_path = await run_weave_txt_async(
                input_path, outputs_dir, options=opts, progress_callback=_progress, model_id=model_id
            )
        elif ext == ".fb2":
            _, output_path = await run_weave_fb2_async(
                input_path, outputs_dir, options=opts, progress_callback=_progress, model_id=model_id
            )
        else:
            return None
        return output_path
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
        # Send initial progress message and edit it every ~10%
        status_msg = await bot.send_message(chat_id, "Обработка... " + _progress_bar(0) + " 0%")
        last_percent = [0]  # use list to allow closure to mutate

        async def send_progress(cid: int, text: str):
            # Progress bar: only update every 10% (or on first/last)
            pass  # we use progress_callback below instead

        async def progress_callback(current: int, total: int):
            if total <= 0:
                return
            percent = min(100, round(100 * current / total))
            if percent >= last_percent[0] + 10 or current == total:
                last_percent[0] = percent
                bar = _progress_bar(percent)
                try:
                    await bot.edit_message_text(
                        f"Обработка... {bar} {percent}%",
                        chat_id=cid,
                        message_id=status_msg.message_id,
                    )
                except Exception:
                    pass  # ignore edit conflicts / rate limit

        output_path = await _run_translation(
            input_path, ext, progress_callback, chat_id, bot, model_id=model_id
        )
        if not output_path or not os.path.exists(output_path):
            raise RuntimeError("Translation produced no output")

        # Remove progress message (optional; or leave and edit to "Готово")
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


def _model_choice_keyboard(choice_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text="⚡ Быстрый (Gemini Flash) — 100 звёзд",
                callback_data=f"tier:{choice_id}:fast",
            ),
        ],
        [
            InlineKeyboardButton(
                text="💎 Премиум (Claude) — 200 звёзд",
                callback_data=f"tier:{choice_id}:premium",
            ),
        ],
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

    username = message.from_user.username
    chat_id = message.chat.id
    file_id = message.document.file_id
    is_admin = _is_admin(username)
    if is_admin:
        global _admin_chat_id
        _admin_chat_id = chat_id

    choice_id = uuid.uuid4().hex
    _pending_choice[choice_id] = {
        "file_id": file_id,
        "chat_id": chat_id,
        "user_id": message.from_user.id,
        "file_name": filename or ("file" + ext),
        "is_admin": is_admin,
    }

    if is_admin:
        await message.answer("Привет, админ! Выбери модель — оплата не требуется.")
    await message.answer(
        "Выберите качество перевода:",
        reply_markup=_model_choice_keyboard(choice_id),
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

    model_id, amount = TIERS[tier]
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
            _do_translation_flow(
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
        _do_translation_flow(
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
