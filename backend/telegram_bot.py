"""
Telegram bot for LingoWeave: document upload -> payment (or admin skip) -> translation -> send result.
"""
import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

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

# Paths aligned with backend/app/main.py
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
ALLOWED_EXTENSIONS = (".epub", ".txt", ".fb2")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

ADMIN_USERNAME = "qaskar"
STARS_CURRENCY = "XTR"

# Tier: (model_id, amount in Stars)
MODEL_FAST = "google/gemini-flash-1.5"
MODEL_PREMIUM = "openai/gpt-4o"
PRICE_FAST = 100
PRICE_PREMIUM = 200
TIERS = {"fast": (MODEL_FAST, PRICE_FAST), "premium": (MODEL_PREMIUM, PRICE_PREMIUM)}

# Pending model choice: choice_id -> { "file_id", "chat_id", "user_id", "file_name", "is_admin" }
_pending_choice: Dict[str, Dict[str, Any]] = {}
# Pending payments: payload_id -> { "file_id", "chat_id", "user_id", "file_name", "model_id" }
_pending: Dict[str, Dict[str, Any]] = {}

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


async def _run_translation(
    input_path: str,
    ext: str,
    send_progress: Optional[Any],
    chat_id: int,
    bot: Bot,
    model_id: Optional[str] = None,
) -> Optional[str]:
    """Run the appropriate weaver; send progress via send_progress(current, total). Returns output path or None."""
    from app.services.epub_weave import WeaveOptions, run_weave_epub_async
    from app.services.txt_weave import run_weave_txt_async
    from app.services.fb2_weave import run_weave_fb2_async

    opts = WeaveOptions(bold_translations=True)
    outputs_dir = str(OUTPUTS_DIR)

    async def progress_callback(current: int, total: int):
        if send_progress:
            await send_progress(chat_id, f"Обработано глав: {current}/{total}...")

    try:
        if ext == ".epub":
            _, output_path = await run_weave_epub_async(
                input_path, outputs_dir, options=opts, progress_callback=progress_callback, model_id=model_id
            )
        elif ext == ".txt":
            _, output_path = await run_weave_txt_async(
                input_path, outputs_dir, options=opts, progress_callback=progress_callback, model_id=model_id
            )
        elif ext == ".fb2":
            _, output_path = await run_weave_fb2_async(
                input_path, outputs_dir, options=opts, progress_callback=progress_callback, model_id=model_id
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


async def _do_translation_flow(
    bot: Bot,
    chat_id: int,
    input_path: str,
    ext: str,
    result_filename: str,
    model_id: Optional[str] = None,
):
    async def send_progress(cid: int, text: str):
        await bot.send_message(cid, text)

    output_path = await _run_translation(input_path, ext, send_progress, chat_id, bot, model_id=model_id)
    if not output_path or not os.path.exists(output_path):
        await bot.send_message(chat_id, "Ошибка при переводе. Попробуйте позже.")
        return

    doc = FSInputFile(output_path, filename=result_filename)
    await bot.send_document(chat_id, doc)
    # Optionally delete temp files later
    try:
        os.remove(input_path)
    except OSError:
        pass


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
                text="💎 Премиум (GPT-4o) — 200 звёзд",
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
        await bot.send_message(chat_id, "Начинаю перевод без оплаты...")
        upload_id = uuid.uuid4().hex
        dest = UPLOADS_DIR / f"{upload_id}{ext}"
        ok = await _download_document(bot, file_id, dest)
        if not ok:
            await bot.send_message(chat_id, "Не удалось скачать файл.")
            return
        result_name = "lingoweave" + ext
        asyncio.create_task(
            _do_translation_flow(bot, chat_id, str(dest), ext, result_name, model_id=model_id)
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
        await message.answer("Не удалось скачать файл. Попробуйте отправить снова.")
        return

    model_id = data.get("model_id")
    result_name = "lingoweave" + ext
    asyncio.create_task(
        _do_translation_flow(bot, chat_id, str(dest), ext, result_name, model_id=model_id)
    )
    await message.answer("Оплата прошла. Начинаю перевод...")


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
