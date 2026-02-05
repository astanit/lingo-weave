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
    FSInputFile,
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
STARS_AMOUNT = 100
STARS_CURRENCY = "XTR"

# Pending payments: payload_id -> { "file_id", "chat_id", "user_id", "file_name" }
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
                input_path, outputs_dir, options=opts, progress_callback=progress_callback
            )
        elif ext == ".txt":
            _, output_path = await run_weave_txt_async(
                input_path, outputs_dir, options=opts, progress_callback=progress_callback
            )
        elif ext == ".fb2":
            _, output_path = await run_weave_fb2_async(
                input_path, outputs_dir, options=opts, progress_callback=progress_callback
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
):
    async def send_progress(cid: int, text: str):
        await bot.send_message(cid, text)

    output_path = await _run_translation(input_path, ext, send_progress, chat_id, bot)
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
        await message.answer("Привет, админ! Начинаю перевод без оплаты...")
        upload_id = uuid.uuid4().hex
        dest = UPLOADS_DIR / f"{upload_id}{ext}"
        ok = await _download_document(bot, file_id, dest)
        if not ok:
            await message.answer("Не удалось скачать файл.")
            return
        result_name = "lingoweave" + ext
        asyncio.create_task(
            _do_translation_flow(bot, chat_id, str(dest), ext, result_name)
        )
        return

    # Regular user: request payment
    await message.answer(
        "Книга получена. Стоимость перевода: 100 звезд Телеграм."
    )
    payload_id = uuid.uuid4().hex
    _pending[payload_id] = {
        "file_id": file_id,
        "chat_id": chat_id,
        "user_id": message.from_user.id,
        "file_name": filename or ("file" + ext),
    }

    await bot.send_invoice(
        chat_id=chat_id,
        title="LingoWeave — перевод книги",
        description="Diglot Weave: перевод с прогрессией и глоссарием по главам.",
        payload=payload_id,
        currency=STARS_CURRENCY,
        prices=[LabeledPrice(label="Перевод книги", amount=STARS_AMOUNT)],
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

    result_name = "lingoweave" + ext
    asyncio.create_task(
        _do_translation_flow(bot, chat_id, str(dest), ext, result_name)
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
