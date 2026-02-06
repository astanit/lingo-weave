"""
TXT Diglot Weave: virtual chapters (~5000 chars), same progression/glossary/bolding logic.
"""
import html
import logging
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple

from app.services.epub_weave import (
    WeaveOptions,
    process_one_segment_async,
)
from app.services.openrouter_translate import OpenRouterTranslator

logger = logging.getLogger(__name__)

SEGMENT_SIZE = 5000


def _split_into_segments(text: str, max_chars: int = SEGMENT_SIZE) -> List[str]:
    """Split text into segments of ~max_chars, breaking at newlines when possible."""
    text = text.strip()
    if not text:
        return []
    segments = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer break at newline
            last_nl = text.rfind("\n", start, end + 1)
            if last_nl > start:
                end = last_nl + 1
        segments.append(text[start:end].strip())
        start = end
    return [s for s in segments if s]


def _wrap_segment_html(plain: str) -> str:
    """Wrap plain text in minimal HTML for the AI (same format as chapter body)."""
    escaped = html.escape(plain)
    return f"<body><p>{escaped.replace(chr(10), '</p><p>')}</p></body>"


def weave_txt(
    input_txt_path: str,
    outputs_dir: str,
    options: WeaveOptions | None = None,
) -> Tuple[str, str]:
    """
    Read TXT, split into virtual chapters (~5000 chars), process each with Diglot Weave
    (progression 5%->100%, glossary, bolding). Join and return (job_id, output_path).
    """
    import asyncio

    options = options or WeaveOptions()
    out_root = Path(outputs_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    out_dir = out_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_txt_path, "r", encoding="utf-8", errors="replace") as f:
        full_text = f.read()

    segments_plain = _split_into_segments(full_text)
    if not segments_plain:
        output_path = str(out_dir / "lingoweave.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("")
        return job_id, output_path

    translator = OpenRouterTranslator()
    global_vocab: Dict[str, str] = {}
    already_glossaried: Set[str] = set()
    total = len(segments_plain)

    async def run_all(progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None):
        results = []
        for idx, plain in enumerate(segments_plain):
            seg_html = _wrap_segment_html(plain)
            weaved = await process_one_segment_async(
                seg_html, idx, total, translator, options, global_vocab, already_glossaried,
                use_uppercase=True,
            )
            # TXT: weaved is already plain text with UPPERCASE (no HTML)
            results.append(weaved)
            if progress_callback:
                await progress_callback(idx + 1, total)
        return results

    results_plain = asyncio.run(run_all())
    output_path = str(out_dir / "lingoweave.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(results_plain))

    return job_id, output_path


async def run_weave_txt_async(
    input_txt_path: str,
    outputs_dir: str,
    options: Optional[WeaveOptions] = None,
    progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
    model_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Async entry point for TXT weave with optional progress (e.g. for Telegram bot)."""
    options = options or WeaveOptions()
    out_root = Path(outputs_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    out_dir = out_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_txt_path, "r", encoding="utf-8", errors="replace") as f:
        full_text = f.read()

    segments_plain = _split_into_segments(full_text)
    if not segments_plain:
        output_path = str(out_dir / "lingoweave.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("")
        return job_id, output_path

    translator = OpenRouterTranslator(model=model_id)
    global_vocab: Dict[str, str] = {}
    already_glossaried: Set[str] = set()
    total = len(segments_plain)

    results_plain = []
    for idx, plain in enumerate(segments_plain):
        seg_html = _wrap_segment_html(plain)
        weaved = await process_one_segment_async(
            seg_html, idx, total, translator, options, global_vocab, already_glossaried,
            use_uppercase=True,
        )
        results_plain.append(weaved)  # TXT: already plain text with UPPERCASE
        if progress_callback:
            await progress_callback(idx + 1, total)

    output_path = str(out_dir / "lingoweave.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(results_plain))

    return job_id, output_path
