import asyncio
import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from app.services.openrouter_translate import OpenRouterTranslator

logger = logging.getLogger(__name__)

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
TOKEN_RE = re.compile(r"(\w+|\s+|[^\w\s]+)", flags=re.UNICODE)


def is_cyrillic_word(token: str) -> bool:
    if not token:
        return False
    if not any(ch.isalpha() for ch in token):
        return False
    return bool(CYRILLIC_RE.search(token))


def tokenize_text(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall(text) if t != ""]


def detokenize(tokens: List[str]) -> str:
    return "".join(tokens)


@dataclass
class WeaveOptions:
    """
    chapter_progression: linear progression from 0.0 to 1.0 across chapters.
    Within each chapter we translate the first N cyrillic words, where N is proportional
    to that chapter's target percentage.
    """

    bold_translations: bool = True


def chapter_target_ratio(chapter_index: int, total_chapters: int) -> float:
    if total_chapters <= 1:
        return 1.0
    return chapter_index / (total_chapters - 1)


def count_chapter_words(html: str) -> int:
    """Count word tokens (containing at least one letter) in the chapter body."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup
    text = body.get_text(separator=" ", strip=True)
    tokens = tokenize_text(text)
    return sum(1 for t in tokens if any(c.isalpha() for c in t))


async def _weave_chapter_async(
    html: str,
    ratio: float,
    translator: OpenRouterTranslator,
    options: WeaveOptions,
) -> str:
    """
    Diglot Weave: word count happens before AI call; then non-blocking API.
    Returns weaved HTML. Raises on failure so caller can fallback.
    """
    total_words = count_chapter_words(html)
    target_words_count = max(0, int(round(total_words * ratio)))
    if target_words_count == 0:
        return html
    return await translator.diglot_weave_chapter(
        html, total_words, target_words_count, ratio=ratio
    )


def _get_item_html(item) -> str:
    raw = item.get_content()
    try:
        if raw is None:
            return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)
    except Exception:
        return str(raw) if raw is not None else ""


def _get_item_id(item, idx: int):
    """Stable id for an item; use for matching results to items."""
    got = getattr(item, "get_id", None)
    if got is not None:
        try:
            id_val = got()
            if id_val is not None:
                return str(id_val)
        except Exception:
            pass
    return f"item_{idx}"


async def _process_single_chapter_async(
    idx: int,
    item,
    total: int,
    translator: OpenRouterTranslator,
    options: WeaveOptions,
) -> Tuple[str, str]:
    """
    Process one chapter (non-blocking). Word count before AI call. Returns (item_id, html).
    On ANY error, returns (item_id, original_html).
    """
    item_id = _get_item_id(item, idx)
    chapter_num = idx + 1
    logger.info("Started chapter %s", chapter_num)
    original = _get_item_html(item)
    if original is None:
        original = ""
    original = str(original)

    try:
        ratio = chapter_target_ratio(idx, total)
        weaved = await _weave_chapter_async(original, ratio, translator, options)
        if weaved is None or not isinstance(weaved, str):
            weaved = original
        result = str(weaved)
    except Exception as e:
        logger.warning("Chapter %s failed (%s), using original text", chapter_num, e)
        result = original

    logger.info("Finished chapter %s", chapter_num)
    return (item_id, result)


async def _weave_epub_async(
    input_epub_path: str,
    outputs_dir: str,
    options: WeaveOptions,
) -> Tuple[str, str]:
    options = options or WeaveOptions()
    out_root = Path(outputs_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex
    out_dir = out_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    book = epub.read_epub(input_epub_path)
    all_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    # Only process ITEM_DOCUMENT (skip images, styles, etc.)
    items = [
        i
        for i in all_items
        if getattr(i, "get_type", lambda: ebooklib.ITEM_DOCUMENT)() == ebooklib.ITEM_DOCUMENT
    ]

    translator = OpenRouterTranslator()
    total = len(items)

    sem = asyncio.Semaphore(20)

    async def process_chapter_async(idx: int, item):
        async with sem:
            return await _process_single_chapter_async(
                idx, item, total, translator, options
            )

    tasks = [process_chapter_async(idx, item) for idx, item in enumerate(items)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Match by ID: store translated content (string) by item_id
    translated_items: Dict[str, str] = {}
    for i, r in enumerate(results):
        item_id = _get_item_id(items[i], i)
        if isinstance(r, Exception):
            logger.warning("Chapter %s raised %s, keeping original", i + 1, r)
            print(f"Chapter {i + 1}: Failed, using fallback", flush=True)
            original_html = _get_item_html(items[i])
            translated_items[item_id] = original_html if original_html is not None else ""
        else:
            rid, html = r
            translated_items[rid] = html if (html is not None and isinstance(html, str)) else _get_item_html(items[i]) or ""
            print(f"Chapter {i + 1}: Success", flush=True)

    # Strict exclusion: never call set_content on toc, nav, ncx, style, image (leave original in object)
    EXCLUDED_SUBSTRINGS = ("toc", "nav", "ncx", "style", "image")

    for item in book.get_items():
        item_id = getattr(item, "get_id", lambda: None)()
        if item_id is None:
            item_id = getattr(item, "identifier", None) or id(item)
        item_id = str(item_id)

        id_lower = item_id.lower()
        if any(sub in id_lower for sub in EXCLUDED_SUBSTRINGS):
            # Technical file: skip entirely, do not call set_content
            continue

        # Only call set_content for EpubHtml items that we translated
        if not isinstance(item, epub.EpubHtml):
            continue
        if item_id not in translated_items:
            print(f"Leaving original: {item_id}", flush=True)
            continue

        new_text = translated_items[item_id]
        if new_text:
            print(f"Updating content for: {item_id}", flush=True)
            try:
                item.set_content(new_text.encode("utf-8"))
            except Exception as e:
                print(f"Failed to set content for {item_id}: {e}", flush=True)
        else:
            print(f"Skipping {item_id} - translation is empty", flush=True)

    output_path = str(out_dir / "lingoweave.epub")
    epub.write_epub(output_path, book)
    return job_id, output_path


def weave_epub(
    input_epub_path: str,
    outputs_dir: str,
    options: WeaveOptions | None = None,
) -> Tuple[str, str]:
    """
    Reads an EPUB and writes a new EPUB with progressively increasing English words.
    Uses parallel chapter processing (up to 5 at a time) and falls back to original
    text on any failure.
    Returns (job_id, output_epub_path).
    """
    options = options or WeaveOptions()
    return asyncio.run(
        _weave_epub_async(
            input_epub_path=input_epub_path,
            outputs_dir=outputs_dir,
            options=options,
        )
    )
