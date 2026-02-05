import asyncio
import logging
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
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


def weave_chapter_html(
    html: str,
    ratio: float,
    translator: OpenRouterTranslator,
    cache: Dict[str, str],
    options: WeaveOptions,
    cache_lock: Optional[threading.Lock] = None,
) -> str:
    """
    Returns weaved HTML. Raises on failure so caller can fallback to original.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Collect text nodes under body
    text_nodes = []
    body = soup.body or soup
    for node in body.find_all(string=True):
        # Skip script/style
        if node.parent and node.parent.name in ("script", "style"):
            continue
        if not node.strip():
            continue
        text_nodes.append(node)

    # First pass: identify which unique RU words are needed for this ratio
    per_node_tokens: List[Tuple[object, List[str], List[int]]] = []
    needed_words: List[str] = []

    for node in text_nodes:
        tokens = tokenize_text(str(node))
        cyrillic_positions = [i for i, t in enumerate(tokens) if is_cyrillic_word(t)]
        total_cyr = len(cyrillic_positions)
        k = int(total_cyr * ratio + 1e-9)
        translate_positions = set(cyrillic_positions[:k])
        for i in translate_positions:
            w = tokens[i]
            if cache_lock:
                with cache_lock:
                    if w not in cache:
                        needed_words.append(w)
            else:
                if w not in cache:
                    needed_words.append(w)
        per_node_tokens.append((node, tokens, list(translate_positions)))

    # Translate missing words (batched). On ANY AI error (timeout, API, empty), skip translation for this batch.
    if needed_words:
        try:
            mapping = translator.translate_words_in_batches(needed_words)
            if mapping is None:
                mapping = {}
            if cache_lock:
                with cache_lock:
                    cache.update(mapping)
            else:
                cache.update(mapping)
        except Exception:
            # Fallback: leave mapping empty so cache.get(ru, ru) keeps original; caller can still get valid HTML
            pass

    # Second pass: apply translations with bolding
    for node, tokens, translate_positions in per_node_tokens:
        for i in translate_positions:
            ru = tokens[i]
            if cache_lock:
                with cache_lock:
                    en = cache.get(ru, ru)
            else:
                en = cache.get(ru, ru)
            if en is None:
                en = ru
            if options.bold_translations and en != ru:
                tokens[i] = f"<b>{en}</b>"
            else:
                tokens[i] = en

        # Replace this text node with HTML (may contain <b>)
        new_html = detokenize(tokens)
        node.replace_with(BeautifulSoup(new_html, "html.parser"))

    return str(soup)


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


def _process_single_chapter(
    idx: int,
    item,
    total: int,
    translator: OpenRouterTranslator,
    cache: Dict[str, str],
    cache_lock: threading.Lock,
    options: WeaveOptions,
) -> Tuple[int, str]:
    """
    Process one chapter. Returns (idx, html). NEVER returns None for html.
    On ANY error (timeout, API error, empty response), returns (idx, original_html).
    """
    chapter_num = idx + 1
    logger.info("Started chapter %s", chapter_num)
    original = _get_item_html(item)
    if original is None:
        original = ""
    original = str(original)

    try:
        ratio = chapter_target_ratio(idx, total)
        weaved = weave_chapter_html(
            original, ratio, translator, cache, options, cache_lock=cache_lock
        )
        # Forced fallback: never return None or non-string
        if weaved is None or not isinstance(weaved, str):
            weaved = original
        result = str(weaved)
    except Exception as e:
        logger.warning("Chapter %s failed (%s), using original text", chapter_num, e)
        result = original

    logger.info("Finished chapter %s", chapter_num)
    return (idx, result)


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
    cache: Dict[str, str] = {}
    cache_lock = threading.Lock()
    total = len(items)

    sem = asyncio.Semaphore(5)
    executor = ThreadPoolExecutor(max_workers=5)
    loop = asyncio.get_event_loop()

    async def process_chapter_async(idx: int, item) -> Tuple[int, str]:
        async with sem:
            return await loop.run_in_executor(
                executor,
                _process_single_chapter,
                idx,
                item,
                total,
                translator,
                cache,
                cache_lock,
                options,
            )

    tasks = [process_chapter_async(idx, item) for idx, item in enumerate(items)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(results):
        chapter_num = i + 1
        if isinstance(r, Exception):
            logger.warning("Chapter %s raised %s, keeping original", chapter_num, r)
            print(f"Chapter {chapter_num}: Failed, using fallback", flush=True)
            original_html = _get_item_html(items[i])
            final_content = original_html if original_html is not None else ""
            final_content = str(final_content)
            items[i].set_content(final_content.encode("utf-8"))
        else:
            idx, translated_html = r
            original_html = _get_item_html(items[idx])
            # Content validation: never pass None to set_content
            final_content = (
                translated_html
                if translated_html is not None and isinstance(translated_html, str)
                else (original_html if original_html is not None else "")
            )
            final_content = str(final_content)
            content_bytes = final_content.encode("utf-8")
            items[idx].set_content(content_bytes)
            print(f"Chapter {idx + 1}: Success", flush=True)

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
