import asyncio
import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple

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
    """Target English percentage: Ch1 20%, linear to last 100%. Formula: 20 + (index/(n-1))*80."""
    if total_chapters <= 1:
        return 1.0
    target_percent = 20 + (chapter_index / (total_chapters - 1)) * 80
    return max(0.20, min(1.0, target_percent / 100.0))


def count_chapter_words(html: str) -> int:
    """Count word tokens (containing at least one letter) in the chapter body."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup
    text = body.get_text(separator=" ", strip=True)
    tokens = tokenize_text(text)
    return sum(1 for t in tokens if any(c.isalpha() for c in t))


def extract_glossary_to_vocab(
    html: str, vocab: Dict[str, str], already_glossaried: Optional[Set[str]] = None
) -> None:
    """
    Parse 'Chapter Vocabulary' section: add Russian -> English to vocab; add English to already_glossaried for memory.
    Format: <h3>Chapter Vocabulary</h3> or <h3>Vocabulary for this Chapter</h3><ul><li><b>English</b> — Russian</li>...
    """
    soup = BeautifulSoup(html, "html.parser")
    for h3 in soup.find_all("h3"):
        if "vocabulary" not in (h3.get_text() or "").lower() and "chapter" not in (h3.get_text() or "").lower():
            continue
        ul = h3.find_next("ul")
        if not ul:
            continue
        for li in ul.find_all("li"):
            text = li.get_text() or ""
            b = li.find("b")
            en = (b.get_text() or "").strip() if b else ""
            for sep in ("—", " – ", " - "):
                if sep in text:
                    ru = text.split(sep, 1)[-1].strip()
                    if en and ru:
                        vocab[ru] = en
                        if already_glossaried is not None:
                            already_glossaried.add(en.lower())
                    break
        break


async def _weave_chapter_async(
    html: str,
    ratio: float,
    translator: OpenRouterTranslator,
    options: WeaveOptions,
    target_percent: float,
    previous_vocab: Optional[Dict[str, str]] = None,
    already_glossaried: Optional[Set[str]] = None,
    use_uppercase: bool = False,
    target_level: Optional[str] = None,
) -> str:
    """
    Diglot Weave: word count before AI call; then non-blocking API.
    use_uppercase=True for .txt: glossary (with <b>) at top, then plain story (no highlighting).
    target_level: A1, A2, B1, B2, C1 for word-selection difficulty.
    """
    total_words = count_chapter_words(html)
    target_words_count = max(0, int(round(total_words * ratio)))
    if target_words_count == 0:
        return html
    return await translator.diglot_weave_chapter(
        html,
        total_words,
        target_words_count,
        ratio=ratio,
        target_percent=target_percent,
        previous_vocab=previous_vocab or {},
        already_glossaried=already_glossaried,
        use_uppercase=use_uppercase,
        target_level=target_level,
    )


async def process_one_segment_async(
    segment_html: str,
    segment_index: int,
    total_segments: int,
    translator: OpenRouterTranslator,
    options: WeaveOptions,
    global_vocab: Dict[str, str],
    already_glossaried: Set[str],
    use_uppercase: bool = False,
    target_level: Optional[str] = None,
) -> str:
    """
    Process one segment (virtual chapter) with Diglot Weave. Updates global_vocab and already_glossaried from glossary.
    use_uppercase=True for .txt: glossary at top + plain story. target_level: A1–C1.
    """
    ratio = chapter_target_ratio(segment_index, total_segments)
    target_percent = round(ratio * 100)
    logger.info("Segment %s/%s: target %s%%", segment_index + 1, total_segments, target_percent)
    weaved = await _weave_chapter_async(
        segment_html,
        ratio,
        translator,
        options,
        target_percent=float(target_percent),
        previous_vocab=global_vocab,
        already_glossaried=already_glossaried,
        use_uppercase=use_uppercase,
        target_level=target_level,
    )
    extract_glossary_to_vocab(weaved, global_vocab, already_glossaried=already_glossaried)
    return weaved


def html_to_plain(html: str) -> str:
    """Convert weaved HTML to plain text: preserve *word* for bold, strip other tags."""
    # Mark bold before stripping tags
    text = re.sub(r"<b>([^<]+)</b>", r"*\1*", html, flags=re.IGNORECASE)
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator="\n").strip()


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
    previous_vocab: Optional[Dict[str, str]] = None,
    already_glossaried: Optional[Set[str]] = None,
    target_level: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Process one chapter (non-blocking). Returns (item_id, html). Uses previous_vocab for consistency.
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
        target_percent = round(ratio * 100)
        print(f"Chapter {chapter_num}: Target percent set to {target_percent}%", flush=True)
        weaved = await _weave_chapter_async(
            original,
            ratio,
            translator,
            options,
            target_percent=target_percent,
            previous_vocab=previous_vocab,
            already_glossaried=already_glossaried,
            target_level=target_level,
        )
        if weaved is None or not isinstance(weaved, str):
            weaved = original
        result = str(weaved)
        if "<b>" not in result and "<b " not in result:
            logger.warning("Chapter %s: No bold tags found in AI response", chapter_num)
    except Exception as e:
        logger.warning("Chapter %s failed (%s), using original text", chapter_num, e)
        result = original

    logger.info("Finished chapter %s", chapter_num)
    return (item_id, result)


async def _weave_epub_async(
    input_epub_path: str,
    outputs_dir: str,
    options: WeaveOptions,
    progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
    model_id: Optional[str] = None,
    target_level: Optional[str] = None,
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

    translator = OpenRouterTranslator(model=model_id)
    total = len(items)
    failed_count = 0
    global_vocab: Dict[str, str] = {}  # Russian -> English across chapters
    already_glossaried: Set[str] = set()  # English words already in previous glossaries (smart glossary memory)
    translated_items: Dict[str, str] = {}

    # Process chapters sequentially: each gets previous vocab + already_glossaried so glossary = "New Words" only
    for idx, item in enumerate(items):
        item_id = _get_item_id(item, idx)
        try:
            rid, html = await _process_single_chapter_async(
                idx,
                item,
                total,
                translator,
                options,
                previous_vocab=global_vocab,
                already_glossaried=already_glossaried,
                target_level=target_level,
            )
            if html is not None and isinstance(html, str):
                translated_items[rid] = html
                extract_glossary_to_vocab(html, global_vocab, already_glossaried=already_glossaried)
                print(f"Chapter {idx + 1}: Success", flush=True)
                if progress_callback:
                    await progress_callback(idx + 1, total)
            else:
                original_html = _get_item_html(item)
                translated_items[item_id] = original_html if original_html is not None else ""
                failed_count += 1
                print(f"Chapter {idx + 1}: No result, using original", flush=True)
        except Exception as e:
            logger.warning("Chapter %s raised %s, keeping original", idx + 1, e)
            print(f"Chapter {idx + 1}: Failed, using fallback", flush=True)
            original_html = _get_item_html(item)
            translated_items[item_id] = original_html if original_html is not None else ""
            failed_count += 1

    if failed_count > 0:
        logger.info("Book finished with %s failed chapters replaced by original text.", failed_count)

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
        if new_text is None or not new_text:
            new_text = _get_item_html(item) or ""
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
    return job_id, output_path, failed_count, total


def weave_epub(
    input_epub_path: str,
    outputs_dir: str,
    options: WeaveOptions | None = None,
) -> Tuple[str, str]:
    """
    Reads an EPUB and writes a new EPUB with progressively increasing English words.
    Returns (job_id, output_epub_path).
    """
    options = options or WeaveOptions()
    job_id, output_path, *_ = asyncio.run(
        _weave_epub_async(
            input_epub_path=input_epub_path,
            outputs_dir=outputs_dir,
            options=options,
        )
    )
    return job_id, output_path


async def run_weave_epub_async(
    input_epub_path: str,
    outputs_dir: str,
    options: WeaveOptions | None = None,
    progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
    model_id: Optional[str] = None,
    target_level: Optional[str] = None,
) -> Tuple[str, str, int, int]:
    """Async entry point for EPUB weave. Returns (job_id, output_path, failed_count, total_chapters). target_level: A1–C1."""
    options = options or WeaveOptions()
    return await _weave_epub_async(
        input_epub_path=input_epub_path,
        outputs_dir=outputs_dir,
        options=options,
        progress_callback=progress_callback,
        model_id=model_id,
        target_level=target_level,
    )
