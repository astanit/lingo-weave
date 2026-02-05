import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup
from ebooklib import epub

from app.services.openrouter_translate import OpenRouterTranslator


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
) -> str:
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
            if w not in cache:
                needed_words.append(w)
        per_node_tokens.append((node, tokens, list(translate_positions)))

    # Translate missing words (batched)
    if needed_words:
        mapping = translator.translate_words_in_batches(needed_words)
        cache.update(mapping)

    # Second pass: apply translations with bolding
    for node, tokens, translate_positions in per_node_tokens:
        for i in translate_positions:
            ru = tokens[i]
            en = cache.get(ru, ru)
            if options.bold_translations and en != ru:
                tokens[i] = f"<b>{en}</b>"
            else:
                tokens[i] = en

        # Replace this text node with HTML (may contain <b>)
        new_html = detokenize(tokens)
        node.replace_with(BeautifulSoup(new_html, "html.parser"))

    return str(soup)


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
    out_root = Path(outputs_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex
    out_dir = out_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    book = epub.read_epub(input_epub_path)
    items = list(book.get_items_of_type(epub.ITEM_DOCUMENT))

    translator = OpenRouterTranslator()
    cache: Dict[str, str] = {}

    total = len(items)
    for idx, item in enumerate(items):
        ratio = chapter_target_ratio(idx, total)
        raw = item.get_body_content()
        try:
            html = raw.decode("utf-8", errors="ignore")
        except Exception:
            html = str(raw)

        weaved = weave_chapter_html(html, ratio, translator, cache, options)
        item.set_content(weaved.encode("utf-8"))

    output_path = str(out_dir / "lingoweave.epub")
    epub.write_epub(output_path, book)
    return job_id, output_path

