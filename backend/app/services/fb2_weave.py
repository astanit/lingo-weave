"""
FB2 Diglot Weave: extract <p> from body, process in segments, re-assemble valid .fb2.
"""
import logging
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple

from lxml import etree

from app.services.epub_weave import (
    WeaveOptions,
    html_to_plain,
    process_one_segment_async,
)
from app.services.openrouter_translate import OpenRouterTranslator

logger = logging.getLogger(__name__)

FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"
NSMAP = {"fb": FB2_NS}
SEGMENT_SIZE = 5000


def _collect_p_elements(root: etree._Element) -> List[Tuple[etree._Element, str]]:
    """Collect all <p> elements and their text from body sections. Returns [(elem, text), ...]."""
    items = []
    for p in root.iter(f"{{{FB2_NS}}}p"):
        text = (p.text or "") + "".join(
            (e.text or "") + (e.tail or "") for e in p
        ).strip()
        items.append((p, text))
    return items


def _segment_by_char_count(
    items: List[Tuple[etree._Element, str]], max_chars: int = SEGMENT_SIZE
) -> List[List[Tuple[etree._Element, str]]]:
    """Group (elem, text) into segments of ~max_chars total."""
    segments = []
    current = []
    current_len = 0
    for elem, text in items:
        current.append((elem, text))
        current_len += len(text) + 1
        if current_len >= max_chars and current:
            segments.append(current)
            current = []
            current_len = 0
    if current:
        segments.append(current)
    return segments


def _wrap_segment_html(segment_items: List[Tuple[etree._Element, str]]) -> str:
    """Wrap segment texts in minimal HTML for AI."""
    parts = []
    for _, text in segment_items:
        parts.append(f"<p>{text}</p>")
    return "<body>" + "\n".join(parts) + "</body>"


def _weaved_html_to_paragraphs(weaved_html: str) -> List[str]:
    """Extract body after <hr>, split into paragraphs (plain text)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(weaved_html, "html.parser")
    body = soup.body or soup
    # Find <hr> and take content after it
    hr = body.find("hr")
    if hr:
        rest = "".join(str(c) for c in hr.find_next_siblings())
    else:
        rest = str(body)
    plain = html_to_plain(rest)
    # Split by double newline
    paras = [p.strip() for p in plain.split("\n\n") if p.strip()]
    return paras


def weave_fb2(
    input_fb2_path: str,
    outputs_dir: str,
    options: WeaveOptions | None = None,
) -> Tuple[str, str]:
    """
    Parse FB2, extract <p> from body, process in segments (Diglot Weave), re-assemble .fb2.
    Returns (job_id, output_path).
    """
    import asyncio

    options = options or WeaveOptions()
    out_root = Path(outputs_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    out_dir = out_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = etree.XMLParser(recover=True, remove_blank_text=False)
    tree = etree.parse(input_fb2_path, parser)
    root = tree.getroot()

    items = _collect_p_elements(root)
    if not items:
        output_path = str(out_dir / "lingoweave.fb2")
        tree.write(output_path, encoding="utf-8", xml_declaration=True, method="xml")
        return job_id, output_path

    segments = _segment_by_char_count(items)
    translator = OpenRouterTranslator()
    global_vocab: Dict[str, str] = {}
    already_glossaried: Set[str] = set()
    total = len(segments)

    async def run_all(progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None):
        results = []
        for idx, seg in enumerate(segments):
            seg_html = _wrap_segment_html(seg)
            weaved = await process_one_segment_async(
                seg_html, idx, total, translator, options, global_vocab, already_glossaried
            )
            results.append(weaved)
            if progress_callback:
                await progress_callback(idx + 1, total)
        return results

    weaved_list = asyncio.run(run_all())

    # Map results back to p elements; insert glossary at start of first segment's first section
    for seg_idx, (segment_items, weaved_html) in enumerate(zip(segments, weaved_list)):
        paras = _weaved_html_to_paragraphs(weaved_html)
        # Match count: if fewer paras than p elements, pad; if more, truncate
        for i, (elem, _) in enumerate(segment_items):
            new_text = paras[i] if i < len(paras) else ""
            elem.text = new_text
            for child in list(elem):
                elem.remove(child)

    output_path = str(out_dir / "lingoweave.fb2")
    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
        method="xml",
    )
    return job_id, output_path


async def run_weave_fb2_async(
    input_fb2_path: str,
    outputs_dir: str,
    options: Optional[WeaveOptions] = None,
    progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
    model_id: Optional[str] = None,
    target_level: Optional[str] = None,
) -> Tuple[str, str]:
    """Async entry point for FB2 weave with optional progress (e.g. for Telegram bot)."""
    options = options or WeaveOptions()
    out_root = Path(outputs_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    out_dir = out_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = etree.XMLParser(recover=True, remove_blank_text=False)
    tree = etree.parse(input_fb2_path, parser)
    root = tree.getroot()

    items = _collect_p_elements(root)
    if not items:
        output_path = str(out_dir / "lingoweave.fb2")
        tree.write(output_path, encoding="utf-8", xml_declaration=True, method="xml")
        return job_id, output_path

    segments = _segment_by_char_count(items)
    translator = OpenRouterTranslator(model=model_id)
    global_vocab: Dict[str, str] = {}
    already_glossaried: Set[str] = set()
    total = len(segments)

    weaved_list = []
    for idx, seg in enumerate(segments):
        seg_html = _wrap_segment_html(seg)
        weaved = await process_one_segment_async(
            seg_html, idx, total, translator, options, global_vocab, already_glossaried,
            target_level=target_level,
        )
        weaved_list.append(weaved)
        if progress_callback:
            await progress_callback(idx + 1, total)

    for seg_idx, (segment_items, weaved_html) in enumerate(zip(segments, weaved_list)):
        paras = _weaved_html_to_paragraphs(weaved_html)
        for i, (elem, _) in enumerate(segment_items):
            new_text = paras[i] if i < len(paras) else ""
            elem.text = new_text
            for child in list(elem):
                elem.remove(child)

    output_path = str(out_dir / "lingoweave.fb2")
    tree.write(output_path, encoding="utf-8", xml_declaration=True, method="xml")
    return job_id, output_path
