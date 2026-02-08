"""
Anki/Quizlet CSV export: glossary words, deduplicate, first occurrence sentence.
Format: semicolon-delimited, Anki headers, Example with <b>WORD</b> for target word.
"""
import re
from pathlib import Path
from typing import List, Tuple

from bs4 import BeautifulSoup

# Anki import directives (must be at top of file)
ANKI_HEADER_LINES = [
    "#separator:Semicolon",
    "#html:true",
    "#columns:Front;Back;Example",
]


def _first_sentence_containing(text: str, word: str) -> str:
    """Find the first sentence in text that contains the given word (case-insensitive). Returns cleaned sentence or empty."""
    if not text or not word or not word.strip():
        return ""
    word_clean = word.strip()
    # Split into sentences (., !, ?, newline)
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Word boundary or inside <b>word</b>
        if re.search(rf"\b{re.escape(word_clean)}\b", s, re.IGNORECASE):
            # Strip HTML tags for example
            soup = BeautifulSoup(s, "html.parser")
            return soup.get_text(separator=" ", strip=True)[:300] or ""
        if word_clean.lower() in s.lower():
            soup = BeautifulSoup(s, "html.parser")
            return soup.get_text(separator=" ", strip=True)[:300] or ""
    return ""


def get_glossary_entries_with_examples(weaved_html: str) -> List[Tuple[str, str, str]]:
    """
    Parse weaved chapter HTML: extract glossary (en, ru) and for each English word
    find the first sentence in the chapter body that contains it.
    Returns list of (front, back, example) for Anki.
    """
    soup = BeautifulSoup(weaved_html, "html.parser")
    entries: List[Tuple[str, str, str]] = []

    # Get body text for sentence search (strip glossary section)
    body = soup.body or soup
    body_text = body.get_text(separator=" ", strip=True)

    # Parse glossary: <h3>Chapter Vocabulary</h3> or "Glossary:" then lines "word — перевод"
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
                        example = _first_sentence_containing(weaved_html, en)
                        entries.append((en, ru, example or ""))
                    break
        break

    # Fallback: plain "Glossary:" section (e.g. TXT-style at end)
    if not entries and "glossary" in weaved_html.lower():
        # Look for lines like "word — перевод" after "Glossary"
        plain = soup.get_text(separator="\n")
        in_glossary = False
        for line in plain.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "glossary" in line.lower():
                in_glossary = True
                continue
            if in_glossary:
                for sep in ("—", " – ", " - ", ":"):
                    if sep in line:
                        parts = line.split(sep, 1)
                        if len(parts) == 2:
                            en, ru = parts[0].strip(), parts[1].strip()
                            if en and ru and en.isascii():
                                example = _first_sentence_containing(weaved_html, en)
                                entries.append((en, ru, example or ""))
                        break
    return entries


def _example_with_bold_uppercase(example: str, word: str) -> str:
    """
    Put the target word in the example as <b>WORD</b> (uppercase).
    Normalize: no newlines (replace with space), collapse spaces.
    """
    if not example or not word or not word.strip():
        return (example or "").strip()
    word_clean = word.strip()
    # Single line, no stray newlines
    normalized = re.sub(r"\s+", " ", (example or "").replace("\n", " ").replace("\r", " ")).strip()
    # Replace first occurrence of word (case-insensitive) with <b>WORD</b>
    pattern = re.compile(re.escape(word_clean), re.IGNORECASE)
    match = pattern.search(normalized)
    if match:
        normalized = (
            normalized[: match.start()]
            + f"<b>{word_clean.upper()}</b>"
            + normalized[match.end() :]
        )
    return normalized


def _csv_escape(field: str) -> str:
    """Escape field for semicolon-delimited CSV: quote if contains ; or \" or newline."""
    if not field:
        return ""
    if ";" in field or '"' in field or "\n" in field or "\r" in field:
        return '"' + field.replace('"', '""') + '"'
    return field


def build_anki_csv(
    entries: List[Tuple[str, str, str]],
    output_dir: str,
    book_title: str,
) -> str:
    """
    Deduplicate by Front (first occurrence wins), write Anki-compatible CSV.
    Delimiter: semicolon. Headers: #separator:Semicolon, #html:true, #columns:Front;Back;Example.
    Example field: target word wrapped in <b> and UPPERCASE. UTF-8 with BOM.
    Filename: [BOOK_TITLE]_LingoWeave_Flashcards.csv
    Returns path to written file.
    """
    if not entries:
        return ""

    # Deduplicate by front (word), keep first
    seen: set = set()
    unique: List[Tuple[str, str, str]] = []
    for front, back, example in entries:
        key = front.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        front_s = front.strip()
        back_s = back.strip()
        example_s = _example_with_bold_uppercase((example or "").strip(), front_s)
        unique.append((front_s, back_s, example_s))

    safe_title = re.sub(r"[^\w\s-]", "", book_title).strip() or "Book"
    safe_title = re.sub(r"[-\s]+", "_", safe_title)[:80]
    filename = f"{safe_title}_LingoWeave_Flashcards.csv"
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        for line in ANKI_HEADER_LINES:
            f.write(line + "\n")
        for front, back, example in unique:
            row = _csv_escape(front) + ";" + _csv_escape(back) + ";" + _csv_escape(example)
            f.write(row + "\n")

    return str(path)
