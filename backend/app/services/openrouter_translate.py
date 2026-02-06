import asyncio
import json
import logging
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "openai/gpt-4o-mini"
TIER2_MODEL = "openai/gpt-4o-mini"
TIER3_MODEL = "google/gemini-2.0-flash-001"
TIER1_ATTEMPTS = 3
TIER1_DELAY_SEC = 5
TIER2_ATTEMPTS = 2
TIER3_ATTEMPTS = 2


def _is_retryable_error(exc: Exception) -> bool:
    """True if we should try next tier (404, 502, 503, JSON, protocol, etc.)."""
    msg = str(exc).lower()
    if isinstance(exc, json.JSONDecodeError):
        return True
    try:
        import httpx
        if isinstance(exc, httpx.RemoteProtocolError):
            return True
    except ImportError:
        pass
    if "model" in msg and "not found" in msg:
        return True
    if "502" in msg or "503" in msg or "504" in msg:
        return True
    if hasattr(exc, "status_code"):
        if getattr(exc, "status_code") in (404, 502, 503, 504):
            return True
    if hasattr(exc, "response") and getattr(exc.response, "status_code", None) in (404, 502, 503, 504):
        return True
    return False


def _output_length_ok(original_html: str, result_content: str) -> bool:
    """True if result is at least MIN_OUTPUT_LENGTH_RATIO of original length (avoid empty/summary chapters)."""
    if not original_html or not result_content:
        return False
    return len(result_content) >= MIN_OUTPUT_LENGTH_RATIO * len(original_html)


def _strip_json_fence(content: str) -> str:
    """Remove ```json and ``` wrapper if present."""
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_diglot_json(content: str) -> Optional[Tuple[str, List[str]]]:
    """
    Parse LLM response as JSON. Returns (main_text, glossary_list) or None if invalid.
    glossary_list items are strings like "word: перевод".
    """
    if not content or not content.strip():
        return None
    s = _strip_json_fence(content)
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        logger.debug("Diglot JSON parse failed: %s", e)
        return None
    if not isinstance(data, dict):
        return None
    main_text = data.get("main_text")
    glossary = data.get("glossary")
    if main_text is None or not isinstance(main_text, str):
        return None
    if glossary is None:
        glossary = []
    if not isinstance(glossary, list):
        glossary = [str(g) for g in glossary] if hasattr(glossary, "__iter__") else []
    glossary = [str(item).strip() for item in glossary if item]
    return (main_text.strip(), glossary)


def _assemble_chapter_from_json(main_text: str, glossary: List[str], use_uppercase: bool) -> str:
    """
    Assemble final chapter: glossary block + separator + main_text.
    Glossary is formatted as HTML so extract_glossary_to_vocab can parse it (EPUB/TXT).
    """
    if not glossary:
        return main_text
    # Format each "word: перевод" as <li><b>word</b> — перевод</li>
    items = []
    for entry in glossary:
        entry = entry.strip()
        for sep in (": ", ":", " — ", " - "):
            if sep in entry:
                parts = entry.split(sep, 1)
                if len(parts) == 2:
                    word, trans = parts[0].strip(), parts[1].strip()
                    if word:
                        items.append(f"<li><b>{word}</b> — {trans}</li>")
                break
        else:
            if entry:
                items.append(f"<li><b>{entry}</b></li>")
    if not items:
        return main_text
    glossary_html = (
        "<h3>Chapter Vocabulary</h3><ul>"
        + "".join(items)
        + "</ul><hr/>"
    )
    return glossary_html + "\n" + main_text


def _is_model_not_found_error(exc: Exception) -> bool:
    """True if API returned 404 or 'Model not found' (for sync fallback)."""
    msg = str(exc).lower()
    if "model" in msg and "not found" in msg:
        return True
    if hasattr(exc, "status_code") and getattr(exc, "status_code") == 404:
        return True
    if hasattr(exc, "response") and getattr(exc.response, "status_code", None) == 404:
        return True
    return False


MIN_OUTPUT_LENGTH_RATIO = 0.8  # Reject if AI returns less than 80% of original length

# JSON output: % calculated ONLY on main_text; glossary generated separately
DIGLOT_SYSTEM_PROMPT = """You are a 'Diglot Weave' teacher. You MUST respond with ONLY valid JSON.

1. MAIN TEXT: Translate the BODY text according to the target percentage ({target_percent}%). The percentage applies ONLY to the main story text — do NOT include the glossary in the word count or percentage calculation.
   - Replace approximately {target_words_count} words in the main text. Text length: {total_words} words (count only the body).
   - Preserve all HTML in main_text (e.g. <p>, <div>, <br>). Wrap every English word in <b>tags</b> (e.g., <b>word</b>).
   - DISTRIBUTION: Scatter words RANDOMLY. FULL TEXT ONLY: return the ENTIRE body. No skipping or summarizing.
2. GLOSSARY: Build a separate list from the English words you replaced in main_text: 10-15 unique, interesting words. Format each entry as "word: перевод". Steps: (a) Collect the replaced words from main_text; (b) Deduplicate; (c) Sort alphabetically; (d) Do not repeat words from previous chapters' glossaries. Skip very simple words (house, man, go). Glossary entries must match how words appear in main_text.
3. PROPER NOUNS: Keep names, cities, places in original Russian Cyrillic (e.g. Амалия, Урсула, Лондон). Never translate them.
{previous_vocab_instruction}
{already_glossaried_instruction}

OUTPUT: Respond with ONLY this JSON (no markdown, no code fence, no commentary):
{"main_text": "<p>...</p>", "glossary": ["word: перевод", "word2: перевод", ...]}"""

# For .txt: main_text is plain (no HTML); glossary separate
DIGLOT_SYSTEM_PROMPT_TXT = """You are a 'Diglot Weave' teacher for .txt output. You MUST respond with ONLY valid JSON.

1. MAIN TEXT: Translate the BODY text according to the target percentage ({target_percent}%). The percentage applies ONLY to the main story text — do NOT include the glossary in the word count or percentage calculation.
   - Replace approximately {target_words_count} words. Text length: {total_words} words (body only).
   - STORY TEXT: Do NOT use any HTML, UPPERCASE, or asterisks. Integrate English as plain, seamless text. Reader distinguishes by language only.
   - DISTRIBUTION: Scatter words RANDOMLY. FULL TEXT ONLY: return the ENTIRE story. No skipping or summarizing.
2. GLOSSARY: Build a separate list from the English words you replaced in main_text: 10-15 unique entries. Format each as "word: перевод". Steps: (a) Collect replaced words from main_text; (b) Deduplicate; (c) Sort alphabetically; (d) Do not repeat words from previous chapters.
3. PROPER NOUNS: Keep names, cities, places in Russian Cyrillic (e.g. Амалия, Лондон).
{previous_vocab_instruction}
{already_glossaried_instruction}

OUTPUT: Respond with ONLY this JSON (no markdown, no code fence, no commentary):
{"main_text": "plain story text here...", "glossary": ["word: перевод", "word2: перевод", ...]}"""


class OpenRouterTranslator:
    def __init__(self, model: Optional[str] = None) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Configure it as an environment variable."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model or os.getenv("OPENROUTER_MODEL", FALLBACK_MODEL)

    def translate_words_ru_to_en(self, words: List[str]) -> Dict[str, str]:
        """
        Translate a list of Russian words to English.
        Returns mapping {ru: en}.
        """
        if not words:
            return {}

        # Dedupe but preserve stable order
        seen = set()
        uniq: List[str] = []
        for w in words:
            if w not in seen:
                uniq.append(w)
                seen.add(w)

        system = (
            "You are a precise translation engine. "
            "Translate Russian tokens into natural English single-word equivalents when possible. "
            "Return STRICT JSON only: an object mapping each original token to its translation. "
            "No extra keys, no commentary, no Markdown."
        )
        user = (
            "Translate the following Russian tokens to English.\n\n"
            f"Tokens: {json.dumps(uniq, ensure_ascii=False)}\n\n"
            "Return JSON object like: {\"привет\": \"hello\"}"
        )

        model_id = self.model
        print(f"DEBUG: Using model slug '{model_id}'")
        try:
            resp = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
        except Exception as e:
            if _is_model_not_found_error(e):
                print(f"DEBUG: Retrying with fallback '{FALLBACK_MODEL}'")
                resp = self.client.chat.completions.create(
                    model=FALLBACK_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.2,
                )
            else:
                raise

        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except Exception:
            # Best-effort recovery: try to extract JSON substring
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(content[start : end + 1])
            else:
                raise

        out: Dict[str, str] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str) and k.strip():
                    out[k] = v.strip()

        # Ensure every requested token exists in mapping (fallback = original)
        for w in uniq:
            if w not in out:
                out[w] = w
        return out

    def translate_words_in_batches(
        self, words: Iterable[str], batch_size: int = 60
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        buf: List[str] = []
        for w in words:
            if w in mapping:
                continue
            buf.append(w)
            if len(buf) >= batch_size:
                mapping.update(self.translate_words_ru_to_en(buf))
                buf = []
        if buf:
            mapping.update(self.translate_words_ru_to_en(buf))
        return mapping

    TRIAL_SYSTEM_PROMPT_40 = (
        "You are a 'Diglot Weave' generator. This is a FREE SAMPLE. "
        "Your task is to replace exactly 40% of the words in this text with English. "
        "Rules:\n"
        "1. Use common English words.\n"
        "2. Bold every English word: <b>word</b>.\n"
        "3. Keep names in Cyrillic.\n"
        "4. Make the translation very obvious and frequent.\n"
        "Return ONLY the transformed text. No glossary. Preserve structure and line breaks."
    )

    TRIAL_SYSTEM_PROMPT_SIMPLE = (
        "Replace about 40% of Russian words with English. Wrap each English word in <b>word</b>. "
        "Keep Cyrillic names unchanged. Output only the text."
    )

    TRIAL_SYSTEM_PROMPT_PLAIN = (
        "Replace about 40% of Russian words with English. Do NOT use any HTML tags (like <b>), UPPERCASE, or special symbols (like asterisks) to highlight English words. "
        "Integrate English words into the Russian sentences as plain, seamless text. The reader should distinguish them only by the language itself. "
        "Keep Cyrillic names unchanged (e.g. Амалия, Лондон). Output only the transformed text."
    )

    async def translate_simple(
        self,
        snippet_text: str,
        target_percent: int = 40,
        model_id: Optional[str] = None,
        highlight_style: str = "BOLD_TAGS",
    ) -> str:
        """Standalone trial. highlight_style: 'PLAIN' for .txt (no highlighting), 'BOLD_TAGS' for .epub/.fb2. Returns UTF-8 safe str."""
        if not (snippet_text or snippet_text.strip()):
            return snippet_text or ""
        text = snippet_text.strip()
        model = model_id or self.model
        use_plain = (highlight_style or "BOLD_TAGS").upper() in ("UPPERCASE", "PLAIN")
        system_prompt = self.TRIAL_SYSTEM_PROMPT_PLAIN if use_plain else self.TRIAL_SYSTEM_PROMPT_40
        try:
            print(f"DEBUG: Using model slug '{model}' for trial (target_percent={target_percent}, highlight_style={highlight_style})")
            resp = await self.async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                return text
            if content.startswith("```"):
                content = re.sub(r"^```(?:html)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            content = content.strip()
            if use_plain:
                # For TXT: accept plain text (optional retry if no Latin/English)
                if re.search(r"[a-zA-Z]{2,}", content):
                    return content
                print("DEBUG: Trial PLAIN result has no obvious English words, retrying with gpt-4o-mini")
                try:
                    resp2 = await self.async_client.chat.completions.create(
                        model=FALLBACK_MODEL,
                        messages=[
                            {"role": "system", "content": self.TRIAL_SYSTEM_PROMPT_PLAIN},
                            {"role": "user", "content": text},
                        ],
                        temperature=0.3,
                    )
                    content2 = (resp2.choices[0].message.content or "").strip()
                    if content2:
                        return content2.strip()
                except Exception as e2:
                    logger.warning("Trial retry failed: %s", e2)
                return content
            if "<b>" not in content and "<b " not in content:
                print("DEBUG: Trial result has no <b> tags, retrying with gpt-4o-mini")
                try:
                    resp2 = await self.async_client.chat.completions.create(
                        model=FALLBACK_MODEL,
                        messages=[
                            {"role": "system", "content": self.TRIAL_SYSTEM_PROMPT_SIMPLE},
                            {"role": "user", "content": text},
                        ],
                        temperature=0.3,
                    )
                    content2 = (resp2.choices[0].message.content or "").strip()
                    if content2 and ("<b>" in content2 or "<b " in content2):
                        return content2.strip()
                except Exception as e2:
                    logger.warning("Trial retry failed: %s", e2)
            return content
        except Exception as e:
            logger.warning("Trial translate_simple failed: %s", e)
            return text

    async def _call_chapter_once(
        self, model_id: str, system: str, user: str
    ) -> Optional[str]:
        """One API call for a chapter. Returns parsed HTML or None on failure."""
        try:
            print(f"DEBUG: Using model slug '{model_id}'")
            resp = await self.async_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
            )
        except Exception as e:
            if _is_retryable_error(e):
                logger.debug("Chapter call failed (retryable): %s", e)
                return None
            raise
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return None
        if content.startswith("```"):
            content = re.sub(r"^```(?:html)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        return content.strip()

    async def diglot_weave_chapter(
        self,
        html: str,
        total_words: int,
        target_words_count: int,
        ratio: float = 1.0,
        target_percent: float = 100.0,
        previous_vocab: Optional[Dict[str, str]] = None,
        already_glossaried: Optional[set] = None,
        use_uppercase: bool = False,
    ) -> str:
        """
        Process a full chapter with Diglot Weave.
        """
        if target_words_count <= 0:
            return html

        if previous_vocab and len(previous_vocab) > 0:
            prev_list = " ".join(f"{ru}→{en}" for ru, en in list(previous_vocab.items())[:80])
            previous_vocab_instruction = (
                f"\n\nPREVIOUS VOCABULARY (use these English words when you see these Russian): {prev_list}"
            )
        else:
            previous_vocab_instruction = ""

        if already_glossaried and len(already_glossaried) > 0:
            already_list = ", ".join(sorted(already_glossaried)[:100])
            already_glossaried_instruction = (
                f"\n\nDo NOT reuse these in glossary: {already_list}."
            )
        else:
            already_glossaried_instruction = ""

        if use_uppercase:
            system = DIGLOT_SYSTEM_PROMPT_TXT.format(
                total_words=total_words,
                target_words_count=target_words_count,
                target_percent=int(round(target_percent)),
                previous_vocab_instruction=previous_vocab_instruction,
                already_glossaried_instruction=already_glossaried_instruction,
            )
            user = (
                f"Process the following text. Replace approximately {target_words_count} words in the BODY only. "
                "Output JSON with main_text (plain story, no HTML/UPPERCASE/asterisks) and glossary (list of 'word: перевод').\n\n"
                + html
            )
        else:
            system = DIGLOT_SYSTEM_PROMPT.format(
                total_words=total_words,
                target_words_count=target_words_count,
                target_percent=int(round(target_percent)),
                previous_vocab_instruction=previous_vocab_instruction,
                already_glossaried_instruction=already_glossaried_instruction,
            )
            if ratio < 0.30:
                system += "\n\n**Low immersion:** With this low percentage, prefer replacing nouns and objects so the sentence logic stays clear. Avoid 'broken English' (e.g. 'I not proud that'). Keep reading flow natural."
            user = (
                "Process the following HTML. Replace approximately "
                f"{target_words_count} words in the BODY only. Reply with ONLY the JSON object (main_text + glossary).\n\n"
                + html
            )

        def _try_parse_and_assemble(out: Optional[str]) -> Optional[str]:
            """Parse JSON; validate % on main_text only; return assembled chapter or None."""
            if not out:
                return None
            data = _parse_diglot_json(out)
            if not data:
                return None
            main_text, glossary = data
            if not _output_length_ok(html, main_text):
                logger.warning(
                    "Chapter main_text too short (<%s%% of original), retrying or using fallback",
                    int(MIN_OUTPUT_LENGTH_RATIO * 100),
                )
                return None
            return _assemble_chapter_from_json(main_text, glossary, use_uppercase)

        # Tier 1: selected model, 3 attempts, 5s delay
        for attempt in range(TIER1_ATTEMPTS):
            try:
                out = await self._call_chapter_once(self.model, system, user)
                assembled = _try_parse_and_assemble(out)
                if assembled:
                    return assembled
            except Exception as e:
                if not _is_retryable_error(e):
                    raise
                logger.debug("Tier 1 attempt %s: %s", attempt + 1, e)
            if attempt < TIER1_ATTEMPTS - 1:
                await asyncio.sleep(TIER1_DELAY_SEC)

        # Tier 2: gpt-4o-mini (first fallback), 2 attempts
        for attempt in range(TIER2_ATTEMPTS):
            try:
                out = await self._call_chapter_once(TIER2_MODEL, system, user)
                assembled = _try_parse_and_assemble(out)
                if assembled:
                    return assembled
            except Exception as e:
                if not _is_retryable_error(e):
                    raise
                logger.debug("Tier 2 attempt %s: %s", attempt + 1, e)

        # Tier 3: gemini-2.0-flash (second fallback), 2 attempts
        for attempt in range(TIER3_ATTEMPTS):
            try:
                out = await self._call_chapter_once(TIER3_MODEL, system, user)
                assembled = _try_parse_and_assemble(out)
                if assembled:
                    return assembled
            except Exception as e:
                if not _is_retryable_error(e):
                    raise
                logger.debug("Tier 3 attempt %s: %s", attempt + 1, e)

        # Ultimate safety: return original
        return html

