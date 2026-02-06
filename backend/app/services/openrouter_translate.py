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
        logger.warning("Diglot JSON parse failed: %s. Will retry or fallback.", e)
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


def _count_english_percent(main_text: str, use_uppercase: bool) -> Optional[float]:
    """
    Count percentage of English/highlighted words in main_text only.
    EPUB: count <b>...</b> tags. TXT: count Latin-letter words (2+ chars).
    Returns 0-100 or None if no words.
    """
    if not main_text or not main_text.strip():
        return None
    # Total word-like tokens (letters/numbers)
    all_tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", main_text)
    total = len(all_tokens)
    if total == 0:
        return None
    if use_uppercase:
        # Plain text: count Latin-letter words (2+ chars)
        english = len([t for t in all_tokens if len(t) >= 2 and re.match(r"^[a-zA-Z]+$", t)])
    else:
        # HTML: count <b>...</b> content as English words
        english = len(re.findall(r"<b[^>]*>([^<]+)</b>", main_text, re.IGNORECASE))
    return 100.0 * english / total if total else None


def _assemble_chapter_from_json(main_text: str, glossary: List[str], _use_uppercase: bool) -> str:
    """
    Assemble final chapter: main_text + "\n\nGlossary:\n" + "\n".join(glossary).
    Glossary is kept separate and appended as plain text.
    """
    if not glossary:
        return main_text
    return main_text + "\n\nGlossary:\n" + "\n".join(glossary)


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

# JSON output: % calculated ONLY on main_text; glossary generated separately. Use {{ }} for literal braces in format().
DIGLOT_SYSTEM_PROMPT = """You are a 'Diglot Weave' teacher. You MUST respond with ONLY valid JSON.

Chain-of-Thought (follow in order):
1. Read the input HTML body. Count total words in the BODY only (ignore any existing glossary).
2. Compute target: exactly {target_percent}% of those words must become English in main_text. That is about {target_words_count} words. Text length: {total_words} words (body only).
3. MAIN TEXT: Replace exactly that many words in the BODY. Percentage applies ONLY to main_text — do NOT include the glossary in word count or % calculation. Preserve all HTML (e.g. <p>, <div>, <br>). Wrap every English word in <b>tags</b> (e.g., <b>word</b>). Scatter words RANDOMLY. Return the ENTIRE body; no skipping or summarizing.
4. GLOSSARY: From the English words you put in main_text, build a SEPARATE list: 10-15 unique entries. Format each as "word: перевод". (a) Collect replaced words from main_text; (b) Deduplicate; (c) Sort alphabetically; (d) Do not repeat words from previous chapters. Skip very simple words (house, man, go).
5. PROPER NOUNS: Keep names, cities, places in original Russian Cyrillic (e.g. Амалия, Урсула, Лондон). Never translate them.
6. Output ONLY the JSON object below. No markdown, no code fence, no commentary.
{previous_vocab_instruction}
{already_glossaried_instruction}

REMINDER: Respond with ONLY this JSON (literal braces, no extra keys):
{{"main_text": "<p>...</p>", "glossary": ["word: перевод", "word2: перевод", ...]}}"""

# For .txt: main_text is plain (no HTML); glossary separate
DIGLOT_SYSTEM_PROMPT_TXT = """You are a 'Diglot Weave' teacher for .txt output. You MUST respond with ONLY valid JSON.

Chain-of-Thought (follow in order):
1. Read the input text. Count total words in the BODY only.
2. Compute target: exactly {target_percent}% of those words must become English in main_text. That is about {target_words_count} words. Text length: {total_words} words (body only).
3. MAIN TEXT: Replace exactly that many words. Percentage applies ONLY to main_text — do NOT include the glossary in word count or % calculation. Do NOT use HTML, UPPERCASE, or asterisks. Integrate English as plain, seamless text. Scatter RANDOMLY. Return the ENTIRE story; no skipping or summarizing.
4. GLOSSARY: From the English words you put in main_text, build a SEPARATE list: 10-15 unique entries. Format each as "word: перевод". (a) Collect replaced words; (b) Deduplicate; (c) Sort alphabetically; (d) Do not repeat words from previous chapters.
5. PROPER NOUNS: Keep names, cities, places in Russian Cyrillic (e.g. Амалия, Лондон).
6. Output ONLY the JSON object below. No markdown, no code fence, no commentary.
{previous_vocab_instruction}
{already_glossaried_instruction}

REMINDER: Respond with ONLY this JSON (literal braces, no extra keys):
{{"main_text": "plain story text here...", "glossary": ["word: перевод", "word2: перевод", ...]}}"""


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

        JSON_RETRIES = 2  # up to 2 retries on JSON decode failure (3 total tries)
        PCT_RETRIES = 2  # up to 2 retries when % off by >10%

        async def _try_one(
            model_id: str, sys: str, user_msg: str
        ) -> Optional[Tuple[str, List[str]]]:
            """Call API, parse JSON (retry up to JSON_RETRIES on decode fail). Return (main_text, glossary) or None."""
            for json_try in range(JSON_RETRIES + 1):
                try:
                    out = await self._call_chapter_once(model_id, sys, user_msg)
                except Exception as e:
                    if not _is_retryable_error(e):
                        raise
                    if json_try < JSON_RETRIES:
                        logger.warning("Chapter call failed, retrying for JSON (%s/%s)", json_try + 1, JSON_RETRIES)
                        continue
                    raise
                data = _parse_diglot_json(out)
                if data:
                    return data
                if json_try < JSON_RETRIES:
                    logger.warning("Invalid JSON, retrying (%s/%s)", json_try + 1, JSON_RETRIES)
            return None

        def _check_and_assemble(
            main_text: str, glossary: List[str], target_pct: float
        ) -> Optional[str]:
            """Validate length and (optionally) %; return assembled chapter or None."""
            if not _output_length_ok(html, main_text):
                logger.warning(
                    "Chapter main_text too short (<%s%% of original), retrying or using fallback",
                    int(MIN_OUTPUT_LENGTH_RATIO * 100),
                )
                return None
            actual_pct = _count_english_percent(main_text, use_uppercase)
            if actual_pct is not None and abs(actual_pct - target_pct) > 10:
                return None  # caller will retry with % addendum
            return _assemble_chapter_from_json(main_text, glossary, use_uppercase)

        target_pct = float(target_percent)

        # Tier 1: selected model, 3 attempts, 5s delay
        for attempt in range(TIER1_ATTEMPTS):
            try:
                current_user = user
                data = await _try_one(self.model, system, current_user)
                if not data:
                    continue
                main_text, glossary = data
                assembled = _check_and_assemble(main_text, glossary, target_pct)
                if assembled:
                    return assembled
                for pct_try in range(PCT_RETRIES):
                    current_user = (
                        user
                        + "\n\n[CRITICAL] Last response had "
                        + str(round(_count_english_percent(main_text, use_uppercase) or 0))
                        + "% English in main_text. You MUST achieve exactly "
                        + str(int(round(target_pct)))
                        + "% in main_text."
                    )
                    data = await _try_one(self.model, system, current_user)
                    if not data:
                        break
                    main_text, glossary = data
                    assembled = _check_and_assemble(main_text, glossary, target_pct)
                    if assembled:
                        return assembled
                if assembled is None and main_text:
                    logger.warning(
                        "English %% in main_text off by >10%% after retries (target %s%%)",
                        int(round(target_pct)),
                    )
            except Exception as e:
                if not _is_retryable_error(e):
                    raise
                logger.debug("Tier 1 attempt %s: %s", attempt + 1, e)
            if attempt < TIER1_ATTEMPTS - 1:
                await asyncio.sleep(TIER1_DELAY_SEC)

        # Tier 2: gpt-4o-mini (first fallback), 2 attempts
        for attempt in range(TIER2_ATTEMPTS):
            try:
                current_user = user
                data = await _try_one(TIER2_MODEL, system, current_user)
                if not data:
                    continue
                main_text, glossary = data
                assembled = _check_and_assemble(main_text, glossary, target_pct)
                if assembled:
                    return assembled
                for pct_try in range(PCT_RETRIES):
                    current_user = (
                        user
                        + "\n\n[CRITICAL] Last response had "
                        + str(round(_count_english_percent(main_text, use_uppercase) or 0))
                        + "% English in main_text. You MUST achieve exactly "
                        + str(int(round(target_pct)))
                        + "% in main_text."
                    )
                    data = await _try_one(TIER2_MODEL, system, current_user)
                    if not data:
                        break
                    main_text, glossary = data
                    assembled = _check_and_assemble(main_text, glossary, target_pct)
                    if assembled:
                        return assembled
            except Exception as e:
                if not _is_retryable_error(e):
                    raise
                logger.debug("Tier 2 attempt %s: %s", attempt + 1, e)

        # Tier 3: gemini-2.0-flash (second fallback), 2 attempts
        for attempt in range(TIER3_ATTEMPTS):
            try:
                current_user = user
                data = await _try_one(TIER3_MODEL, system, current_user)
                if not data:
                    continue
                main_text, glossary = data
                assembled = _check_and_assemble(main_text, glossary, target_pct)
                if assembled:
                    return assembled
                for pct_try in range(PCT_RETRIES):
                    current_user = (
                        user
                        + "\n\n[CRITICAL] Last response had "
                        + str(round(_count_english_percent(main_text, use_uppercase) or 0))
                        + "% English in main_text. You MUST achieve exactly "
                        + str(int(round(target_pct)))
                        + "% in main_text."
                    )
                    data = await _try_one(TIER3_MODEL, system, current_user)
                    if not data:
                        break
                    main_text, glossary = data
                    assembled = _check_and_assemble(main_text, glossary, target_pct)
                    if assembled:
                        return assembled
            except Exception as e:
                if not _is_retryable_error(e):
                    raise
                logger.debug("Tier 3 attempt %s: %s", attempt + 1, e)

        # Ultimate safety: return original
        return html

