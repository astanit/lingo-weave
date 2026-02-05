import asyncio
import json
import logging
import os
import re
from typing import Dict, Iterable, List, Optional

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


DIGLOT_SYSTEM_PROMPT = """You are a 'Diglot Weave' teacher.

1. Translate the text according to the target percentage ({target_percent}%).
2. Create a 'New Vocabulary' list for the start of this chapter.
   - Select 10-15 most sophisticated or important English words from your translation. Do NOT include every translated word.
   - FILTERING: Skip very simple words (e.g. house, man, go, good, big) in the glossary even if they appear in the text. Focus on difficult, rare, or meaning-heavy words.
   - Format: <h3>Chapter Vocabulary</h3><ul><li><b>word</b> — перевод</li>...</ul><hr/>
   - CRITICAL: Focus on interesting verbs and adjectives. Skip basic nouns if the list is too long.
   - UNIQUENESS: Do not include words that have already appeared in glossaries of previous chapters. Every glossary should feel like a "New Words" list.
   - If no suitable words (e.g. chapter too short), you may omit the glossary or use fewer items.
3. Ensure the glossary words match exactly how they are used in the text.

Rules for translation:
- DISTRIBUTION: Scatter words RANDOMLY. Do not translate only the beginning of sentences.
- FILTERS: Never translate names (Урсула, Амалия, etc.) or places.
- FORMAT: Wrap every English word in the body in <b>tags</b>. Example: <b>carriage</b>.
- Replace approximately {target_words_count} words. Text length: {total_words} words.
{previous_vocab_instruction}
{already_glossaried_instruction}

ASSEMBLY: Output must be [GLOSSARY] + <hr /> + [TRANSLATED TEXT]. Preserve all HTML (e.g. <p>, <div>, <br>). Respond with ONLY the full HTML. No markdown."""


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
    ) -> str:
        """
        Process a full chapter HTML with Diglot Weave. Triple-tier fallback:
        Tier 1 = selected model (3 attempts, 5s delay), Tier 2 = gpt-4o-mini (2 attempts),
        Tier 3 = gemini-2.0-flash (2 attempts). If all fail, return original HTML.
        """
        if target_words_count <= 0:
            return html

        if previous_vocab and len(previous_vocab) > 0:
            prev_list = " ".join(f"{ru}→{en}" for ru, en in list(previous_vocab.items())[:80])
            previous_vocab_instruction = (
                f"\n\nPREVIOUS CHAPTERS VOCABULARY (use these same English words when you see these Russian words): {prev_list}"
            )
        else:
            previous_vocab_instruction = ""

        if already_glossaried and len(already_glossaried) > 0:
            already_list = ", ".join(sorted(already_glossaried)[:100])
            already_glossaried_instruction = (
                f"\n\nDo NOT include these words in this chapter's glossary (already in previous chapters): {already_list}. "
                "Every glossary should be a 'New Words' list only."
            )
        else:
            already_glossaried_instruction = (
                "\n\nTry to pick words that are unique to this specific context and haven't likely been featured in basic introductory vocabulary."
            )

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
            "Process the following HTML. Add a 'Chapter Vocabulary' (10-15 sophisticated words only, format in instructions), then <hr />, then the translated text. "
            f"Replace approximately {target_words_count} words with English (wrap in <b>). Skip simple words in the glossary. "
            "Reply with ONLY the full HTML (glossary + <hr /> + chapter).\n\n"
            + html
        )

        # Tier 1: selected model, 3 attempts, 5s delay
        for attempt in range(TIER1_ATTEMPTS):
            try:
                out = await self._call_chapter_once(self.model, system, user)
                if out:
                    return out
            except Exception as e:
                if not _is_retryable_error(e):
                    raise
                logger.debug("Tier 1 attempt %s: %s", attempt + 1, e)
            if attempt < TIER1_ATTEMPTS - 1:
                await asyncio.sleep(TIER1_DELAY_SEC)

        # Tier 2: gpt-4o-mini, 2 attempts
        for attempt in range(TIER2_ATTEMPTS):
            try:
                out = await self._call_chapter_once(TIER2_MODEL, system, user)
                if out:
                    return out
            except Exception as e:
                if not _is_retryable_error(e):
                    raise
                logger.debug("Tier 2 attempt %s: %s", attempt + 1, e)

        # Tier 3: gemini-2.0-flash, 2 attempts
        for attempt in range(TIER3_ATTEMPTS):
            try:
                out = await self._call_chapter_once(TIER3_MODEL, system, user)
                if out:
                    return out
            except Exception as e:
                if not _is_retryable_error(e):
                    raise
                logger.debug("Tier 3 attempt %s: %s", attempt + 1, e)

        # Ultimate safety: return original
        return html

