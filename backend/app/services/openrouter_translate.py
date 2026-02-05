import json
import os
import re
from typing import Dict, Iterable, List, Optional

from openai import AsyncOpenAI, OpenAI

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
    def __init__(self) -> None:
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
        self.model = os.getenv("OPENROUTER_MODEL", "google/gemini-flash-1.5")

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

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )

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
        Process a full chapter HTML with Diglot Weave (non-blocking). Returns HTML:
        [GLOSSARY] + <hr /> + [TRANSLATED TEXT]. Smart glossary: 10-15 sophisticated words only; no repeats from previous chapters.
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

        resp = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return html

        # Strip markdown code block if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:html)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        return content.strip()

