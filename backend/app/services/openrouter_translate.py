import json
import os
import re
from typing import Dict, Iterable, List, Optional

from openai import AsyncOpenAI, OpenAI

DIGLOT_SYSTEM_PROMPT = """You are a Diglot Weave generator.
For this segment, your target is {target_percent}% English words.

Rules:
1. Replace approximately {target_words_count} words.
2. DISTRIBUTION: Scatter words RANDOMLY. Do not translate only the beginning of sentences.
3. FILTERS: Never translate names (Урсула, Амалия, etc.) or places.
4. TYPES: Focus on common nouns, verbs, and adjectives.
5. FORMAT: Wrap every English word in <b>tags</b>. Example: <b>carriage</b>.
6. NO REPETITION: Use unique words for translation within this segment.

OUTPUT FORMAT (strict):
Before the translated chapter text, generate a "Chapter Vocabulary" section.
- Include ONLY unique words that you actually translated in this segment. If you translated no words, omit the glossary.
- Format exactly as follows (use standard HTML so it looks good in any EPUB reader):

<h3>Vocabulary for this Chapter</h3>
<ul>
<li><b>English Word</b> — Russian Translation</li>
<li><b>another</b> — другой</li>
</ul>
<hr />

Then provide the translated chapter text (with <b>English</b> words in the body). The glossary must match the translations used in the text (consistency).
{previous_vocab_instruction}

Text length: {total_words} words. Preserve all HTML tags (e.g. <p>, <div>, <br>). Respond with ONLY the full HTML document (glossary + <hr /> + chapter). No markdown."""


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
    ) -> str:
        """
        Process a full chapter HTML with Diglot Weave (non-blocking). Returns HTML with
        glossary at top (Vocabulary for this Chapter) then <hr /> then translated body.
        previous_vocab: Russian -> English from earlier chapters; use same translations and focus glossary on NEW words.
        """
        if target_words_count <= 0:
            return html

        if previous_vocab and len(previous_vocab) > 0:
            prev_list = " ".join(f"{ru}→{en}" for ru, en in list(previous_vocab.items())[:80])
            previous_vocab_instruction = (
                f"\n\nPREVIOUS CHAPTERS VOCABULARY (use these same English words when you see these Russian words): {prev_list}\n"
                "In the glossary for THIS chapter, list mainly NEW words you translate here. You may add a short 'Review' subsection with a few words from the list above."
            )
        else:
            previous_vocab_instruction = ""

        system = DIGLOT_SYSTEM_PROMPT.format(
            total_words=total_words,
            target_words_count=target_words_count,
            target_percent=int(round(target_percent)),
            previous_vocab_instruction=previous_vocab_instruction,
        )
        if ratio < 0.30:
            system += "\n\n**Low immersion:** With this low percentage, prefer replacing nouns and objects so the sentence logic stays clear. Avoid 'broken English' (e.g. 'I not proud that'). Keep reading flow natural."

        user = (
            "Process the following HTML. Add a Chapter Vocabulary section at the top (see format in instructions), then <hr />, then the translated text. "
            f"Replace approximately {target_words_count} UNIQUE Russian words with English (wrap in <b>). Use previous vocabulary when the same word appears. "
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

