import json
import os
import re
from typing import Dict, Iterable, List

from openai import OpenAI

DIGLOT_SYSTEM_PROMPT = """You are a 'Diglot Weave' linguistic expert. Your goal is to help a Russian reader learn English via gradual immersion.

Rules for this task:
1. **Scattered Distribution:** DO NOT translate chunks or full phrases at the beginning of sentences. You must scatter English words randomly like 'salt and pepper' throughout the entire text (beginning, middle, and end of paragraphs).
2. **Individual Word Focus:** Replace individual Russian words with their English equivalents. Avoid replacing more than 2-3 words in a row unless necessary for a specific idiom.
3. **Target Count:** In this text of approximately {total_words} words, you MUST replace EXACTLY {target_words_count} words with English.
4. **Variety:** Target various parts of speech—nouns, verbs, and adjectives—randomly across the text.
5. **Formatting:** Every single English word or short phrase MUST be wrapped in <b> tags. Example: <b>word</b>.
6. **Grammar Integrity:** The English words should be in their dictionary form (e.g., "He opened the <b>door</b>") or lightly adapted so the Russian reader can still follow the sentence structure easily.
7. **HTML Preservation:** Keep all original HTML tags (like <p>, <div>, <br>) exactly where they are.

Respond with ONLY the processed HTML document. No explanations, no markdown, no code block—just the raw HTML."""


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

    def diglot_weave_chapter(
        self, html: str, total_words: int, target_words_count: int, ratio: float = 1.0
    ) -> str:
        """
        Process a full chapter HTML with Diglot Weave: replace exactly target_words_count
        Russian words with English (wrapped in <b>), scattered randomly. Returns processed HTML only.
        """
        if target_words_count <= 0:
            return html

        system = DIGLOT_SYSTEM_PROMPT.format(
            total_words=total_words,
            target_words_count=target_words_count,
        )
        if ratio < 0.30:
            system += "\n\n**Low immersion:** With this low percentage, prefer replacing nouns and objects so the sentence logic stays clear. Avoid 'broken English' (e.g. 'I not proud that'). Keep reading flow natural."

        user = (
            "Process the following HTML. Replace exactly "
            f"{target_words_count} Russian words with English equivalents, scattered randomly. "
            "Wrap every English word in <b> tags. Preserve all HTML structure. "
            "Reply with ONLY the processed HTML, nothing else.\n\n"
            + html
        )

        resp = self.client.chat.completions.create(
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

