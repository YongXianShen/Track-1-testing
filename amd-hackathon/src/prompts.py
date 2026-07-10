"""Compact category prompts and deterministic output cleanup."""
from __future__ import annotations

import math
import re

_INSTRUCTIONS: dict[str, str] = {
    "factual": "Answer every part accurately and concisely. Include every requested name, place, date, quantity, or explanation. No preamble.",
    "math": "Solve accurately. Show one compact calculation, then `Answer: <final>` with any required units. No preamble.",
    "sentiment": "Return `Label — brief justification`. Use the prompt's labels; otherwise Positive, Negative, Neutral, or Mixed.",
    "summary": "Output only the requested summary. Obey exact sentence, word, bullet, length, and style constraints.",
    "ner": "Extract all requested named entities exactly. Output one per line as `Entity — TYPE`; no commentary.",
    "debug": "Briefly identify the bug, then provide corrected runnable code. Preserve the requested language and function signature.",
    "logic": "Use every constraint. Give one compact justification, then `Answer: <final>`.",
    "codegen": "Return only minimal correct code. Preserve the requested language and signature; handle duplicates and edge cases. No markdown unless requested.",
}

_DEFAULT_LIMITS = {
    "factual": 150,
    "math": 170,
    "sentiment": 55,
    "summary": 170,
    "ner": 150,
    "debug": 340,
    "logic": 170,
    "codegen": 430,
}


def _summary_limit(prompt: str) -> int:
    # Keep enough headroom for exact requested outputs while avoiding waste.
    word_match = re.search(r"(?:exactly|in|under|no more than)\s+(\d+)\s+words?", prompt, re.I)
    if word_match:
        words = max(1, int(word_match.group(1)))
        return min(300, max(40, int(math.ceil(words * 1.7)) + 16))
    sentence_match = re.search(r"(?:exactly\s+)?(\d+)\s+sentences?", prompt, re.I)
    if sentence_match:
        return min(260, max(55, int(sentence_match.group(1)) * 70))
    if re.search(r"\b(?:one|single) sentence\b", prompt, re.I):
        return 105
    return _DEFAULT_LIMITS["summary"]


def render(category: str, prompt: str) -> tuple[list[dict[str, str]], int]:
    instruction = _INSTRUCTIONS.get(category, _INSTRUCTIONS["factual"])
    limit = _summary_limit(prompt) if category == "summary" else _DEFAULT_LIMITS.get(category, 150)
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ], limit


def _strip_reasoning(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S)
    cleaned = re.sub(r"<analysis>.*?</analysis>", "", cleaned, flags=re.I | re.S)
    return cleaned.strip()


def postprocess(category: str, text: str) -> str:
    text = _strip_reasoning(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if category == "codegen":
        match = re.fullmatch(r"```[^\n]*\n(.*?)\n```", text, flags=re.S)
        if match:
            return match.group(1).strip()

    if category == "sentiment":
        one_line = re.sub(r"\s+", " ", text).strip()
        # Keep the model's requested label and short justification; don't replace
        # nuanced classifications with a brittle local dictionary.
        return one_line[:260]

    return text
