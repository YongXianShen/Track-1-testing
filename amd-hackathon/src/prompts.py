"""Compact, category-specific prompts and answer validation for V18."""
from __future__ import annotations

import re

_SPEC = {
    "factual": ("Answer all parts accurately and concisely (maximum 100 words).", 160),
    "math": ("Solve accurately. Show one compact calculation, then `Answer: <final>` with units.", 135),
    "sentiment": ("Return Positive, Negative, Neutral, or Mixed, then one brief reason.", 36),
    "summary": ("Output only the summary. Obey every exact length and format constraint.", 145),
    "ner": ("List only requested entities, preserving text, as `Entity — TYPE`, one per line.", 115),
    "debug": ("State the bug briefly, then give minimal corrected runnable code.", 310),
    "logic": ("Apply every constraint. Give one compact deduction, then `Answer: <final>`.", 135),
    "codegen": ("Return only minimal correct self-contained code. Handle edge cases.", 350),
}

_NUMBER_WORDS = {
    "one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _requested_count(prompt: str, unit: str) -> int | None:
    match = re.search(
        rf"\b(one|single|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*[- ]?{unit}s?\b",
        prompt,
        re.I,
    )
    if not match:
        return None
    value = match.group(1).lower()
    return int(value) if value.isdigit() else _NUMBER_WORDS.get(value)


def _summary_cap(prompt: str, default: int) -> int:
    words = _requested_count(prompt, "word")
    if words is not None:
        return max(20, min(default, words * 3 + 12))
    sentences = _requested_count(prompt, "sentence") or _requested_count(prompt, "line")
    if sentences is not None:
        return max(42, min(default, sentences * 62))
    bullets = _requested_count(prompt, "bullet")
    if bullets is not None:
        return max(50, min(default, bullets * 48))
    return default


def _code_cap(prompt: str, default: int) -> int:
    if re.search(r"\b(?:full program|application|API|endpoint|class|multiple functions|complete implementation)\b", prompt, re.I):
        return default
    if re.search(r"\b(?:function|method|regex|query)\b", prompt, re.I):
        return min(default, 275 if "debug" not in prompt.lower() else 290)
    return default


def render(category: str, prompt: str) -> tuple[list[dict[str, str]], int]:
    instruction, cap = _SPEC.get(category, _SPEC["factual"])
    if category == "summary":
        cap = _summary_cap(prompt, cap)
    elif category in {"debug", "codegen"}:
        cap = _code_cap(prompt, cap)
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ], int(cap)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S).strip()


def _exact_word_count(prompt: str) -> int | None:
    if not re.search(r"\b(?:exactly|in)\b.{0,15}\b(?:word|words)\b", prompt, re.I):
        return None
    return _requested_count(prompt, "word")


def _requires_one_sentence(prompt: str) -> bool:
    return bool(re.search(r"\b(?:exactly\s+)?(?:one|single|1)\s*[- ]?sentence\b", prompt, re.I))


def _enforce_summary(text: str, prompt: str) -> str:
    count = _exact_word_count(prompt)
    if count is not None and count > 0:
        words = text.split()
        if len(words) > count:
            text = " ".join(words[:count]).rstrip(" ,;:")
            if text and text[-1] not in ".!?":
                text += "."
    if _requires_one_sentence(prompt):
        # Preserve content while removing extra sentence boundaries.
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        if len(parts) > 1:
            cleaned = [p.rstrip(".!?") for p in parts]
            text = "; ".join(cleaned).rstrip(" ;") + "."
    return text


def postprocess(category: str, text: str, prompt: str = "") -> str:
    text = _strip_think(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if category == "summary":
        text = _enforce_summary(text, prompt)
    if category == "sentiment":
        low = text.lower()
        for label in ("mixed", "positive", "negative", "neutral"):
            if re.search(rf"\b{label}\b", low):
                compact = re.sub(r"\s+", " ", text).strip()
                return compact[:150]
    if category == "codegen":
        match = re.fullmatch(r"```[a-zA-Z0-9_+\-.#]*\s*\n(.*?)\n```", text, flags=re.S)
        if match:
            return match.group(1).strip()
    return text


def is_usable(category: str, answer: str, prompt: str) -> bool:
    text = (answer or "").strip()
    if not text or re.search(r"\b(?:cannot answer|unable to|no answer)\b", text, re.I):
        return False
    if category == "sentiment":
        return bool(re.search(r"\b(?:positive|negative|neutral|mixed)\b", text, re.I))
    if category == "summary":
        exact_words = _exact_word_count(prompt)
        if exact_words is not None and len(text.split()) != exact_words:
            return False
        if _requires_one_sentence(prompt):
            sentences = [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
            if len(sentences) != 1:
                return False
    if category in {"debug", "codegen"}:
        return bool(re.search(r"```|\bdef\s+\w+|\bclass\s+\w+|=>|\breturn\b|#include|console\.log|System\.out|\bSELECT\b", text, re.I))
    return True
