"""Scoring-aware compact prompts for Track 1 V17.4.

The wording follows the public judging principles: completeness, exact format,
and concise direct answers. Model routing and paid-deployment behavior are
unchanged from the proven V17.2 build.
"""
from __future__ import annotations

import re

_SPEC = {
    "factual": ("Answer all parts; include contrasts, reasons, and uses. ≤120 words.", 180),
    "math": ("Give all values with brief arithmetic; end 'Answer: <final>' with units.", 150),
    "sentiment": ("One sentence: label and reason; if mixed, mention both sides.", 50),
    "summary": ("Only the summary; exact format/limits; retain all major themes and both sides when present.", 150),
    "ner": ("All distinct entities, exact spans: 'Entity — PERSON/ORGANIZATION/LOCATION/DATE'.", 120),
    "debug": ("Name the bug briefly, then give minimal corrected runnable code.", 390),
    "logic": ("Use every constraint; ≤2 deductions; end 'Answer: <final>'.", 160),
    "codegen": ("Only minimal correct self-contained code; handle duplicates and edge cases.", 400),
}


def _requested_count(prompt: str, unit: str) -> int | None:
    words = {
        "one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    m = re.search(rf"\b(one|single|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+{unit}s?\b", prompt, re.I)
    if not m:
        return None
    return words.get(m.group(1).lower(), int(m.group(1)) if m.group(1).isdigit() else None)


def _per_item_word_limit(prompt: str) -> int | None:
    m = re.search(r"(?:each|per)\s+(?:bullet|point|line|sentence)\D{0,25}(?:no (?:longer|more) than|at most|maximum of|under)\s+(\d+)\s+words?", prompt, re.I)
    if not m:
        m = re.search(r"(?:no (?:longer|more) than|at most|maximum of|under)\s+(\d+)\s+words?\s+(?:each|per)\s+(?:bullet|point|line|sentence)", prompt, re.I)
    return int(m.group(1)) if m else None


def _summary_cap(prompt: str, default: int) -> int:
    # Check bullets first so a per-bullet word limit is not mistaken for a total
    # word-count request.
    bullet_count = _requested_count(prompt, "bullet") or _requested_count(prompt, "point")
    if bullet_count is not None:
        per_item = _per_item_word_limit(prompt) or 18
        return max(72, min(default, bullet_count * (per_item * 2 + 6)))
    sentence_count = _requested_count(prompt, "sentence") or _requested_count(prompt, "line")
    if sentence_count is not None:
        return max(44, min(default, sentence_count * 58))
    word_count = _requested_count(prompt, "word")
    if word_count is not None:
        return max(24, min(default, word_count * 3 + 14))
    return default


def _code_cap(prompt: str, default: int) -> int:
    if re.search(r"\b(?:full program|application|API|endpoint|class|multiple functions|complete implementation)\b", prompt, re.I):
        return default
    if re.search(r"\bfunction\b|\bmethod\b|\bregex\b|\bquery\b", prompt, re.I):
        return min(default, 340)
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


def postprocess(category: str, text: str) -> str:
    text = _strip_think(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if category == "sentiment":
        low = text.lower()
        for label in ("mixed", "positive", "negative", "neutral"):
            if re.search(rf"\b{label}\b", low):
                return re.sub(r"\s+", " ", text).strip()[:220]
    if category == "codegen":
        m = re.fullmatch(r"```[a-zA-Z0-9_+\-.#]*\s*\n(.*?)\n```", text, flags=re.S)
        if m:
            return m.group(1).strip()
    return text
