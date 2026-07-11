"""V17.2 compact prompts.

Routing, models, local solvers, output caps, and fallback behavior are unchanged
from the proven V17. Only repeated instruction text is shortened to reduce scored
input tokens while preserving answer requirements.
"""
from __future__ import annotations

import re

_SPEC = {
    "factual": ("Answer every part accurately and concisely in at most 120 words.", 200),
    "math": ("Solve accurately. Show at most two short steps; end with 'Answer: <final>' and units if needed.", 190),
    "sentiment": ("Use requested labels; otherwise give Positive, Negative, Neutral, or Mixed plus one brief reason.", 48),
    "summary": ("Return only the summary; obey every requested format and length constraint exactly.", 170),
    "ner": ("Return only requested entities, preserving exact text, one per line as 'Entity — TYPE'.", 160),
    "debug": ("Briefly identify the bug, then give minimal corrected runnable code.", 430),
    "logic": ("Apply every constraint. Use at most two short deductions; end with 'Answer: <final>'.", 190),
    "codegen": ("Return only minimal, correct, self-contained code that handles edge cases.", 460),
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


def _summary_cap(prompt: str, default: int) -> int:
    word_count = _requested_count(prompt, "word")
    if word_count is not None:
        return max(24, min(default, word_count * 3 + 16))
    sentence_count = _requested_count(prompt, "sentence") or _requested_count(prompt, "line")
    if sentence_count is not None:
        return max(48, min(default, sentence_count * 70))
    bullet_count = _requested_count(prompt, "bullet")
    if bullet_count is not None:
        return max(60, min(default, bullet_count * 55))
    return default


def _code_cap(prompt: str, default: int) -> int:
    # Simple function tasks are common and should not be allowed to ramble.  Larger
    # artifacts keep the full budget to avoid truncation.
    if re.search(r"\b(?:full program|application|API|endpoint|class|multiple functions|complete implementation)\b", prompt, re.I):
        return default
    if re.search(r"\bfunction\b|\bmethod\b|\bregex\b|\bquery\b", prompt, re.I):
        return min(default, 360)
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
                compact = re.sub(r"\s+", " ", text).strip()
                return compact[:180]
    if category == "codegen":
        m = re.fullmatch(r"```[a-zA-Z0-9_+\-.#]*\s*\n(.*?)\n```", text, flags=re.S)
        if m:
            return m.group(1).strip()
    return text
