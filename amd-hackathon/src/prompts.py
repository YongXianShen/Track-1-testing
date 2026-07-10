"""Accuracy-preserving category prompts with conservative token caps."""
from __future__ import annotations

import math
import re

BASE = "Answer in English. No preamble. Follow every requested format exactly."

SPEC = {
    "factual": {
        "max_tokens": 220,
        "instruction": BASE + " Answer every part clearly and accurately in under 120 words. Include essential context, not just a bare fact.",
    },
    "math": {
        "max_tokens": 240,
        "instruction": BASE + " Solve carefully. Use at most two short steps, then write 'Answer: <final>' with units when relevant.",
    },
    "sentiment": {
        "max_tokens": 64,
        "instruction": BASE + " Use the label set in the task. Otherwise use Positive, Negative, Neutral, or Mixed. Give the label and one brief justification.",
    },
    "summary": {
        "max_tokens": 180,
        "instruction": BASE + " Output only the summary. Strictly obey sentence, word, bullet, tone, and length constraints. Do not add outside facts.",
    },
    "ner": {
        "max_tokens": 160,
        "instruction": BASE + " Extract all and only the requested named entities, preserving exact text. Follow the task's format; otherwise use one line per entity as 'TYPE: Entity'.",
    },
    "debug": {
        "max_tokens": 500,
        "instruction": BASE + " Identify the bug in one concise sentence, then provide the corrected minimal runnable implementation in the original language.",
    },
    "logic": {
        "max_tokens": 260,
        "instruction": BASE + " Use every constraint. Give at most two short deductions, then write 'Answer: <final>'.",
    },
    "codegen": {
        "max_tokens": 520,
        "instruction": BASE + " Return only correct, minimal, self-contained code in the requested language. Handle stated edge cases. Do not add comments unless requested.",
    },
}


def _adaptive_summary_cap(prompt: str, default: int) -> int:
    low = prompt.lower()
    word_match = re.search(r"(?:exactly|in|under|no more than|at most)\s+(\d+)\s+words?", low)
    if word_match:
        words = max(1, min(int(word_match.group(1)), 300))
        return min(default, max(32, math.ceil(words * 1.7) + 16))

    sentence_match = re.search(r"(?:exactly|in|under|at most)\s+(\d+)\s+sentences?", low)
    if sentence_match:
        sentences = max(1, min(int(sentence_match.group(1)), 8))
        return min(default, 35 + 65 * sentences)

    if re.search(r"\b(?:one|single|1)[ -]?sentence\b", low):
        return min(default, 110)

    bullet_match = re.search(r"(?:exactly|in|at most)\s+(\d+)\s+bullets?", low)
    if bullet_match:
        bullets = max(1, min(int(bullet_match.group(1)), 10))
        return min(default, 30 + 45 * bullets)
    return default


def render(category: str, prompt: str) -> tuple[list[dict[str, str]], int]:
    spec = SPEC.get(category, SPEC["factual"])
    max_tokens = int(spec["max_tokens"])
    if category == "summary":
        max_tokens = _adaptive_summary_cap(prompt, max_tokens)
    return [
        {"role": "system", "content": spec["instruction"]},
        {"role": "user", "content": prompt},
    ], max_tokens


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S).strip()


def postprocess(category: str, text: str) -> str:
    text = _strip_think(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if category == "sentiment":
        low = text.lower()
        for label in ("mixed", "positive", "negative", "neutral"):
            if re.search(rf"\b{label}\b", low):
                one = re.sub(r"\s+", " ", text).strip()
                return one[:190] if len(one) <= 190 else label.capitalize()
    if category == "codegen":
        match = re.fullmatch(r"```[a-zA-Z0-9_+\-.#]*\s*\n(.*?)\n```", text, flags=re.S)
        if match:
            return match.group(1).strip()
    return text
