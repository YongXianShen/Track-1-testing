"""Short category prompts. Aim: pass accuracy gate with low token count."""
from __future__ import annotations

import re

BASE = "Answer in English. No preamble. Follow the requested format exactly."

SPEC = {
    "factual": {
        "max_tokens": 220,
        "instruction": BASE + " Answer all parts clearly in 1-4 concise sentences. If a location/body/date is asked, include it.",
    },
    "math": {
        "max_tokens": 260,
        "instruction": BASE + " Solve carefully. Return at most one short calculation plus 'Answer: <final>'. Keep units if relevant.",
    },
    "sentiment": {
        "max_tokens": 70,
        "instruction": BASE + " Classify as Positive, Negative, Neutral, or Mixed unless the prompt gives another label set. Give label plus a short reason.",
    },
    "summary": {
        "max_tokens": 180,
        "instruction": BASE + " Output only the summary. Strictly obey sentence, word, bullet, length, and style constraints.",
    },
    "ner": {
        "max_tokens": 180,
        "instruction": BASE + " Extract only requested named entities. Preserve exact text. Format: Entity — TYPE. Use PERSON, ORGANIZATION, LOCATION, DATE as applicable.",
    },
    "debug": {
        "max_tokens": 520,
        "instruction": BASE + " State the bug briefly, then give the corrected implementation. Keep code minimal and runnable.",
    },
    "logic": {
        "max_tokens": 280,
        "instruction": BASE + " Use every constraint. Give at most two short deductions, then 'Answer: <final>'.",
    },
    "codegen": {
        "max_tokens": 620,
        "instruction": BASE + " Return only correct, minimal, self-contained code. Handle edge cases. No extra explanation.",
    },
}


def render(category: str, prompt: str) -> tuple[list[dict[str, str]], int]:
    spec = SPEC.get(category, SPEC["factual"])
    return [
        {"role": "system", "content": spec["instruction"]},
        {"role": "user", "content": prompt},
    ], int(spec["max_tokens"])


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S).strip()


def postprocess(category: str, text: str) -> str:
    text = _strip_think(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if category == "sentiment":
        low = text.lower()
        for label in ("mixed", "positive", "negative", "neutral"):
            if re.search(rf"\b{label}\b", low):
                # Keep a reason if it is short. This usually helps LLM-judge.
                one = re.sub(r"\s+", " ", text).strip()
                return one[:180] if len(one) <= 180 else label.capitalize()
    if category in {"codegen"}:
        # If the model wraps pure code in one fence, remove fence to save output noise.
        m = re.fullmatch(r"```[a-zA-Z0-9_+\-.#]*\s*\n(.*?)\n```", text, flags=re.S)
        if m:
            return m.group(1).strip()
    return text
