"""Category prompts tuned for accuracy first, then low visible output tokens."""
from __future__ import annotations

import re

_BASE = "English. No preamble. Follow the requested format exactly."

SPEC = {
    "factual": {
        "max_tokens": 300,
        "instruction": f"{_BASE} Answer clearly in under 120 words. Include all requested parts.",
    },
    "math": {
        "max_tokens": 360,
        "instruction": f"{_BASE} Solve accurately. Give at most 2 short steps, then put 'Answer: ' with the final result.",
    },
    "sentiment": {
        "max_tokens": 95,
        "instruction": f"{_BASE} Label as Positive, Negative, Neutral, or Mixed, then give one short reason.",
    },
    "summary": {
        "max_tokens": 210,
        "instruction": f"{_BASE} Output only the summary. Obey any sentence, word, bullet, style, or length constraint.",
    },
    "ner": {
        "max_tokens": 230,
        "instruction": f"{_BASE} Extract only requested entities. One per line as 'type: exact text'. Use person, organization, location, date when applicable.",
    },
    "debug": {
        "max_tokens": 500,
        "instruction": f"{_BASE} Name the bug in one sentence, then provide corrected code in one fenced block.",
    },
    "logic": {
        "max_tokens": 380,
        "instruction": f"{_BASE} Use every constraint. Give at most 2 short steps, then put 'Answer: ' with the final result.",
    },
    "codegen": {
        "max_tokens": 500,
        "instruction": f"{_BASE} Return only correct, self-contained code in one fenced block. No comments.",
    },
}

CATEGORIES = tuple(SPEC)


def render(category: str, prompt: str) -> tuple[list[dict], int]:
    spec = SPEC[category]
    return [
        {"role": "system", "content": spec["instruction"]},
        {"role": "user", "content": prompt},
    ], spec["max_tokens"]


def postprocess(category: str, text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S).strip()
    # Do not strip Answer: for math/logic; the reference-style answer passed with it.
    if category == "sentiment":
        low = text.lower()
        for label in ("positive", "negative", "neutral", "mixed"):
            if re.search(rf"\b{label}\b", low):
                reason = text.strip()
                # Keep a one-line reason if present; otherwise label alone.
                if len(reason) <= 160 and reason.lower().startswith(label):
                    return reason
                return label.capitalize()
    return text
