"""Compact batch prompting and robust response parsing for Track 1 V17.5.

The evaluator counts Fireworks input and output tokens.  We therefore send all
remaining MiniMax tasks in one request and all remaining Kimi code tasks in one
request, while preserving the proven V17.4 local solvers and model split.
"""
from __future__ import annotations

import json
import re
from typing import Any

from . import prompts

_CODE = {
    "factual": "F",
    "math": "M",
    "sentiment": "S",
    "summary": "U",
    "ner": "N",
    "logic": "L",
    "debug": "D",
    "codegen": "C",
}

_GENERAL_SYSTEM = (
    "Return only JSON {id:answer}; solve items independently. "
    "F all parts, M arithmetic/final/units, S one-sentence label+reason (both sides if mixed), "
    "U exact limits+all themes, N every exact entity+type, L every constraint+final. Be concise."
)


_CODE_SYSTEM = (
    "Return only JSON {id:answer}; solve independently. "
    "D name bug briefly then minimal runnable fix. C minimal self-contained code only; handle stated edge cases; no fences."
)



def _quoted_or_tail(prompt: str) -> str:
    quoted = re.findall(r"['\"]([^'\"]{12,})['\"]", prompt, flags=re.S)
    if quoted:
        return quoted[-1].strip()
    return prompt.strip()


def _summary_rule(prompt: str) -> str:
    low = re.sub(r"\s+", " ", prompt)
    rules: list[str] = []
    for pattern in (
        r"exactly\s+(?:one|two|three|four|five|\d+)\s+sentences?",
        r"exactly\s+(?:one|two|three|four|five|\d+)\s+(?:bullet points?|bullets?|points?)",
        r"(?:each|per)\s+(?:bullet|point|line|sentence).{0,30}(?:no longer than|at most|under)\s+\d+\s+words?",
        r"(?:no longer than|at most|under)\s+\d+\s+words?\s+(?:each|per)\s+(?:bullet|point|line|sentence)",
        r"exactly\s+\d+\s+words?",
    ):
        match = re.search(pattern, low, re.I)
        if match:
            rules.append(match.group(0))
    return "; ".join(rules) or "concise summary"


def compact_prompt(category: str, prompt: str) -> dict[str, Any]:
    """Remove redundant task wording while retaining all answer-relevant text."""
    if category == "sentiment":
        return {"text": _quoted_or_tail(prompt)}
    if category == "summary":
        return {"rule": _summary_rule(prompt), "text": _quoted_or_tail(prompt)}
    if category == "ner":
        return {"text": _quoted_or_tail(prompt)}
    return {"text": prompt.strip()}


def make_messages(items: list[dict[str, str]], code_batch: bool) -> list[dict[str, str]]:
    payload = [
        {
            "id": item["task_id"],
            "c": _CODE[item["category"]],
            **compact_prompt(item["category"], item["prompt"]),
        }
        for item in items
    ]
    return [
        {"role": "system", "content": _CODE_SYSTEM if code_batch else _GENERAL_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def batch_cap(items: list[dict[str, str]], code_batch: bool) -> int:
    """A hard ceiling; actual concise answers should be much shorter."""
    caps = {
        "factual": 68,
        "math": 56,
        "sentiment": 36,
        "summary": 105,
        "ner": 72,
        "logic": 72,
        "debug": 170,
        "codegen": 190,
    }
    total = sum(caps[item["category"]] for item in items) + 24
    return min(total, 650 if code_batch else 520)


def _extract_json(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def parse_answers(text: str, expected_ids: set[str], categories: dict[str, str]) -> dict[str, str]:
    obj = _extract_json(text)
    raw: dict[str, Any] = {}
    if isinstance(obj, dict):
        raw = obj
    elif isinstance(obj, list):
        for row in obj:
            if isinstance(row, dict) and "id" in row and "answer" in row:
                raw[str(row["id"])] = row["answer"]
    answers: dict[str, str] = {}
    for task_id in expected_ids:
        value = raw.get(task_id)
        if isinstance(value, (str, int, float)):
            answer = prompts.postprocess(categories[task_id], str(value))
            if answer.strip():
                answers[task_id] = answer.strip()
    return answers
