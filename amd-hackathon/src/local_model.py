"""Runtime-optimized local Phi-4 inference for Track 1 V17.10.

This module makes zero Fireworks calls.  Instead of generating one response per
unresolved task, it answers compact batches in strict JSON.  The model is loaded
once, prompt evaluation is shared across tasks, and no repair generation is used.
"""
from __future__ import annotations

import ast
import json
import os
import re
import threading
from typing import Any

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/Phi-4-mini-instruct-IQ4_XS.gguf")
N_CTX = int(os.environ.get("LOCAL_MODEL_CTX", "4096"))
N_THREADS = int(os.environ.get("LOCAL_MODEL_THREADS", "2"))
N_BATCH = int(os.environ.get("LOCAL_MODEL_BATCH", "128"))
SEED = int(os.environ.get("LOCAL_MODEL_SEED", "42"))
GENERAL_MAX_TOKENS = int(os.environ.get("LOCAL_GENERAL_MAX_TOKENS", "720"))
CODE_MAX_TOKENS = int(os.environ.get("LOCAL_CODE_MAX_TOKENS", "640"))

_MODEL: Any | None = None
_LOCK = threading.Lock()

_CATEGORY_CODE = {
    "factual": "F",
    "math": "M",
    "sentiment": "S",
    "summary": "U",
    "ner": "N",
    "logic": "L",
    "debug": "D",
    "codegen": "C",
}

_BATCH_RULES = (
    "Return only one JSON object mapping each id to its final answer string. "
    "Answer every id exactly once. F: answer all requested facts, differences, reasons and uses. "
    "M: calculate in order and include every requested value and unit. "
    "S: use Positive, Negative, Neutral or Mixed plus one concrete sentence; mixed must mention both sides. "
    "U: output only the summary and obey exact sentence, bullet and word limits. "
    "N: list every distinct entity as 'span — PERSON/ORGANIZATION/LOCATION/DATE'. "
    "L: apply every constraint and end with 'Answer:'. "
    "D: briefly identify the bug and give minimal corrected runnable code. "
    "C: give minimal self-contained correct code only. Do not add meta-commentary."
)


def _load() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Bundled model missing: {MODEL_PATH}")
        from llama_cpp import Llama

        _MODEL = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_threads_batch=N_THREADS,
            n_batch=N_BATCH,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            seed=SEED,
            verbose=False,
        )
        return _MODEL


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S)
    text = re.sub(r"^```(?:json|text)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = _clean(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}




_WORD_NUMBERS = {
    "one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _requested_count(prompt: str, unit: str) -> int | None:
    match = re.search(
        rf"\b(one|single|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+{unit}s?\b",
        prompt, re.I,
    )
    if not match:
        return None
    raw = match.group(1).lower()
    return int(raw) if raw.isdigit() else _WORD_NUMBERS.get(raw)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _sentence_count(text: str) -> int:
    return len([x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()])


def _strip_fence(text: str) -> str:
    match = re.fullmatch(r"```[A-Za-z0-9_+.#-]*\s*\n(.*?)\n```", text.strip(), flags=re.S)
    return match.group(1).strip() if match else text.strip()


def _python_syntax_ok(text: str) -> bool:
    try:
        ast.parse(_strip_fence(text))
        return True
    except SyntaxError:
        return False


def validate(category: str, prompt: str, answer: str) -> tuple[bool, str]:
    answer = _clean(answer)
    if not answer or len(answer) < 3:
        return False, "empty"
    low = answer.lower()
    if any(x in low for x in ("i cannot answer", "unable to determine", "not enough information")):
        return False, "refusal"
    if category == "factual":
        return (_word_count(answer) >= 8, "incomplete factual answer")
    if category == "math":
        return (bool(re.search(r"\d", answer)), "missing numerical result")
    if category == "sentiment":
        labels = [x for x in ("positive", "negative", "neutral", "mixed") if re.search(rf"\b{x}\b", low)]
        return (bool(labels) and _word_count(answer) >= 6, "missing label or reason")
    if category == "summary":
        bullets = _requested_count(prompt, "bullet") or _requested_count(prompt, "point")
        sentences = _requested_count(prompt, "sentence") or _requested_count(prompt, "line")
        exact_words = _requested_count(prompt, "word") if "exactly" in prompt.lower() else None
        if bullets is not None:
            lines = [line for line in answer.splitlines() if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line)]
            return (len(lines) == bullets, "wrong bullet count")
        if sentences is not None:
            return (_sentence_count(answer) == sentences, "wrong sentence count")
        if exact_words is not None:
            return (_word_count(answer) == exact_words, "wrong word count")
        return True, ""
    if category == "ner":
        labels = re.findall(r"\b(?:PERSON|ORGANIZATION|LOCATION|DATE)\b", answer.upper())
        return (bool(labels), "missing entity labels")
    if category == "logic":
        return (bool(re.search(r"\banswer\s*:", answer, re.I)), "missing final answer")
    if category in {"debug", "codegen"}:
        if re.search(r"\bpython\b|\bdef\b", prompt, re.I):
            return (_python_syntax_ok(answer), "invalid Python")
        return (bool(re.search(r"\bdef\s+\w+|\bclass\s+\w+|#include|\bfunction\s+\w+|\bSELECT\b|=>|\{", answer, re.I)), "missing code")
    return True, ""


def _generate_json(payload: list[dict[str, str]], max_tokens: int) -> dict[str, str]:
    model = _load()
    user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    kwargs: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": _BATCH_RULES},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "repeat_penalty": 1.03,
        "seed": SEED,
    }
    # llama-cpp-python supports JSON-constrained output.  If a wheel/runtime
    # lacks this option, retry once without the constraint rather than failing.
    try:
        result = model.create_chat_completion(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except (TypeError, ValueError):
        result = model.create_chat_completion(**kwargs)

    choices = result.get("choices") or []
    if not choices:
        return {}
    message = choices[0].get("message") or {}
    parsed = _extract_json(str(message.get("content") or ""))
    answers: dict[str, str] = {}
    for key, value in parsed.items():
        if isinstance(value, str):
            answers[str(key)] = value.strip()
        elif value is not None:
            answers[str(key)] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return answers


def answer_batch(items: list[dict[str, str]], *, code_batch: bool = False) -> dict[str, str]:
    """Answer a compact batch.

    Each item must contain ``id``, ``category`` and ``prompt``.  A category code
    is sent instead of a repeated natural-language instruction to keep prompt
    evaluation small.  The caller validates/post-processes individual answers.
    """
    if not items:
        return {}
    payload = [
        {
            "id": str(item["id"]),
            "c": _CATEGORY_CODE.get(item["category"], "F"),
            "q": item["prompt"],
        }
        for item in items
    ]
    cap = CODE_MAX_TOKENS if code_batch else GENERAL_MAX_TOKENS
    return _generate_json(payload, cap)
