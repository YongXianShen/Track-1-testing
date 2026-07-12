"""Conservative local GGUF inference used by V17.8 hybrid variants.

The model is loaded lazily and reused. Only configured categories are attempted.
Invalid or failed local outputs fall back to the proven V17.5 MiniMax/Kimi path.
This module never calls Fireworks.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/model.gguf")
MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "local-gguf")
N_CTX = int(os.environ.get("LOCAL_MODEL_CTX", "1536"))
N_THREADS = int(os.environ.get("LOCAL_MODEL_THREADS", "2"))
N_BATCH = int(os.environ.get("LOCAL_MODEL_BATCH", "64"))
SEED = int(os.environ.get("LOCAL_MODEL_SEED", "42"))

_DEFAULT_CATEGORIES = "sentiment,summary"
CATEGORIES = {
    item.strip().lower()
    for item in os.environ.get("LOCAL_MODEL_CATEGORIES", _DEFAULT_CATEGORIES).split(",")
    if item.strip()
}

_MODEL: Any | None = None
_LOCK = threading.Lock()

_SPEC: dict[str, tuple[str, int]] = {
    "factual": ("Answer every part accurately and directly in at most 80 words.", 128),
    "math": ("Show only essential arithmetic and end with Answer: followed by the final value and unit.", 112),
    "sentiment": ("One sentence only: label Positive, Negative, Neutral, or Mixed, then justify it from the text; mention both sides if mixed.", 56),
    "summary": ("Return only the requested summary and obey exact sentence, bullet, line, and word limits.", 150),
    "ner": ("List every distinct entity as Entity — PERSON/ORGANIZATION/LOCATION/DATE; preserve exact spans.", 120),
    "logic": ("Apply every constraint, give at most two brief deductions, and end with Answer: followed by the result.", 120),
    "debug": ("Briefly identify the bug, then provide minimal corrected runnable code.", 260),
    "codegen": ("Return minimal self-contained correct code only and handle stated edge cases.", 280),
}


def enabled_for(category: str) -> bool:
    return category in CATEGORIES


def _load_model() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"local model missing: {MODEL_PATH}")
        from llama_cpp import Llama

        # Let llama-cpp read the chat template embedded in the GGUF. This works
        # for both Qwen2.5 and Gemma 2 and avoids forcing the wrong template.
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


def _strip(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S)
    text = re.sub(r"^```(?:text|json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text.strip())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()])


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _requested_count(prompt: str, unit: str) -> int | None:
    words = {
        "one": 1, "single": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    match = re.search(
        rf"\b(one|single|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+{unit}s?\b",
        prompt,
        re.I,
    )
    if not match:
        return None
    raw = match.group(1).lower()
    return int(raw) if raw.isdigit() else words.get(raw)


def validate(category: str, prompt: str, answer: str) -> bool:
    answer = _strip(answer)
    if not answer or len(answer) < 3:
        return False
    low = answer.lower()
    if category == "sentiment":
        labels = [name for name in ("positive", "negative", "neutral", "mixed") if re.search(rf"\b{name}\b", low)]
        if not labels or len(answer.split()) < 5:
            return False
        # Contrast-heavy prompts must not be accepted as one-sided negative.
        prompt_low = prompt.lower()
        if any(word in prompt_low for word in (" but ", " however ", " although ", " yet ")):
            if labels[0] == "negative":
                return False
    elif category == "ner":
        labels = ("PERSON", "ORGANIZATION", "LOCATION", "DATE")
        if not any(label in answer.upper() for label in labels):
            return False
    elif category == "summary":
        sentences = _requested_count(prompt, "sentence") or _requested_count(prompt, "line")
        bullets = _requested_count(prompt, "bullet") or _requested_count(prompt, "point")
        words = _requested_count(prompt, "word")
        if bullets is not None:
            lines = [line for line in answer.splitlines() if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line)]
            if len(lines) != bullets:
                return False
            per_item = re.search(r"(?:each|per bullet)[^\d]{0,20}(\d+)\s+words?", prompt, re.I)
            if per_item and any(_word_count(line) > int(per_item.group(1)) for line in lines):
                return False
        elif sentences is not None and _sentence_count(answer) != sentences:
            return False
        elif words is not None and "exactly" in prompt.lower() and _word_count(answer) != words:
            return False
    elif category in {"debug", "codegen"}:
        if not re.search(r"\bdef\s+\w+|\bclass\s+\w+|#include|function\s+\w+|SELECT\b|=>|\{", answer, re.I):
            return False
    return True


def complete(category: str, prompt: str) -> str:
    instruction, cap = _SPEC.get(category, _SPEC["factual"])
    model = _load_model()
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=cap,
        repeat_penalty=1.05,
        seed=SEED,
    )
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return _strip(str(message.get("content") or ""))
