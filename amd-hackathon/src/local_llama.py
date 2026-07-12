"""Local Llama 3.2 3B Instruct Q4_K_M inference.

The model is loaded lazily and reused for all local generations.  This module
never calls Fireworks.  It is intentionally conservative: only categories in
LLAMA_CATEGORIES are attempted, and malformed outputs fall back to the proven
remote V17.5 path unless LOCAL_ONLY=1.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any

MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH",
    "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
)
N_CTX = int(os.environ.get("LLAMA_CTX", "2048"))
N_THREADS = int(os.environ.get("LLAMA_THREADS", "2"))
N_BATCH = int(os.environ.get("LLAMA_BATCH", "128"))
SEED = int(os.environ.get("LLAMA_SEED", "42"))

_DEFAULT_CATEGORIES = "sentiment,summary,ner"
CATEGORIES = {
    x.strip().lower()
    for x in os.environ.get("LLAMA_CATEGORIES", _DEFAULT_CATEGORIES).split(",")
    if x.strip()
}

_MODEL: Any | None = None
_LOCK = threading.Lock()

_SPEC: dict[str, tuple[str, int]] = {
    "factual": ("Answer every part directly and accurately in at most 90 words.", 150),
    "math": ("Show brief arithmetic and end with 'Answer: <final>' including units.", 120),
    "sentiment": ("One sentence: Positive, Negative, Neutral, or Mixed, then a reason. Mention both sides if mixed.", 64),
    "summary": ("Return only the requested summary. Obey exact sentence, bullet, line, and word limits.", 180),
    "ner": ("List every distinct entity as 'Entity — PERSON/ORGANIZATION/LOCATION/DATE'. Preserve exact spans.", 140),
    "logic": ("Apply every constraint, use at most two short deductions, and end with 'Answer: <final>'.", 140),
    "debug": ("Briefly name the bug, then provide minimal corrected runnable code.", 300),
    "codegen": ("Return minimal self-contained correct code only, handling stated edge cases.", 320),
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
            chat_format="llama-3",
        )
        return _MODEL


def _strip(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S)
    text = re.sub(r"^```(?:text|json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text.strip())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _sentence_count(text: str) -> int:
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return len(parts)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _requested_count(prompt: str, unit: str) -> int | None:
    words = {
        "one": 1, "single": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10,
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
    if not answer or len(answer) < 2:
        return False
    low = answer.lower()
    if category == "sentiment":
        if not any(re.search(rf"\b{x}\b", low) for x in ("positive", "negative", "neutral", "mixed")):
            return False
        if len(answer.split()) < 4:
            return False
    elif category == "ner":
        if not any(label in answer.upper() for label in ("PERSON", "ORGANIZATION", "LOCATION", "DATE")):
            return False
    elif category == "summary":
        sentences = _requested_count(prompt, "sentence") or _requested_count(prompt, "line")
        bullets = _requested_count(prompt, "bullet") or _requested_count(prompt, "point")
        words = _requested_count(prompt, "word")
        if bullets is not None:
            actual = len([line for line in answer.splitlines() if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line)])
            if actual != bullets:
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
