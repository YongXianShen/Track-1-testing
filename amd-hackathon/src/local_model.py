"""Guarded local inference for Track 1 V17.9.

This module never calls Fireworks or any external API.  It uses a bundled
Phi-4-mini-instruct GGUF model and deterministic validators.  Task-specific
prompts are generic and do not depend on hidden task IDs or cached answers.
"""
from __future__ import annotations

import ast
import os
import re
import threading
from typing import Any

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/Phi-4-mini-instruct-IQ4_XS.gguf")
N_CTX = int(os.environ.get("LOCAL_MODEL_CTX", "2048"))
N_THREADS = int(os.environ.get("LOCAL_MODEL_THREADS", "2"))
N_BATCH = int(os.environ.get("LOCAL_MODEL_BATCH", "32"))
SEED = int(os.environ.get("LOCAL_MODEL_SEED", "42"))
MAX_REPAIRS = int(os.environ.get("LOCAL_MODEL_REPAIRS", "1"))

_MODEL: Any | None = None
_LOCK = threading.Lock()

_BASE = (
    "Solve the task yourself. Check accuracy and every requested part before answering. "
    "Return only the final answer, without meta-commentary."
)

_INSTRUCTIONS: dict[str, tuple[str, int]] = {
    "factual": (
        _BASE + " Explain all requested differences, reasons, and uses directly in at most 100 words.",
        150,
    ),
    "math": (
        _BASE + " Use correct ordered arithmetic. Include all requested values and units; end with 'Answer:'.",
        140,
    ),
    "sentiment": (
        _BASE + " Use one sentence: Positive, Negative, Neutral, or Mixed, followed by a concrete reason. If mixed, mention both sides.",
        70,
    ),
    "summary": (
        _BASE + " Output only the summary. Obey exact sentence, bullet, line, and word limits; retain every major side/theme.",
        190,
    ),
    "ner": (
        _BASE + " Extract every distinct entity with exact span as 'Entity — PERSON/ORGANIZATION/LOCATION/DATE'. Output one per line.",
        160,
    ),
    "logic": (
        _BASE + " Apply every constraint. Give at most two brief deductions and end with 'Answer: <result>'.",
        150,
    ),
    "debug": (
        _BASE + " Briefly identify the bug, then provide minimal corrected runnable code that handles stated edge cases.",
        360,
    ),
    "codegen": (
        _BASE + " Return minimal self-contained correct code only. Handle duplicates, empty input, and specified edge cases.",
        380,
    ),
}

_WORD_NUMBERS = {
    "one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


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
    text = re.sub(r"^```(?:text|json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _requested_count(prompt: str, unit: str) -> int | None:
    match = re.search(
        rf"\b(one|single|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+{unit}s?\b",
        prompt,
        re.I,
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
    code = _strip_fence(text)
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def validate(category: str, prompt: str, answer: str) -> tuple[bool, str]:
    answer = _clean(answer)
    if not answer or len(answer) < 3:
        return False, "The answer is empty."
    low = answer.lower()
    if any(phrase in low for phrase in ("i cannot answer", "unable to determine", "summary unavailable", "not enough information")):
        return False, "The answer refuses instead of solving the task."

    if category == "factual":
        if _word_count(answer) < 8:
            return False, "The explanation is too incomplete."
        if len(re.findall(r"\?", prompt)) >= 2 and _sentence_count(answer) < 2:
            return False, "The prompt has multiple parts but the answer appears incomplete."

    elif category == "math":
        if not re.search(r"\d", answer):
            return False, "A numerical result is missing."
        # Multi-result questions should normally contain at least two numeric values.
        if re.search(r"\b(?:and|total cost|how much.+and|both)\b", prompt, re.I):
            if len(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", answer)) < 2:
                return False, "Not every requested numerical result is present."

    elif category == "sentiment":
        labels = [x for x in ("positive", "negative", "neutral", "mixed") if re.search(rf"\b{x}\b", low)]
        if not labels or _word_count(answer) < 6:
            return False, "A supported label and concrete reason are required."
        source = prompt.lower()
        contrast = any(token in source for token in (" but ", " however ", " although ", " yet ", " while "))
        if contrast and labels[0] == "negative":
            return False, "The contrastive review may contain a positive outcome that was ignored."

    elif category == "summary":
        bullets = _requested_count(prompt, "bullet") or _requested_count(prompt, "point")
        sentences = _requested_count(prompt, "sentence") or _requested_count(prompt, "line")
        exact_words = _requested_count(prompt, "word") if "exactly" in prompt.lower() else None
        if bullets is not None:
            lines = [line for line in answer.splitlines() if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line)]
            if len(lines) != bullets:
                return False, f"Return exactly {bullets} bullet points."
            per = re.search(r"(?:each|per\s+bullet).{0,30}?(\d+)\s+words?", prompt, re.I)
            if per and any(_word_count(line) > int(per.group(1)) for line in lines):
                return False, "A bullet exceeds the stated word limit."
        elif sentences is not None and _sentence_count(answer) != sentences:
            return False, f"Return exactly {sentences} sentences."
        elif exact_words is not None and _word_count(answer) != exact_words:
            return False, f"Return exactly {exact_words} words."

    elif category == "ner":
        labels = re.findall(r"\b(?:PERSON|ORGANIZATION|LOCATION|DATE)\b", answer.upper())
        if not labels:
            return False, "No valid entity labels were returned."
        # Ensure obvious dates in the source are not omitted.
        if re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", prompt, re.I):
            if "DATE" not in answer.upper():
                return False, "A visible date was omitted."

    elif category == "logic":
        if not re.search(r"\banswer\s*:", answer, re.I):
            return False, "End with an explicit Answer: result."

    elif category in {"debug", "codegen"}:
        if re.search(r"\bpython\b|\bdef\b", prompt, re.I) and not _python_syntax_ok(answer):
            return False, "The returned Python code is not syntactically valid."
        if not re.search(r"\bdef\s+\w+|\bclass\s+\w+|#include|\bfunction\s+\w+|\bSELECT\b|=>|\{", answer, re.I):
            return False, "Runnable code is missing."

    return True, ""


def _generate(system: str, user: str, max_tokens: int) -> str:
    model = _load()
    result = model.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        repeat_penalty=1.04,
        seed=SEED,
    )
    choices = result.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return _clean(str(message.get("content") or ""))


def answer(category: str, prompt: str, allow_repair: bool = True) -> str:
    instruction, cap = _INSTRUCTIONS.get(category, _INSTRUCTIONS["factual"])
    draft = _generate(instruction, prompt, cap)
    ok, issue = validate(category, prompt, draft)
    if ok or not allow_repair or MAX_REPAIRS <= 0:
        return _strip_fence(draft) if category == "codegen" else draft

    repair_system = (
        instruction
        + " The previous answer failed a deterministic format/completeness check. Correct it; do not discuss the check."
    )
    repair_user = f"TASK:\n{prompt}\n\nPREVIOUS ANSWER:\n{draft}\n\nPROBLEM:\n{issue}"
    repaired = _generate(repair_system, repair_user, cap)
    repaired_ok, _ = validate(category, prompt, repaired)
    chosen = repaired if repaired_ok else draft
    return _strip_fence(chosen) if category == "codegen" else chosen
