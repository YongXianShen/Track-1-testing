"""Track 1 V17.9: guarded Phi-4-mini fully-local agent.

No Fireworks/API calls are made.  Exact deterministic solvers run first; a
bundled Phi-4-mini-instruct IQ4_XS model handles unresolved tasks.  The code is
generic and never branches on hidden task IDs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from . import local_model, prompts, router, solvers

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
DEADLINE_SECONDS = float(os.environ.get("DEADLINE_SECONDS", str(8.4 * 60)))
REPAIR_MIN_REMAINING = float(os.environ.get("REPAIR_MIN_REMAINING", "55"))
START = time.monotonic()


def log(event: str, **fields: Any) -> None:
    print(event + " " + json.dumps(fields, ensure_ascii=False), file=sys.stderr, flush=True)


def task_prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "text"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def fallback(category: str, prompt: str) -> str:
    # Last-resort outputs preserve the required schema if the model cannot run.
    if category == "sentiment":
        return "Neutral — the text does not show a clearly dominant positive or negative view."
    if category == "ner":
        return "No named entities identified."
    if category == "summary":
        source = re_source(prompt)
        return source[:500].strip()
    if category == "math":
        return "Answer: unable to determine"
    if category == "logic":
        return "Answer: no unique result could be determined."
    if category in {"debug", "codegen"}:
        return "def solution(*args, **kwargs):\n    raise NotImplementedError"
    return "The requested information could not be determined locally."


def re_source(prompt: str) -> str:
    import re
    quoted = re.findall(r"['\"]([^'\"]{20,})['\"]", prompt, flags=re.S)
    if quoted:
        return quoted[-1]
    parts = re.split(r":\s*", prompt, maxsplit=1)
    return parts[-1]


def write_results(tasks: list[dict[str, Any]], results: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    payload = [
        {"task_id": task["task_id"], "answer": str(results.get(str(task["task_id"]), "")).strip()}
        for task in tasks
    ]
    temp = OUTPUT_PATH + ".tmp"
    with open(temp, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
    os.replace(temp, OUTPUT_PATH)


def write_usage(local_count: int, model_count: int) -> None:
    path = os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "model_usage.json")
    data = {
        "version": "V17.9-Phi4-ZeroToken",
        "fireworks_calls": 0,
        "fireworks_tokens": 0,
        "local_model": "Phi-4-mini-instruct-IQ4_XS",
        "deterministic_answers": local_count,
        "local_model_answers": model_count,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def run(tasks: list[dict[str, Any]]) -> dict[str, str]:
    results: dict[str, str] = {}
    deterministic_count = 0
    model_count = 0

    # Shorter model prompts first provide maximum useful coverage before deadline.
    prepared: list[tuple[int, dict[str, Any], str, str]] = []
    for index, task in enumerate(tasks):
        prompt = task_prompt(task)
        category = router.classify(prompt) or "factual"
        prepared.append((index, task, prompt, category))

    unresolved: list[tuple[int, dict[str, Any], str, str]] = []
    for item in prepared:
        _, task, prompt, category = item
        task_id = str(task["task_id"])
        if not prompt:
            results[task_id] = ""
            continue
        exact = solvers.solve(category, prompt)
        if exact:
            results[task_id] = exact
            deterministic_count += 1
            log("DETERMINISTIC", task_id=task_id, category=category)
        else:
            unresolved.append(item)

    # Keep code near the end because its longer outputs are slower; prioritize
    # factual/math/logic/structured language coverage if the runtime is tight.
    category_priority = {
        "factual": 0, "math": 1, "logic": 2, "sentiment": 3,
        "ner": 4, "summary": 5, "debug": 6, "codegen": 7,
    }
    unresolved.sort(key=lambda x: (category_priority.get(x[3], 9), len(x[2])))

    for _, task, prompt, category in unresolved:
        task_id = str(task["task_id"])
        remaining = DEADLINE_SECONDS - (time.monotonic() - START)
        if remaining < 8:
            results[task_id] = fallback(category, prompt)
            log("DEADLINE_FALLBACK", task_id=task_id, category=category)
            continue
        try:
            raw = local_model.answer(category, prompt, allow_repair=remaining >= REPAIR_MIN_REMAINING)
            answer = prompts.postprocess(category, raw).strip()
            results[task_id] = answer or fallback(category, prompt)
            model_count += 1
            log("LOCAL_MODEL", task_id=task_id, category=category, remaining_s=round(remaining, 1))
        except Exception as error:
            results[task_id] = fallback(category, prompt)
            log("LOCAL_MODEL_ERROR", task_id=task_id, category=category, error=str(error)[:220])

    write_usage(deterministic_count, model_count)
    return results


def main() -> int:
    try:
        with open(INPUT_PATH, encoding="utf-8-sig") as file:
            raw = json.load(file)
        if not isinstance(raw, list):
            raise ValueError("tasks.json must contain a JSON list")
        tasks = [item for item in raw if isinstance(item, dict) and item.get("task_id") is not None]
        results = run(tasks)
        write_results(tasks, results)
        log(
            "DONE",
            version="V17.9-Phi4-ZeroToken",
            tasks=len(tasks),
            answered=sum(bool(value) for value in results.values()),
            fireworks_calls=0,
            elapsed_s=round(time.monotonic() - START, 1),
        )
        return 0
    except Exception as error:
        log("FATAL", error=str(error)[:300])
        return 1


if __name__ == "__main__":
    sys.exit(main())
