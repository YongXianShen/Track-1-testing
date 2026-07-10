"""Track 1 V16: fully local, zero-Fireworks-token agent."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

import exact
import tasking
from local_runtime import LocalRuntime

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
DEADLINE_SECONDS = float(os.environ.get("DEADLINE_SECONDS", "510"))
VERIFY_REASONING = os.environ.get("VERIFY_REASONING", "1").lower() in {"1", "true", "yes"}
START = time.monotonic()


def log(event: str, **fields: Any) -> None:
    print(event + " " + json.dumps(fields, ensure_ascii=False), file=sys.stderr, flush=True)


def task_prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "text"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def remaining() -> float:
    return DEADLINE_SECONDS - (time.monotonic() - START)


def model_answer(runtime: LocalRuntime, category: str, prompt: str) -> str:
    messages, max_tokens = tasking.render(category, prompt)
    timeout = max(15.0, min(85.0, remaining() - 5.0))
    raw = runtime.complete(messages, max_tokens=max_tokens, timeout=timeout)
    answer = tasking.postprocess(category, raw)

    violation = tasking.format_violation(category, prompt, answer)
    if violation and remaining() > 55:
        repair_messages = [
            {"role": "system", "content": "Repair only the objective formatting error. Output only the corrected final answer."},
            {"role": "user", "content": f"Original task:\n{prompt}\n\nPrevious answer:\n{answer}\n\nRepair instruction:\n{violation}"},
        ]
        answer = tasking.postprocess(category, runtime.complete(repair_messages, max_tokens=max_tokens, timeout=min(70.0, remaining() - 5.0)))

    if VERIFY_REASONING and category in {"math", "logic"} and remaining() > 75:
        verify_messages = [
            {"role": "system", "content": "Check the candidate against every condition. Return the corrected concise final answer only. Keep 'Answer: <final>'."},
            {"role": "user", "content": f"Task:\n{prompt}\n\nCandidate:\n{answer}"},
        ]
        checked = tasking.postprocess(category, runtime.complete(verify_messages, max_tokens=96, timeout=min(70.0, remaining() - 5.0)))
        if checked.strip():
            answer = checked
    return answer.strip()


def write_results(tasks: list[dict[str, Any]], answers: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    payload = [{"task_id": task["task_id"], "answer": answers.get(str(task["task_id"]), "").strip()} for task in tasks]
    temp = OUTPUT_PATH + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temp, OUTPUT_PATH)


def main() -> int:
    runtime: LocalRuntime | None = None
    try:
        with open(INPUT_PATH, encoding="utf-8-sig") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list):
            raise ValueError("tasks.json must contain a JSON list")
        tasks = [task for task in raw if isinstance(task, dict) and task.get("task_id")]
        answers = {str(task["task_id"]): "" for task in tasks}

        # Solve exact cases before starting the model. This also reduces CPU time.
        pending: list[tuple[dict[str, Any], str, str]] = []
        for task in tasks:
            prompt = task_prompt(task)
            category = tasking.classify(prompt)
            local = exact.solve(category, prompt)
            if local:
                answers[str(task["task_id"])] = local
                log("EXACT", task_id=str(task["task_id"]), category=category)
            else:
                pending.append((task, category, prompt))

        if pending:
            runtime = LocalRuntime()
            runtime.start(timeout=min(150.0, max(30.0, remaining() - 30.0)))
            log("MODEL_READY", model=os.environ.get("LOCAL_MODEL_NAME", "Qwen3.5-2B-Q4_K_M"), pending=len(pending))

            for index, (task, category, prompt) in enumerate(pending):
                task_id = str(task["task_id"])
                if remaining() < 18:
                    log("DEADLINE_SKIP", task_id=task_id, remaining=round(remaining(), 1))
                    continue
                try:
                    answers[task_id] = model_answer(runtime, category, prompt)
                    log("LOCAL_MODEL", task_id=task_id, category=category, chars=len(answers[task_id]))
                except Exception as exc:
                    log("MODEL_ERROR", task_id=task_id, category=category, error=str(exc)[:300])
                    # One short retry, only when enough time remains.
                    if remaining() > 35:
                        try:
                            messages, _ = tasking.render(category, prompt)
                            answers[task_id] = tasking.postprocess(category, runtime.complete(messages, max_tokens=96, timeout=min(30.0, remaining() - 5.0)))
                        except Exception as retry_exc:
                            log("RETRY_ERROR", task_id=task_id, error=str(retry_exc)[:220])
                # Persist progress so a late failure still leaves valid output.
                write_results(tasks, answers)

        write_results(tasks, answers)
        usage_path = os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "model_usage.json")
        with open(usage_path, "w", encoding="utf-8") as handle:
            json.dump({"fireworks_calls": 0, "fireworks_tokens": 0, "local_model": os.environ.get("LOCAL_MODEL_NAME", "Qwen3.5-2B-Q4_K_M")}, handle, indent=2)
        log("DONE", tasks=len(tasks), answered=sum(bool(v) for v in answers.values()), fireworks_calls=0, elapsed_s=round(time.monotonic() - START, 1))
        return 0
    except Exception as exc:
        log("FATAL", error=str(exc)[:400])
        # Best effort: always create a schema-valid result when input was readable.
        return 1
    finally:
        if runtime:
            runtime.stop()


if __name__ == "__main__":
    sys.exit(main())
