"""Brocacho Precision Router V14.

A low-token, one-call-first agent for AMD Hackathon Track 1. It uses only
harness-provided models and does not depend on paid Gemma deployment.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

from . import client, local, models, prompts, router

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
DEADLINE_SECONDS = float(os.environ.get("DEADLINE_SECONDS", str(8.6 * 60)))
CONCURRENCY = max(1, min(int(os.environ.get("CONCURRENCY", "4")), 8))
ENABLE_LOCAL = os.environ.get("ENABLE_LOCAL", "1").lower() in {"1", "true", "yes"}

_STARTED = time.monotonic()
_USAGE: list[dict[str, Any]] = []


def _log(event: str, **fields: Any) -> None:
    print(f"{event} {json.dumps(fields, ensure_ascii=False)}", file=sys.stderr, flush=True)


def _task_prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "text"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


async def _call(
    task_id: str,
    category: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    try:
        answer, usage = await client.complete(model, category, messages, max_tokens)
        _USAGE.append({"task_id": task_id, "category": category, "model": model, **usage})
        _log("USAGE", task_id=task_id, category=category, model=model, **usage)
        return answer
    except Exception as error:
        _log("CALL_ERROR", task_id=task_id, category=category, model=model, error=str(error)[:240])
        return ""


async def _solve(
    task: dict[str, Any],
    plan: models.ModelPlan,
    semaphore: asyncio.Semaphore,
    results: dict[str, str],
) -> None:
    task_id = str(task.get("task_id", ""))
    prompt = _task_prompt(task)
    if not prompt:
        results[task_id] = ""
        return

    category = router.classify(prompt)

    if ENABLE_LOCAL:
        local_answer = local.solve(category, prompt)
        if local_answer is not None:
            results[task_id] = local_answer
            _log("LOCAL", task_id=task_id, category=category)
            return

    messages, max_tokens = prompts.render(category, prompt)
    primary = models.primary_for(category, plan)

    async with semaphore:
        answer = await _call(task_id, category, primary, messages, max_tokens)
        if not answer.strip():
            fallback = models.fallback_for(category, plan)
            if fallback != primary:
                answer = await _call(task_id, category + ":fallback", fallback, messages, max_tokens)

    results[task_id] = prompts.postprocess(category, answer)


async def _run(tasks: list[dict[str, Any]], results: dict[str, str]) -> models.ModelPlan:
    plan = models.build_plan()
    _log("MODEL_PLAN", **plan.as_dict(), enable_gemma=os.environ.get("ENABLE_GEMMA", "0"))
    semaphore = asyncio.Semaphore(CONCURRENCY)
    jobs = [_solve(task, plan, semaphore, results) for task in tasks]
    remaining = max(5.0, DEADLINE_SECONDS - (time.monotonic() - _STARTED))
    try:
        await asyncio.wait_for(asyncio.gather(*jobs, return_exceptions=True), timeout=remaining)
    except asyncio.TimeoutError:
        _log("DEADLINE", elapsed_seconds=round(time.monotonic() - _STARTED, 2))
    finally:
        await client.close()
    return plan


def _write_results(tasks: list[dict[str, Any]], results: dict[str, str]) -> None:
    directory = os.path.dirname(OUTPUT_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = [
        {"task_id": task["task_id"], "answer": str(results.get(str(task["task_id"]), "")).strip()}
        for task in tasks
    ]
    temporary = OUTPUT_PATH + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, OUTPUT_PATH)


def _write_usage(plan: models.ModelPlan) -> None:
    try:
        output_dir = os.path.dirname(OUTPUT_PATH) or "."
        totals = {
            "prompt_tokens": sum(item["prompt_tokens"] for item in _USAGE),
            "completion_tokens": sum(item["completion_tokens"] for item in _USAGE),
            "total_tokens": sum(item["prompt_tokens"] + item["completion_tokens"] for item in _USAGE),
            "calls": len(_USAGE),
        }
        with open(os.path.join(output_dir, "model_usage.json"), "w", encoding="utf-8") as handle:
            json.dump({"model_plan": plan.as_dict(), "calls": _USAGE, "totals": totals}, handle, indent=2)
    except Exception as error:
        _log("USAGE_LOG_ERROR", error=str(error)[:180])


def main() -> int:
    try:
        with open(INPUT_PATH, encoding="utf-8-sig") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list):
            raise ValueError("/input/tasks.json must contain a JSON list")

        tasks = [item for item in raw if isinstance(item, dict) and item.get("task_id")]
        required = ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ValueError("Missing environment variables: " + ", ".join(missing))

        results = {str(task["task_id"]): "" for task in tasks}
        plan = asyncio.run(_run(tasks, results))
        _write_results(tasks, results)
        _write_usage(plan)
        _log(
            "DONE",
            tasks=len(tasks),
            answered=sum(bool(answer) for answer in results.values()),
            elapsed_seconds=round(time.monotonic() - _STARTED, 2),
        )
        return 0
    except Exception as error:
        _log("FATAL", error=str(error)[:300])
        return 1


if __name__ == "__main__":
    sys.exit(main())
