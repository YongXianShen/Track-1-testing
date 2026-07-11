"""Track 1 Stable Lean V18.

A conservative optimization of V17: preserve the proven model strategy, remove
routine routing calls, add only high-confidence zero-token solvers, validate
answers before a fallback call, and shorten prompts/output budgets.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

from . import client, models, prompts, router, solvers

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
DEADLINE_SECONDS = float(os.environ.get("DEADLINE_SECONDS", str(8.5 * 60)))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "4"))
ENABLE_LOCAL = os.environ.get("ENABLE_LOCAL", "1").strip().lower() in {"1", "true", "yes"}
# Off by default: V18's deterministic router always returns a category. Enable
# only for experiments with prompts that the regex router cannot classify.
ENABLE_LLM_ROUTE_FALLBACK = os.environ.get("ENABLE_LLM_ROUTE_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}

START = time.monotonic()
CALL_LOG: list[dict[str, Any]] = []
MODEL_PLAN: dict[str, str] = {}


def log(event: str, **fields: Any) -> None:
    print(event + " " + json.dumps(fields, ensure_ascii=False), file=sys.stderr, flush=True)


def _prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "text"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


async def try_complete(
    task_id: str,
    category: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    reasoning_effort: str | None = None,
) -> tuple[str, bool]:
    try:
        text, usage = await client.complete(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        truncated = bool(usage.pop("truncated", False))
        CALL_LOG.append({
            "task_id": task_id,
            "category": category,
            "model": model,
            **usage,
            "truncated": truncated,
        })
        log("USAGE", task_id=task_id, category=category, model=model, **usage, truncated=truncated)
        return text, truncated
    except Exception as error:
        log("ERROR", task_id=task_id, category=category, model=model, error=str(error)[:220])
        return "", False


async def classify_by_llm(task_id: str, prompt: str, plan: models.ModelPlan) -> str:
    text, _ = await try_complete(
        task_id,
        "router",
        plan.SMALL,
        router.fallback_messages(prompt),
        max_tokens=2,
    )
    return router.parse_fallback_letter(text)


async def solve_task(
    task: dict[str, Any],
    plan: models.ModelPlan,
    sem: asyncio.Semaphore,
    results: dict[str, str],
) -> None:
    task_id = str(task.get("task_id", ""))
    try:
        prompt = _prompt(task)
        if not prompt:
            results[task_id] = ""
            return

        category = router.classify(prompt)
        if category is None and ENABLE_LLM_ROUTE_FALLBACK:
            async with sem:
                category = await classify_by_llm(task_id, prompt, plan)
        category = category or "factual"

        if ENABLE_LOCAL:
            local = solvers.solve(category, prompt)
            if local is not None:
                log("LOCAL", task_id=task_id, category=category)
                results[task_id] = local
                return

        messages, max_tokens = prompts.render(category, prompt)
        primary = models.model_for(category, plan)
        effort = models.reasoning_effort_for(category, primary)

        async with sem:
            raw, truncated = await try_complete(
                task_id,
                category,
                primary,
                messages,
                max_tokens,
                reasoning_effort=effort,
            )
            answer = prompts.postprocess(category, raw, prompt)
            retry = not prompts.is_usable(category, answer, prompt)
            # A cut-off program is normally unusable. For concise natural-language
            # tasks, validation is more reliable than finish_reason alone.
            if truncated and category in {"debug", "codegen"}:
                retry = True

            if retry:
                backup = models.fallback_model(category, plan)
                backup_effort = models.reasoning_effort_for(category, backup)
                raw2, _ = await try_complete(
                    task_id,
                    category + ":fallback",
                    backup,
                    messages,
                    max_tokens,
                    reasoning_effort=backup_effort,
                )
                answer2 = prompts.postprocess(category, raw2, prompt)
                if prompts.is_usable(category, answer2, prompt):
                    answer = answer2

        results[task_id] = answer.strip()
    except Exception as error:
        log("TASK_ERROR", task_id=task_id, error=str(error)[:220])
        results[task_id] = ""


async def run(tasks: list[dict[str, Any]], results: dict[str, str]) -> None:
    plan = models.build_plan()
    MODEL_PLAN.update(plan.as_dict())
    log("MODEL_PLAN", **MODEL_PLAN, version="V18", enable_gemma=os.environ.get("ENABLE_GEMMA", "0"))
    sem = asyncio.Semaphore(max(1, min(CONCURRENCY, 8)))
    jobs = [solve_task(task, plan, sem, results) for task in tasks]
    remaining = DEADLINE_SECONDS - (time.monotonic() - START)
    try:
        await asyncio.wait_for(asyncio.gather(*jobs, return_exceptions=True), timeout=max(5, remaining))
    except asyncio.TimeoutError:
        log("DEADLINE", elapsed=round(time.monotonic() - START, 1))
    finally:
        await client.aclose()


def write_results(tasks: list[dict[str, Any]], results: dict[str, str]) -> None:
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = [
        {"task_id": task["task_id"], "answer": str(results.get(str(task["task_id"]), "")).strip()}
        for task in tasks
    ]
    tmp = OUTPUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, OUTPUT_PATH)


def write_usage_log() -> None:
    try:
        totals = {
            "prompt_tokens": sum(int(call.get("prompt_tokens", 0)) for call in CALL_LOG),
            "completion_tokens": sum(int(call.get("completion_tokens", 0)) for call in CALL_LOG),
            "total_tokens": sum(
                int(call.get("prompt_tokens", 0)) + int(call.get("completion_tokens", 0))
                for call in CALL_LOG
            ),
            "calls": len(CALL_LOG),
        }
        path = os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "model_usage.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {"version": "V18", "model_plan": MODEL_PLAN, "calls": CALL_LOG, "totals": totals},
                file,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as error:
        log("WARN", error=str(error)[:180])


def main() -> int:
    try:
        with open(INPUT_PATH, encoding="utf-8-sig") as file:
            raw = json.load(file)
        if not isinstance(raw, list):
            raise ValueError("tasks.json must be a list")
        tasks = [task for task in raw if isinstance(task, dict) and task.get("task_id")]
        required = ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
        if any(not os.environ.get(name) for name in required):
            raise ValueError("FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS are required")
        results = {str(task["task_id"]): "" for task in tasks}
        asyncio.run(run(tasks, results))
        write_results(tasks, results)
        write_usage_log()
        log(
            "DONE",
            tasks=len(tasks),
            answered=sum(bool(value) for value in results.values()),
            elapsed_s=round(time.monotonic() - START, 1),
        )
        return 0
    except Exception as error:
        log("FATAL", error=str(error)[:300])
        return 1


if __name__ == "__main__":
    sys.exit(main())
