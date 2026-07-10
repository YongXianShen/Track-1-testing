"""Track 1 stable precision router V15.

Built directly from the proven V12 baseline. It requires no paid deployment and
keeps the same model plan while fixing high-risk routing and local-sentiment cases.
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
ENABLE_LLM_ROUTE_FALLBACK = os.environ.get("ENABLE_LLM_ROUTE_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}

START = time.monotonic()
CALL_LOG: list[dict[str, Any]] = []
MODEL_PLAN: dict[str, str] = {}


def log(event: str, **fields: Any) -> None:
    print(event + " " + json.dumps(fields, ensure_ascii=False), file=sys.stderr, flush=True)


def _prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "text"):
        val = task.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


async def try_complete(task_id: str, category: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    try:
        text, usage = await client.complete(model, messages, max_tokens)
        CALL_LOG.append({"task_id": task_id, "category": category, "model": model, **usage})
        log("USAGE", task_id=task_id, category=category, model=model, **usage)
        return text
    except Exception as err:
        log("ERROR", task_id=task_id, category=category, model=model, error=str(err)[:220])
        return ""


async def classify_by_llm(task_id: str, prompt: str, plan: models.ModelPlan) -> str:
    text = await try_complete(task_id, "router", plan.SMALL, router.fallback_messages(prompt), max_tokens=2)
    return router.parse_fallback_letter(text)


async def solve_task(task: dict[str, Any], plan: models.ModelPlan, sem: asyncio.Semaphore, results: dict[str, str]) -> None:
    task_id = str(task.get("task_id", ""))
    prompt = _prompt(task)
    if not prompt:
        results[task_id] = ""
        return

    category = router.classify(prompt)
    if category is None and ENABLE_LLM_ROUTE_FALLBACK:
        category = await classify_by_llm(task_id, prompt, plan)
    category = category or "factual"

    if ENABLE_LOCAL:
        local = solvers.solve(category, prompt)
        if local:
            log("LOCAL", task_id=task_id, category=category)
            results[task_id] = local
            return

    messages, max_tokens = prompts.render(category, prompt)
    primary = models.model_for(category, plan)
    async with sem:
        answer = await try_complete(task_id, category, primary, messages, max_tokens)
        if not answer.strip():
            backup = models.fallback_model(category, plan)
            answer = await try_complete(task_id, category + ":fallback", backup, messages, max_tokens)
    results[task_id] = prompts.postprocess(category, answer)


async def run(tasks: list[dict[str, Any]], results: dict[str, str]) -> None:
    plan = models.build_plan()
    MODEL_PLAN.update(plan.as_dict())
    log("MODEL_PLAN", **MODEL_PLAN, enable_gemma=os.environ.get("ENABLE_GEMMA", "0"))
    sem = asyncio.Semaphore(max(1, min(CONCURRENCY, 8)))
    jobs = [solve_task(t, plan, sem, results) for t in tasks]
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
    payload = [{"task_id": t["task_id"], "answer": str(results.get(str(t["task_id"]), "")).strip()} for t in tasks]
    tmp = OUTPUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, OUTPUT_PATH)


def write_usage_log() -> None:
    try:
        totals = {
            "prompt_tokens": sum(int(c.get("prompt_tokens", 0)) for c in CALL_LOG),
            "completion_tokens": sum(int(c.get("completion_tokens", 0)) for c in CALL_LOG),
            "total_tokens": sum(int(c.get("prompt_tokens", 0)) + int(c.get("completion_tokens", 0)) for c in CALL_LOG),
            "calls": len(CALL_LOG),
        }
        path = os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "model_usage.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": "v15-stable-precision", "model_plan": MODEL_PLAN, "calls": CALL_LOG, "totals": totals}, f, ensure_ascii=False, indent=2)
    except Exception as err:
        log("WARN", error=str(err)[:180])


def main() -> int:
    try:
        with open(INPUT_PATH, encoding="utf-8-sig") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("tasks.json must be a list")
        tasks = [t for t in raw if isinstance(t, dict) and t.get("task_id")]
        if not os.environ.get("FIREWORKS_API_KEY") or not os.environ.get("FIREWORKS_BASE_URL") or not os.environ.get("ALLOWED_MODELS"):
            raise ValueError("FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS are required")
        results = {str(t["task_id"]): "" for t in tasks}
        asyncio.run(run(tasks, results))
        write_results(tasks, results)
        write_usage_log()
        log("DONE", tasks=len(tasks), answered=sum(1 for v in results.values() if v), elapsed_s=round(time.monotonic() - START, 1))
        return 0
    except Exception as err:
        log("FATAL", error=str(err)[:300])
        return 1


if __name__ == "__main__":
    sys.exit(main())
