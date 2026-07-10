"""Track 1 Gemma-aware hybrid token router V11.

Reads /input/tasks.json and writes /output/results.json.
Accuracy target: use proven tiered routing, Gemma only where it is likely to help,
and very narrow zero-token local solvers.
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
CONCURRENCY = int(os.environ.get("CONCURRENCY", "5"))
ENABLE_LOCAL = os.environ.get("ENABLE_LOCAL", "1").strip().lower() in {"1", "true", "yes"}

START = time.monotonic()
CALL_LOG: list[dict[str, Any]] = []


def log(event: str, **fields: Any) -> None:
    print(event + " " + json.dumps(fields, ensure_ascii=False), file=sys.stderr, flush=True)


def _prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "text"):
        if isinstance(task.get(key), str) and task[key].strip():
            return task[key]
    return ""


async def try_complete(task_id: str, category: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    try:
        text, usage = await client.complete(model, messages, max_tokens)
        log("USAGE", task_id=task_id, category=category, model=model, **usage)
        CALL_LOG.append({"task_id": task_id, "category": category, "model": model, **usage})
        return text
    except Exception as error:
        log("ERROR", task_id=task_id, category=category, model=model, error=str(error)[:220])
        return ""


async def llm_classify(task_id: str, prompt: str, tiers: models.Tiers) -> str:
    text = await try_complete(task_id, "router", tiers.SMALL, router.fallback_messages(prompt), max_tokens=2)
    return router.parse_fallback_letter(text)


async def solve_task(task: dict[str, Any], tiers: models.Tiers, sem: asyncio.Semaphore, results: dict[str, str]) -> None:
    task_id = str(task.get("task_id"))
    prompt = _prompt(task)
    if not prompt:
        results[task_id] = ""
        return
    async with sem:
        category = router.classify(prompt)
        if category is None:
            category = await llm_classify(task_id, prompt, tiers)

        if ENABLE_LOCAL:
            local = solvers.solve(category, prompt)
            if local:
                log("LOCAL", task_id=task_id, category=category)
                results[task_id] = local
                return

        messages, max_tokens = prompts.render(category, prompt)
        tier = models.tier_for(category, tiers)
        primary = models.model_for_tier(tier, tiers)
        text = await try_complete(task_id, category, primary, messages, max_tokens)

        # Rescue empty answers only; do not add consensus/review because it often changes a correct answer.
        if not text.strip():
            retry_tier = "LARGE" if tier not in {"LARGE", "CODE"} else ("MEDIUM" if tier == "LARGE" else "LARGE")
            retry_model = models.model_for_tier(retry_tier, tiers)
            retry_max = max(max_tokens, 650 if retry_tier == "LARGE" else max_tokens)
            text = await try_complete(task_id, category, retry_model, messages, retry_max)

        results[task_id] = prompts.postprocess(category, text)


async def run(tasks: list[dict[str, Any]], results: dict[str, str]) -> None:
    tiers = models.build_tiers()
    log("TIERS", **tiers.as_dict())
    sem = asyncio.Semaphore(max(1, min(CONCURRENCY, 8)))
    jobs = [solve_task(task, tiers, sem, results) for task in tasks]
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


def write_inference_log() -> None:
    try:
        totals = {
            "prompt_tokens": sum(int(c.get("prompt_tokens", 0)) for c in CALL_LOG),
            "completion_tokens": sum(int(c.get("completion_tokens", 0)) for c in CALL_LOG),
            "calls": len(CALL_LOG),
        }
        path = os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "inference_log.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"calls": CALL_LOG, "totals": totals}, f, ensure_ascii=False)
    except Exception as error:
        log("WARN", note="could not write inference_log", error=str(error)[:160])


def main() -> int:
    try:
        with open(INPUT_PATH, encoding="utf-8-sig") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("tasks.json must be a list")
        tasks = [t for t in raw if isinstance(t, dict) and t.get("task_id")]
        results = {str(t["task_id"]): "" for t in tasks}
        if not os.environ.get("FIREWORKS_API_KEY") or not os.environ.get("FIREWORKS_BASE_URL") or not os.environ.get("ALLOWED_MODELS"):
            raise ValueError("FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS are required")
        asyncio.run(run(tasks, results))
        write_results(tasks, results)
        write_inference_log()
        log("DONE", tasks=len(tasks), answered=sum(1 for a in results.values() if a), elapsed_s=round(time.monotonic() - START, 1))
        return 0
    except Exception as error:
        log("FATAL", error=str(error)[:300])
        return 1


if __name__ == "__main__":
    sys.exit(main())
