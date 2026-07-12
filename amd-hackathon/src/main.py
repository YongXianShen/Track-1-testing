"""Track 1 Local-GGUF Hybrid V17.8.

Built from the proven V17.5 batch router. Deterministic solvers run first.
A bundled 4-bit instruction model handles selected language categories locally.
Invalid or failed local answers fall back to the proven MiniMax/Kimi batch path.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

from . import batching, client, local_model, models, prompts, router, solvers

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
DEADLINE_SECONDS = float(os.environ.get("DEADLINE_SECONDS", str(8.5 * 60)))
ENABLE_LOCAL = os.environ.get("ENABLE_LOCAL", "1").strip().lower() in {"1", "true", "yes"}
ENABLE_LOCAL_MODEL = os.environ.get("ENABLE_LOCAL_MODEL", "1").strip().lower() in {"1", "true", "yes"}
ENABLE_BATCH = os.environ.get("ENABLE_BATCH", "1").strip().lower() in {"1", "true", "yes"}
LOCAL_ONLY = os.environ.get("LOCAL_ONLY", "0").strip().lower() in {"1", "true", "yes"}
LOCAL_MODEL_MIN_REMAINING = float(os.environ.get("LOCAL_MODEL_MIN_REMAINING", "70"))
VERSION = os.environ.get("AGENT_VERSION", "V17.8")

START = time.monotonic()
CALL_LOG: list[dict[str, Any]] = []
LOCAL_MODEL_LOG: list[dict[str, Any]] = []
MODEL_PLAN: dict[str, str] = {}


def log(event: str, **fields: Any) -> None:
    print(event + " " + json.dumps(fields, ensure_ascii=False), file=sys.stderr, flush=True)


def _prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "text"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


async def try_complete(label: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, bool]:
    try:
        text, usage = await client.complete(model, messages, max_tokens)
        truncated = bool(usage.pop("truncated", False))
        CALL_LOG.append({"label": label, "model": model, **usage, "truncated": truncated})
        log("USAGE", label=label, model=model, **usage, truncated=truncated)
        return text, truncated
    except Exception as error:
        log("ERROR", label=label, model=model, error=str(error)[:220])
        return "", False


async def individual_fallback(item: dict[str, str], plan: models.ModelPlan) -> str:
    messages, max_tokens = prompts.render(item["category"], item["prompt"])
    primary = models.model_for(item["category"], plan)
    answer, truncated = await try_complete("fallback:" + item["task_id"], primary, messages, max_tokens)
    if not answer.strip() or truncated:
        backup = models.fallback_model(item["category"], plan)
        if backup != primary:
            candidate, candidate_truncated = await try_complete(
                "fallback2:" + item["task_id"], backup, messages, max_tokens
            )
            if candidate.strip() and not candidate_truncated:
                answer = candidate
    return prompts.postprocess(item["category"], answer)


async def solve_batch(items: list[dict[str, str]], plan: models.ModelPlan, code_batch: bool) -> dict[str, str]:
    if not items or LOCAL_ONLY:
        return {}
    model = plan.CODE if code_batch else plan.REASON
    messages = batching.make_messages(items, code_batch=code_batch)
    cap = batching.batch_cap(items, code_batch=code_batch)
    label = "batch:code" if code_batch else "batch:general"
    text, truncated = await try_complete(label, model, messages, cap)
    categories = {item["task_id"]: item["category"] for item in items}
    answers = {} if truncated else batching.parse_answers(text, set(categories), categories)
    missing = [item for item in items if item["task_id"] not in answers]
    if missing:
        log("BATCH_MISSING", label=label, count=len(missing), ids=[x["task_id"] for x in missing])
        recovered = await asyncio.gather(*(individual_fallback(item, plan) for item in missing))
        for item, answer in zip(missing, recovered):
            if answer.strip():
                answers[item["task_id"]] = answer.strip()
    return answers


async def try_local_model(item: dict[str, str]) -> str:
    before = time.monotonic()
    try:
        answer = await asyncio.to_thread(local_model.complete, item["category"], item["prompt"])
        valid = local_model.validate(item["category"], item["prompt"], answer)
        LOCAL_MODEL_LOG.append({
            "task_id": item["task_id"],
            "category": item["category"],
            "valid": valid,
            "elapsed_s": round(time.monotonic() - before, 2),
        })
        log("LOCAL_MODEL", task_id=item["task_id"], category=item["category"], valid=valid,
            elapsed_s=round(time.monotonic() - before, 2))
        return prompts.postprocess(item["category"], answer) if valid else ""
    except Exception as error:
        LOCAL_MODEL_LOG.append({"task_id": item["task_id"], "category": item["category"],
                                "valid": False, "error": str(error)[:180]})
        log("LOCAL_MODEL_ERROR", task_id=item["task_id"], error=str(error)[:220])
        return ""


async def run(tasks: list[dict[str, Any]], results: dict[str, str]) -> None:
    plan = models.build_plan()
    MODEL_PLAN.update(plan.as_dict())
    log("MODEL_PLAN", **MODEL_PLAN, version=VERSION, local_model=local_model.MODEL_NAME,
        local_categories=sorted(local_model.CATEGORIES), local_only=LOCAL_ONLY)

    pending_general: list[dict[str, str]] = []
    pending_code: list[dict[str, str]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        prompt = _prompt(task)
        if not prompt:
            continue
        category = router.classify(prompt) or "factual"
        if ENABLE_LOCAL:
            local = solvers.solve(category, prompt)
            if local:
                results[task_id] = local
                log("LOCAL_RULE", task_id=task_id, category=category)
                continue
        item = {"task_id": task_id, "category": category, "prompt": prompt}
        (pending_code if category in {"debug", "codegen"} else pending_general).append(item)

    if ENABLE_LOCAL_MODEL:
        remaining_items: list[dict[str, str]] = []
        for item in pending_general + pending_code:
            remaining = DEADLINE_SECONDS - (time.monotonic() - START)
            should_try = local_model.enabled_for(item["category"]) or LOCAL_ONLY
            if should_try and remaining > LOCAL_MODEL_MIN_REMAINING:
                answer = await try_local_model(item)
                if answer:
                    results[item["task_id"]] = answer
                    continue
            remaining_items.append(item)
        pending_general = [x for x in remaining_items if x["category"] not in {"debug", "codegen"}]
        pending_code = [x for x in remaining_items if x["category"] in {"debug", "codegen"}]

    if LOCAL_ONLY:
        for item in pending_general + pending_code:
            if DEADLINE_SECONDS - (time.monotonic() - START) <= LOCAL_MODEL_MIN_REMAINING:
                break
            answer = await try_local_model(item)
            if answer:
                results[item["task_id"]] = answer
        await client.aclose()
        return

    if ENABLE_BATCH:
        remaining = DEADLINE_SECONDS - (time.monotonic() - START)
        try:
            general_answers, code_answers = await asyncio.wait_for(
                asyncio.gather(
                    solve_batch(pending_general, plan, code_batch=False),
                    solve_batch(pending_code, plan, code_batch=True),
                ),
                timeout=max(5, remaining),
            )
            results.update(general_answers)
            results.update(code_answers)
        except asyncio.TimeoutError:
            log("DEADLINE", elapsed=round(time.monotonic() - START, 1))
    else:
        all_pending = pending_general + pending_code
        answers = await asyncio.gather(*(individual_fallback(item, plan) for item in all_pending))
        for item, answer in zip(all_pending, answers):
            results[item["task_id"]] = answer
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
            "total_tokens": sum(int(call.get("prompt_tokens", 0)) + int(call.get("completion_tokens", 0)) for call in CALL_LOG),
            "fireworks_calls": len(CALL_LOG),
            "local_model_calls": len(LOCAL_MODEL_LOG),
        }
        path = os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "model_usage.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"version": VERSION, "local_model": local_model.MODEL_NAME,
                       "model_plan": MODEL_PLAN, "local_calls": LOCAL_MODEL_LOG,
                       "fireworks_calls": CALL_LOG, "totals": totals}, file,
                      ensure_ascii=False, indent=2)
    except Exception as error:
        log("WARN", error=str(error)[:180])


def main() -> int:
    try:
        with open(INPUT_PATH, encoding="utf-8-sig") as file:
            raw = json.load(file)
        if not isinstance(raw, list):
            raise ValueError("tasks.json must be a list")
        tasks = [task for task in raw if isinstance(task, dict) and task.get("task_id")]
        if not LOCAL_ONLY:
            required = ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
            if any(not os.environ.get(name) for name in required):
                raise ValueError("FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS are required")
        results = {str(task["task_id"]): "" for task in tasks}
        asyncio.run(run(tasks, results))
        write_results(tasks, results)
        write_usage_log()
        log("DONE", tasks=len(tasks), answered=sum(bool(value) for value in results.values()),
            elapsed_s=round(time.monotonic() - START, 1))
        return 0
    except Exception as error:
        log("FATAL", error=str(error)[:300])
        return 1


if __name__ == "__main__":
    sys.exit(main())
