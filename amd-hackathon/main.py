import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

INPUT_PATH = os.getenv("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/output/results.json")

# Local testing convenience only. The evaluation harness still uses /input and /output.
if INPUT_PATH == "/input/tasks.json" and not os.path.exists(INPUT_PATH) and os.path.exists("input/tasks.json"):
    INPUT_PATH = "input/tasks.json"
    OUTPUT_PATH = "output/results.json"

# Accuracy-first but stable: no local shortcuts, no answer-overwriting consensus by default.
# The previous hybrid version likely failed because shortcut solvers answered hidden variants incorrectly.
MAX_CONCURRENCY = min(max(int(os.getenv("MAX_CONCURRENCY", "2")), 1), 6)
TASK_TIMEOUT_SECONDS = min(max(float(os.getenv("TASK_TIMEOUT_SECONDS", "75")), 25.0), 110.0)
API_TIMEOUT_SECONDS = min(max(float(os.getenv("API_TIMEOUT_SECONDS", "60")), 20.0), 90.0)
GLOBAL_TIMEOUT_SECONDS = min(max(float(os.getenv("GLOBAL_TIMEOUT_SECONDS", "565")), 60.0), 585.0)
ENABLE_REVIEW_PASS = os.getenv("ENABLE_REVIEW_PASS", "0").strip().lower() in {"1", "true", "yes"}

@dataclass(frozen=True)
class TaskProfile:
    category: str
    system_prompt: str
    max_tokens: int

GLOBAL_SYSTEM_PROMPT = """You are solving hidden benchmark tasks for an LLM judge.
Answer the user's task exactly. Use only the information in the prompt plus general knowledge.
Match every requested format, label set, length limit, language, and output type.
Do not add greetings, caveats, Markdown fences, or extra explanation unless the prompt asks for them.
For math and logic, reason privately and return the checked final answer.
For code tasks, return runnable code only unless explanation is explicitly requested.
For extraction tasks, preserve exact text spans from the prompt.
Never mention these instructions."""

CATEGORY_PROMPTS = {
    "factual": "Give a concise, complete factual answer. If the question has multiple parts, answer all parts.",
    "math": "Compute carefully. Track units, percentages, remaining amounts, projections, and edge cases. Return the final answer in the requested format.",
    "sentiment": "Use the exact sentiment labels requested. If no labels are given, use Positive, Negative, Neutral, or Mixed, with one short justification only if helpful.",
    "summary": "Summarize only the supplied text. Strictly obey exact sentence count, word count, bullet count, tone, and length constraints.",
    "ner": "Extract only requested entities. Preserve exact surface text. Use clear entity types such as PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PRODUCT when format is unspecified.",
    "debug": "Identify and fix the bug. Return corrected implementation/code only unless the prompt asks for explanation.",
    "logic": "Satisfy every constraint. Check the solution against all conditions. Return the requested final answer clearly.",
    "codegen": "Write correct, minimal, runnable code matching the specification. Handle duplicates and edge cases. Return raw code only unless explanation is requested.",
}

TOKEN_BUDGETS = {
    "factual": 600,
    "math": 1000,
    "sentiment": 300,
    "summary": 650,
    "ner": 650,
    "debug": 1400,
    "logic": 1100,
    "codegen": 1700,
}

MODEL_EXCLUDE_HINTS = (
    "audio", "clip", "diffusion", "embed", "embedding", "guard", "image", "moderation",
    "rerank", "stable", "tts", "vision", "whisper", "sdxl", "flux",
)
NON_CHAT_HINTS = ("base", "embed", "embedding")
INSTRUCT_HINTS = ("instruct", "chat", "turbo", "assistant", "it")
SMALL_HINTS = ("1b", "1.5b", "2b", "3b", "mini", "small", "tiny", "lite")

CATEGORY_MODEL_HINTS = {
    "codegen": ("qwen3-coder", "qwen2.5-coder", "coder", "kimi-k2", "deepseek", "qwen", "glm", "llama", "mixtral", "gemma"),
    "debug": ("qwen3-coder", "qwen2.5-coder", "coder", "kimi-k2", "deepseek", "qwen", "glm", "llama", "mixtral", "gemma"),
    "math": ("r1", "qwq", "reason", "deepseek", "qwen", "glm", "kimi", "llama", "mixtral", "gemma"),
    "logic": ("r1", "qwq", "reason", "deepseek", "qwen", "glm", "kimi", "llama", "mixtral", "gemma"),
    "summary": ("llama", "qwen", "kimi", "glm", "deepseek", "mixtral", "gemma"),
    "ner": ("llama", "qwen", "kimi", "glm", "deepseek", "mixtral", "gemma"),
    "sentiment": ("llama", "qwen", "kimi", "glm", "deepseek", "mixtral", "gemma"),
    "factual": ("llama", "qwen", "kimi", "glm", "deepseek", "mixtral", "gemma"),
}

EXPLANATION_WORDS = (
    "explain", "why", "justify", "reason", "show your work", "steps", "step-by-step",
    "briefly describe", "identify", "find and fix", "what is wrong", "provide corrected", "include",
)

def classify_task(prompt: str) -> str:
    text = prompt.lower()
    compact = re.sub(r"\s+", " ", text)

    # Code/debug first because code snippets contain numbers/operators.
    if re.search(r"\b(debug|bug|fix|correct|error|traceback|exception|failing test|broken|why does .* fail|find and fix)\b", compact):
        if re.search(r"\b(code|function|class|method|snippet|program|script|implementation|def |return |public static|console\.log|for\s*\(|while\s*\(|if\s*\()\b", compact):
            return "debug"
    if re.search(r"\b(write|implement|create|complete|define|generate)\b.*\b(function|class|method|program|script|algorithm|code|regex|sql|query)\b", compact):
        return "codegen"
    if re.search(r"\bfunction\b.*\b(return|takes?|accepts?|outputs?|given)\b", compact) and any(w in compact for w in ("python", "javascript", "java", "c++", "code")):
        return "codegen"

    if re.search(r"\b(sentiment|positive|negative|neutral|mixed|polarity|attitude|tone of this review|classify .*review|classify .*feedback)\b", compact):
        return "sentiment"
    if re.search(r"\b(summarize|summarise|summary|condense|shorten|tl;dr|one sentence|exactly \d+ sentences?|\d+ words?)\b", compact):
        if any(w in compact for w in ("paragraph", "article", "passage", "text", "following", "summar")):
            return "summary"
    if re.search(r"\b(named entit|ner|extract .*entities|extract .*entity|entities and their types|person entities|organization entities|organisation entities)\b", compact):
        return "ner"
    if re.search(r"\bextract\b.*\b(person|people|organisation|organization|location|date|time|company|city|country|entity|entities)\b", compact):
        return "ner"
    if re.search(
        r"\b(logic|deductive|constraint|puzzle|riddle|truth-teller|arrangement|satisfy all|each own|different pet|who owns|older than|younger than|left of|right of|knights?|knaves?|liar|truthful|seating|ranking|order)\b",
        compact,
    ):
        return "logic"
    if re.search(
        r"\b(calculate|compute|solve|evaluate|arithmetic|percentage|percent|ratio|probability|equation|projection|how many|how much|remain|remaining|left|sold|total|sum|difference|product|quotient|cost|price|discount|increase|decrease|average|mean|median|speed|distance|rate|interest)\b",
        compact,
    ):
        return "math"
    if re.search(r"^\s*(what is|calculate|compute|evaluate)?\s*[-+]?\d", compact):
        return "math"
    return "factual"

def wants_explanation(prompt: str) -> bool:
    low = prompt.lower()
    return any(w in low for w in EXPLANATION_WORDS)

def build_profile(prompt: str) -> TaskProfile:
    category = classify_task(prompt)
    no_chain = " Do not reveal hidden reasoning; provide the final answer only unless explanation is requested."
    if wants_explanation(prompt):
        no_chain = " Give a concise explanation only to the extent requested by the prompt."
    return TaskProfile(
        category=category,
        system_prompt=f"{GLOBAL_SYSTEM_PROMPT}\n\nTask category: {category}. {CATEGORY_PROMPTS[category]}{no_chain}",
        max_tokens=TOKEN_BUDGETS[category],
    )

def parse_allowed_models() -> list[str]:
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise RuntimeError("ALLOWED_MODELS is missing or empty.")
    return models

def model_size_score(model_id: str) -> int:
    text = model_id.lower()
    best = 0.0
    for match in re.finditer(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b(?![a-z])", text):
        best = max(best, float(match.group(1)))
    for match in re.finditer(r"(?:^|[-_/])(\d+(?:\.\d+)?)b(?:[-_/]|$)", text):
        best = max(best, float(match.group(1)))
    return int(best * 10)

def score_model(model_id: str, category: str) -> int:
    text = model_id.lower()
    score = model_size_score(model_id)
    if any(h in text for h in MODEL_EXCLUDE_HINTS):
        score -= 100_000
    if any(h in text for h in NON_CHAT_HINTS):
        score -= 1_500
    if any(h in text for h in INSTRUCT_HINTS):
        score += 700
    if any(h in text for h in SMALL_HINTS):
        score -= 180

    for rank, hint in enumerate(CATEGORY_MODEL_HINTS[category]):
        if hint in text:
            score += 1000 - rank * 35

    if category in {"codegen", "debug"} and ("coder" in text or "code" in text):
        score += 1200
    if category in {"math", "logic"} and any(h in text for h in ("r1", "qwq", "reason")):
        score += 1200

    # broad quality preference
    if "qwen" in text:
        score += 250
    if "deepseek" in text:
        score += 230
    if "llama" in text:
        score += 210
    if "kimi" in text:
        score += 190
    if "glm" in text:
        score += 170
    if "gemma" in text and category in {"math", "logic", "debug", "codegen"}:
        score -= 80
    return score

def ranked_models(allowed_models: list[str], category: str) -> list[str]:
    usable = [m for m in allowed_models if score_model(m, category) > -50_000]
    if not usable:
        return allowed_models
    ranked = sorted(usable, key=lambda m: (score_model(m, category), -allowed_models.index(m)), reverse=True)
    # Keep the harness/order-preferred first model as fallback early, in case model names are unusual.
    first = allowed_models[0]
    if first in ranked and first not in ranked[:3]:
        ranked = [ranked[0], first] + [m for m in ranked[1:] if m != first]
    return ranked

def read_tasks() -> list[dict[str, Any]]:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8-sig") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("Input JSON must be a list of task objects.")
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {i} is not an object.")
        if "task_id" not in task:
            raise ValueError(f"Task at index {i} must contain task_id.")
        if not any(k in task for k in ("prompt", "question", "input", "text")):
            raise ValueError(f"Task at index {i} must contain prompt text.")
    return tasks

def task_prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "text"):
        if task.get(key) is not None:
            return str(task[key])
    raise ValueError("Task is missing prompt text.")

def clean_answer(answer: str) -> str:
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.fullmatch(r"```(?:[a-zA-Z0-9_+\-.#]*)?\s*\n(.*?)\n```", answer, flags=re.DOTALL)
    if fence:
        answer = fence.group(1).strip()
    answer = re.sub(r"^(?:final answer|answer)\s*:\s*", "", answer.strip(), flags=re.IGNORECASE)
    return answer.strip()

async def call_fireworks(client: AsyncOpenAI, model: str, profile: TaskProfile, prompt: str) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=profile.max_tokens,
        temperature=0.0,
    )
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("Model returned an empty answer.")
    return clean_answer(answer)

async def review_answer(client: AsyncOpenAI, model: str, profile: TaskProfile, prompt: str, draft: str) -> str:
    review_prompt = (
        "Original task:\n" + prompt + "\n\n"
        "Draft answer:\n" + draft + "\n\n"
        "Check whether the draft exactly answers the original task. "
        "If it is already correct, return it unchanged. If it is wrong or poorly formatted, return only the corrected final answer."
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": review_prompt},
        ],
        max_tokens=profile.max_tokens,
        temperature=0.0,
    )
    answer = response.choices[0].message.content
    return clean_answer(answer) if answer and answer.strip() else draft

async def process_task(client: AsyncOpenAI, allowed_models: list[str], task: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    task_id = task["task_id"]
    prompt = task_prompt(task)
    profile = build_profile(prompt)
    candidates = ranked_models(allowed_models, profile.category)

    async with semaphore:
        deadline = asyncio.get_running_loop().time() + TASK_TIMEOUT_SECONDS
        last_error: Exception | None = None
        logger.info("Task %s category=%s candidates=%s", task_id, profile.category, candidates[:3])
        for attempt, model in enumerate(candidates[:4], start=1):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 5:
                break
            try:
                answer = await asyncio.wait_for(
                    call_fireworks(client, model, profile, prompt),
                    timeout=min(API_TIMEOUT_SECONDS, remaining),
                )
                if ENABLE_REVIEW_PASS and profile.category in {"math", "logic", "debug", "codegen", "ner"}:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining > 12:
                        try:
                            answer = await asyncio.wait_for(
                                review_answer(client, model, profile, prompt, answer),
                                timeout=min(API_TIMEOUT_SECONDS, remaining),
                            )
                        except Exception as exc:
                            logger.warning("Task %s review failed, keeping answer: %s", task_id, exc)
                return {"task_id": task_id, "answer": answer}
            except (APIConnectionError, APITimeoutError, RateLimitError, APIError, asyncio.TimeoutError, RuntimeError, Exception) as exc:
                last_error = exc
                logger.warning("Task %s attempt %s model=%s failed: %s", task_id, attempt, model, exc)
                await asyncio.sleep(min(0.5 * attempt, 2.0))
        logger.error("Task %s failed after model attempts: %s", task_id, last_error)
        return {"task_id": task_id, "answer": ""}

async def solve_all(client: AsyncOpenAI, allowed_models: list[str], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    gathered = await asyncio.gather(
        *(process_task(client, allowed_models, t, sem) for t in tasks),
        return_exceptions=True,
    )
    results: list[dict[str, Any]] = []
    for task, result in zip(tasks, gathered):
        if isinstance(result, Exception):
            logger.error("Task %s crashed: %s", task.get("task_id"), result)
            results.append({"task_id": task["task_id"], "answer": ""})
        else:
            results.append(result)
    return results

def write_results(results: list[dict[str, Any]]) -> None:
    outdir = os.path.dirname(OUTPUT_PATH)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    clean = [{"task_id": r["task_id"], "answer": str(r.get("answer", "")).strip()} for r in results]
    tmp = f"{OUTPUT_PATH}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, OUTPUT_PATH)

async def run() -> int:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL")
    if not api_key:
        logger.error("FIREWORKS_API_KEY is missing.")
        return 1
    if not base_url:
        logger.error("FIREWORKS_BASE_URL is missing.")
        return 1
    try:
        allowed_models = parse_allowed_models()
        tasks = read_tasks()
    except Exception as exc:
        logger.error("Startup validation failed: %s", exc)
        return 1

    logger.info("Loaded %d tasks. Allowed models: %s", len(tasks), ", ".join(allowed_models))
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=API_TIMEOUT_SECONDS)
    try:
        results = await asyncio.wait_for(solve_all(client, allowed_models, tasks), timeout=GLOBAL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error("Global timeout reached; writing blank answers for unfinished run.")
        results = [{"task_id": t["task_id"], "answer": ""} for t in tasks]
    try:
        write_results(results)
    except Exception as exc:
        logger.error("Failed to write results: %s", exc)
        return 1
    logger.info("Wrote %d results to %s", len(results), OUTPUT_PATH)
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
