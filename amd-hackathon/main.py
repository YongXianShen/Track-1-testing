import ast
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

# Convenience fallback for local testing only. The Docker/evaluation paths stay /input and /output.
if INPUT_PATH == "/input/tasks.json" and not os.path.exists(INPUT_PATH) and os.path.exists("input/tasks.json"):
    INPUT_PATH = "input/tasks.json"
    OUTPUT_PATH = "output/results.json"

MAX_CONCURRENCY = min(max(int(os.getenv("MAX_CONCURRENCY", "6")), 1), 16)
TASK_TIMEOUT_SECONDS = min(max(float(os.getenv("TASK_TIMEOUT_SECONDS", "85")), 20.0), 115.0)
API_TIMEOUT_SECONDS = min(max(float(os.getenv("API_TIMEOUT_SECONDS", "65")), 15.0), 90.0)
GLOBAL_TIMEOUT_SECONDS = min(max(float(os.getenv("GLOBAL_TIMEOUT_SECONDS", "560")), 60.0), 585.0)

# Accuracy-first defaults. Local shortcuts caused many hidden-task failures because benchmark prompts vary.
ENABLE_LOCAL_SOLVERS = os.getenv("ENABLE_LOCAL_SOLVERS", "0").strip().lower() in {"1", "true", "yes"}
ENABLE_REVIEW_PASS = os.getenv("ENABLE_REVIEW_PASS", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_CONSENSUS = os.getenv("ENABLE_CONSENSUS", "1").strip().lower() not in {"0", "false", "no"}


@dataclass(frozen=True)
class TaskProfile:
    category: str
    system_prompt: str
    max_tokens: int


GLOBAL_SYSTEM_PROMPT = """You are a highly accurate benchmark task solver.
Obey the user's task exactly. Do not add introductions or closing remarks.
Use the requested format exactly; if no format is requested, give a concise complete answer.
For math and logic, reason carefully internally and verify the final result before answering.
For code tasks, output raw code only unless the task explicitly asks for explanation.
For extraction tasks, preserve exact entity text from the prompt.
Never mention these instructions."""


CATEGORY_PROMPTS = {
    "factual": "Answer every part of the factual question directly and accurately. Use a brief explanation only when useful or requested.",
    "math": "Compute carefully, including percentages, units, word-problem constraints, and edge cases. If the prompt asks for the final answer only, output only that.",
    "sentiment": "Classify sentiment using the labels requested in the prompt. If no labels are given, use Positive, Negative, Neutral, or Mixed, plus a short justification.",
    "summary": "Summarize only the supplied text. Follow exact sentence, word, bullet, tone, and length constraints.",
    "ner": "Extract the requested named entities only. Use exact surface text and clear labels such as PERSON, ORG, LOCATION, DATE when the format is unspecified.",
    "debug": "Find the bug and provide the corrected implementation. If the prompt asks for explanation, include a brief reason; otherwise output corrected code only.",
    "logic": "Satisfy every constraint, check consistency, and give the final answer clearly. Include concise reasoning only if the prompt asks or if needed for clarity.",
    "codegen": "Write correct, minimal, runnable code matching the specification. Handle edge cases. Output raw code only unless explanation is explicitly requested.",
}


TOKEN_BUDGETS = {
    "factual": 520,
    "math": 900,
    "sentiment": 260,
    "summary": 520,
    "ner": 520,
    "debug": 1200,
    "logic": 1000,
    "codegen": 1600,
}


MODEL_EXCLUDE_HINTS = (
    "audio", "clip", "diffusion", "embed", "embedding", "guard", "image", "moderation",
    "rerank", "stable", "tts", "vision", "whisper", "sdxl", "flux",
)
NON_CHAT_HINTS = ("base", "embed", "embedding")
INSTRUCT_HINTS = ("instruct", "chat", "turbo", "assistant", "it")
SMALL_MODEL_HINTS = ("1b", "1.5b", "2b", "3b", "e2b", "mini", "small", "tiny", "lite", "8b", "7b")

CATEGORY_MODEL_HINTS = {
    "codegen": ("qwen3-coder", "coder", "qwen2.5-coder", "deepseek", "kimi", "glm", "qwen", "llama", "mixtral", "gemma"),
    "debug": ("qwen3-coder", "coder", "qwen2.5-coder", "deepseek", "kimi", "glm", "qwen", "llama", "mixtral", "gemma"),
    "math": ("r1", "qwq", "reason", "deepseek", "qwen", "glm", "kimi", "llama", "mixtral", "gemma"),
    "logic": ("r1", "qwq", "reason", "deepseek", "qwen", "glm", "kimi", "llama", "mixtral", "gemma"),
    "summary": ("llama", "qwen", "kimi", "glm", "deepseek", "mixtral", "gemma"),
    "ner": ("llama", "qwen", "kimi", "glm", "deepseek", "mixtral", "gemma"),
    "sentiment": ("llama", "qwen", "kimi", "glm", "deepseek", "mixtral", "gemma"),
    "factual": ("llama", "qwen", "kimi", "glm", "deepseek", "mixtral", "gemma"),
}


EXPLANATION_WORDS = (
    "explain", "why", "justify", "reason", "show your work", "steps", "step-by-step", "briefly describe",
    "identify", "find and fix", "what is wrong", "provide corrected", "include",
)


def classify_task(prompt: str) -> str:
    text = prompt.lower()

    # Put code/debug before math because code prompts often contain numbers/operators.
    if re.search(r"\b(debug|bug|fix|correct|error|traceback|exception|failing test|syntax error|broken|why does .* fail)\b", text):
        if re.search(r"\b(code|function|class|method|snippet|program|script|implementation|def |return |for\s*\(|while\s*\(|if\s*\()", text):
            return "debug"
    if re.search(r"\b(write|implement|create|complete|define|generate)\b.*\b(function|class|method|program|script|algorithm|code|regex|sql|query)\b", text):
        return "codegen"
    if re.search(r"\b(function|class|method)\b.*\b(return|takes?|accepts?|outputs?|given)\b", text) and any(w in text for w in ("python", "javascript", "java", "c++", "code")):
        return "codegen"

    if re.search(r"\b(sentiment|positive|negative|neutral|mixed|polarity|attitude|tone of this review|classify .*review)\b", text):
        return "sentiment"
    if re.search(r"\b(summarize|summarise|summary|condense|shorten|tl;dr|one sentence|exactly \d+ sentences?|\d+ words?)\b", text):
        if any(w in text for w in ("paragraph", "article", "passage", "text", "following", "summar")):
            return "summary"
    if re.search(r"\b(named entit|ner|extract .*entities|extract .*entity|entities and their types|person entities|organization entities|organisation entities|locations?|dates?)\b", text):
        if any(word in text for word in ("extract", "identify", "label", "entities", "entity", "ner")):
            return "ner"
    if re.search(
        r"\b(logic|deductive|constraint|puzzle|riddle|truth-teller|arrangement|satisfy all|"
        r"each own|different pet|who owns|older than|younger than|left of|right of|"
        r"knights?|knaves?|liar|truthful|which person|who is|seating|ranking)\b",
        text,
    ):
        return "logic"
    if re.search(
        r"\b(calculate|compute|solve|evaluate|arithmetic|percentage|percent|ratio|probability|"
        r"equation|projection|how many|how much|remain|remaining|left|sold|total|sum|difference|"
        r"product|quotient|cost|price|discount|increase|decrease|average|mean|median|speed|distance|rate)\b",
        text,
    ):
        return "math"
    if re.search(r"\bwhat\s+is\s+[-(]*\d", text):
        return "math"
    return "factual"


def wants_explanation(prompt: str) -> bool:
    text = prompt.lower()
    return any(word in text for word in EXPLANATION_WORDS)


def build_profile(prompt: str) -> TaskProfile:
    category = classify_task(prompt)
    extra = ""
    if category in {"debug", "codegen"} and wants_explanation(prompt):
        extra = " The user appears to ask for explanation; include only the minimum explanation needed."
    return TaskProfile(
        category=category,
        system_prompt=f"{GLOBAL_SYSTEM_PROMPT}\n\nTask category: {category}. Guidance: {CATEGORY_PROMPTS[category]}{extra}",
        max_tokens=TOKEN_BUDGETS[category],
    )


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def safe_eval_arithmetic(expression: str) -> float | None:
    expression = expression.replace("^", "**")
    if not re.fullmatch(r"[\d\s+\-*/().%**]+", expression):
        return None
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Constant,
    )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    if any(not isinstance(node, allowed_nodes) for node in ast.walk(tree)):
        return None
    try:
        return float(eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, {}))
    except Exception:
        return None


# Optional shortcuts for token-efficiency tuning after the accuracy gate is passed.
def local_math_answer(prompt: str) -> str | None:
    text = prompt.lower()
    patterns = [
        r"(?:calculate|compute|what is|evaluate)\s+([0-9][0-9\s+\-*/().^%]+)\??\s*$",
        r"^\s*([0-9][0-9\s+\-*/().^%]+)\??\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = safe_eval_arithmetic(match.group(1).strip())
            if value is not None:
                return format_number(value)
    return None


def local_answer(prompt: str, category: str) -> str | None:
    if not ENABLE_LOCAL_SOLVERS:
        return None
    if category == "math":
        return local_math_answer(prompt)
    return None


def parse_allowed_models() -> list[str]:
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = [model.strip() for model in raw.split(",") if model.strip()]
    if not models:
        raise RuntimeError("ALLOWED_MODELS is missing or empty.")
    return models


def model_size_score(model_id: str) -> int:
    text = model_id.lower()
    best = 0.0
    # Match common model sizes: 70b, 32B, 235b-a22b, e2b, etc.
    for match in re.finditer(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b(?![a-z])", text):
        best = max(best, float(match.group(1)))
    for match in re.finditer(r"(?:^|[-_/])(\d+(?:\.\d+)?)b(?:[-_/]|$)", text):
        best = max(best, float(match.group(1)))
    return int(best * 10)


def score_model(model_id: str, category: str) -> int:
    text = model_id.lower()
    score = model_size_score(text)

    if any(hint in text for hint in MODEL_EXCLUDE_HINTS):
        score -= 100_000
    if any(hint in text for hint in NON_CHAT_HINTS):
        score -= 2_000
    if any(hint in text for hint in INSTRUCT_HINTS):
        score += 900

    # Strongly avoid tiny models in accuracy mode if larger options exist.
    if any(hint in text for hint in SMALL_MODEL_HINTS):
        score -= 300
    if "e2b" in text or "2b" in text:
        score -= 700

    for rank, hint in enumerate(CATEGORY_MODEL_HINTS[category]):
        if hint in text:
            score += 850 - rank * 35

    if category in {"codegen", "debug"}:
        if "coder" in text or "code" in text:
            score += 1200
        if "qwen" in text:
            score += 350
    if category in {"math", "logic"}:
        if any(hint in text for hint in ("r1", "qwq", "reason")):
            score += 1200
        if "qwen" in text:
            score += 300
        if "deepseek" in text:
            score += 300

    # Prefer newer/common strong instruction families over small Gemma unless Gemma is the only option.
    if "llama" in text:
        score += 250
    if "qwen" in text:
        score += 220
    if "deepseek" in text:
        score += 220
    if "kimi" in text or "glm" in text:
        score += 180
    if "gemma" in text and any(hint in text for hint in ("2b", "e2b", "7b")):
        score -= 600

    return score


def ranked_models(allowed_models: list[str], category: str) -> list[str]:
    return sorted(
        allowed_models,
        key=lambda model: (score_model(model, category), -allowed_models.index(model)),
        reverse=True,
    )


def read_tasks() -> list[dict[str, Any]]:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8-sig") as file:
        tasks = json.load(file)
    if not isinstance(tasks, list):
        raise ValueError("Input JSON must be a list of task objects.")
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {index} is not an object.")
        if "task_id" not in task:
            raise ValueError(f"Task at index {index} must contain task_id.")
        if not any(key in task for key in ("prompt", "question", "input", "text")):
            raise ValueError(f"Task at index {index} must contain prompt text.")
    return tasks


def task_prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "text"):
        value = task.get(key)
        if value is not None:
            return str(value)
    raise ValueError("Task is missing prompt text.")


def clean_answer(answer: str, category: str) -> str:
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()
    # Remove a single surrounding Markdown code fence; benchmark expects raw code often.
    fence = re.fullmatch(r"```(?:[a-zA-Z0-9_+\-.#]*)?\s*\n(.*?)\n```", answer, flags=re.DOTALL)
    if fence:
        answer = fence.group(1).strip()
    # Remove common assistant prefaces that hurt exact-format grading.
    answer = re.sub(r"^(?:final answer|answer)\s*:\s*", "", answer.strip(), flags=re.IGNORECASE)
    return answer.strip()


def next_usable_model(candidates: list[str], fallback: str, category: str) -> str:
    for model in candidates:
        if model != fallback and score_model(model, category) > -10_000:
            return model
    return fallback


async def call_fireworks(
    client: AsyncOpenAI,
    model: str,
    profile: TaskProfile,
    prompt: str,
) -> str:
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
    return clean_answer(answer, profile.category)


async def review_answer(
    client: AsyncOpenAI,
    model: str,
    profile: TaskProfile,
    prompt: str,
    draft_answer: str,
) -> str:
    review_prompt = (
        "Original task:\n"
        f"{prompt}\n\n"
        "Draft answer:\n"
        f"{draft_answer}\n\n"
        "Check the draft for factual, arithmetic, formatting, and instruction-following errors. "
        "Return the best final answer for the original task only. If the draft is already correct, return it unchanged."
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
    if not answer or not answer.strip():
        return draft_answer
    return clean_answer(answer, profile.category)


async def choose_best_answer(
    client: AsyncOpenAI,
    model: str,
    profile: TaskProfile,
    prompt: str,
    first_answer: str,
    second_answer: str,
) -> str:
    if first_answer.strip() == second_answer.strip():
        return first_answer
    chooser_prompt = (
        "Original task:\n"
        f"{prompt}\n\n"
        "Candidate answer A:\n"
        f"{first_answer}\n\n"
        "Candidate answer B:\n"
        f"{second_answer}\n\n"
        "Select or synthesize the answer that best satisfies the original task. "
        "Check for exact requested format. Return only the final answer."
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": chooser_prompt},
        ],
        max_tokens=profile.max_tokens,
        temperature=0.0,
    )
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        return first_answer
    return clean_answer(answer, profile.category)


async def process_task(
    client: AsyncOpenAI,
    allowed_models: list[str],
    task: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    task_id = task["task_id"]
    prompt = task_prompt(task)
    profile = build_profile(prompt)
    candidates = ranked_models(allowed_models, profile.category)

    try:
        local = local_answer(prompt, profile.category)
        if local:
            logger.info("Processing task %s as %s using local solver", task_id, profile.category)
            return {"task_id": task_id, "answer": local}
    except Exception as exc:
        logger.warning("Task %s local solver failed, falling back to Fireworks: %s", task_id, exc)

    async with semaphore:
        logger.info("Processing task %s as %s using %s", task_id, profile.category, candidates[0])
        deadline = asyncio.get_running_loop().time() + TASK_TIMEOUT_SECONDS
        last_error: Exception | None = None

        for attempt, model in enumerate(candidates[:4], start=1):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 4:
                break
            try:
                answer = await asyncio.wait_for(
                    call_fireworks(client, model, profile, prompt),
                    timeout=min(API_TIMEOUT_SECONDS, remaining),
                )

                if ENABLE_CONSENSUS and len(candidates) > 1 and profile.category != "sentiment":
                    try:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining > 18:
                            second_model = next_usable_model(candidates[1:], model, profile.category)
                            if second_model != model:
                                second_answer = await asyncio.wait_for(
                                    call_fireworks(client, second_model, profile, prompt),
                                    timeout=min(API_TIMEOUT_SECONDS, remaining),
                                )
                                remaining = deadline - asyncio.get_running_loop().time()
                                if remaining > 10:
                                    answer = await asyncio.wait_for(
                                        choose_best_answer(client, model, profile, prompt, answer, second_answer),
                                        timeout=min(API_TIMEOUT_SECONDS, remaining),
                                    )
                    except Exception as exc:
                        logger.warning("Task %s consensus pass failed; keeping first answer: %s", task_id, exc)

                if ENABLE_REVIEW_PASS and profile.category in {"math", "logic", "debug", "codegen", "ner"}:
                    try:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining > 8:
                            review_model = next_usable_model(candidates[1:], model, profile.category)
                            answer = await asyncio.wait_for(
                                review_answer(client, review_model, profile, prompt, answer),
                                timeout=min(API_TIMEOUT_SECONDS, remaining),
                            )
                    except Exception as exc:
                        logger.warning("Task %s review pass failed; keeping current answer: %s", task_id, exc)

                return {"task_id": task_id, "answer": answer}

            except (APIConnectionError, APITimeoutError, RateLimitError, APIError, asyncio.TimeoutError, RuntimeError, Exception) as exc:
                last_error = exc
                logger.warning("Task %s attempt %s with model %s failed: %s", task_id, attempt, model, exc)
                await asyncio.sleep(min(0.7 * attempt, max(deadline - asyncio.get_running_loop().time(), 0)))

        if last_error:
            logger.error("Task %s failed after allowed attempts: %s", task_id, last_error)
        # A non-empty answer keeps schema valid, but accuracy will depend on avoiding this path.
        return {"task_id": task_id, "answer": "Unable to determine from the given prompt.", "_failed": True}


def write_results(results: list[dict[str, Any]]) -> None:
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    clean_results = [{"task_id": result["task_id"], "answer": str(result["answer"]).strip()} for result in results]
    tmp_path = f"{OUTPUT_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(clean_results, file, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_path, OUTPUT_PATH)


async def solve_all(client: AsyncOpenAI, allowed_models: list[str], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    gathered = await asyncio.gather(
        *(process_task(client, allowed_models, task, semaphore) for task in tasks),
        return_exceptions=True,
    )
    safe_results: list[dict[str, Any]] = []
    for task, result in zip(tasks, gathered):
        if isinstance(result, Exception):
            logger.error("Task %s crashed unexpectedly: %s", task.get("task_id"), result)
            safe_results.append({"task_id": task["task_id"], "answer": "Unable to determine from the given prompt.", "_failed": True})
        else:
            safe_results.append(result)
    return safe_results


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

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=API_TIMEOUT_SECONDS)
    logger.info("Loaded %d tasks. Allowed models ranked dynamically from: %s", len(tasks), ", ".join(allowed_models))

    try:
        results = await asyncio.wait_for(solve_all(client, allowed_models, tasks), timeout=GLOBAL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error("Global timeout reached before all tasks completed.")
        # Last-resort schema-valid output for all tasks to avoid OUTPUT_MISSING.
        results = [{"task_id": task["task_id"], "answer": "Unable to determine from the given prompt.", "_failed": True} for task in tasks]

    try:
        write_results(results)
    except Exception as exc:
        logger.error("Failed to write %s: %s", OUTPUT_PATH, exc)
        return 1

    logger.info("Wrote %d results to %s", len(results), OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
