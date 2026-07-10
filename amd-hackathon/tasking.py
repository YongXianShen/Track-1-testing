"""Task classification, category prompts, and output cleanup for Track 1."""
from __future__ import annotations

import math
import re

CATEGORIES = {"factual", "math", "sentiment", "summary", "ner", "debug", "logic", "codegen"}

_EXPLICIT = {
    "sentiment": re.compile(
        r"\bsentiment\b|\bpolarity\b|\b(classify|label|determine|identify|analy[sz]e|judge)\b.{0,60}\b(positive|negative|neutral|mixed|tone|mood)\b|\bpositive\s*(?:or|/)\s*negative\b",
        re.I,
    ),
    "summary": re.compile(
        r"summari[sz]e|summary|tl;?dr|condense|shorten|compress|recap|gist|\b(?:one|single|1)[ -]?(?:sentence|line)\b|\bin (?:one|two|three|four|five|\d+) (?:sentences?|words?|bullets?|lines?)\b",
        re.I,
    ),
    "ner": re.compile(
        r"named entit|\bNER\b|entities and their types|\b(?:extract|identify|find|list|tag|label|name)\b.{0,100}\b(?:persons?|people|organi[sz]ations?|companies|locations?|places?|cities|countries|dates?|entities)\b",
        re.I,
    ),
}

_STRONG_CODE = re.compile(r"```|\bdef\s+\w+|=>|print\(|console\.log|System\.out|#include|\bimport\s+\w+|\)\s*\{|\bSELECT\b.{0,80}\bFROM\b|\bclass\s+\w+", re.I)
_WEAK_CODE = re.compile(r"\bcode\b|\bsnippet\b|\bfunction\b|\bmethod\b|\bclass\b|\bprogram\b|\bscript\b|\balgorithm\b|\bregex\b|\bSQL\b|\bpython\b|\bjavascript\b|\btypescript\b|\bjava\b|\bc\+\+\b|\brust\b|\breturn\b|\bloop\b|\bquery\b", re.I)
_WRITE = re.compile(r"\b(write|implement|create|build|generate|develop|complete|define|provide)\b", re.I)
_FIX = re.compile(r"\bfix\b|\bdebug\b|\bbugs?\b|\bbuggy\b|\bbroken\b|off-by-one|doesn'?t work|not working|should\b.{0,60}\bbut\b|find and fix|what(?:'s| is) wrong|traceback|why .{0,45}(?:fail|crash|error|return|work)", re.I)
_DIGIT = re.compile(r"\d")
_MATH = re.compile(r"calculate|compute|evaluate|solve|convert|how (?:many|much|old|far|fast|long)|percent(?:age)?|%|average|mean|sum|total|remain|remaining|left|sold|discount|profit|ratio|interest|cost|price|items?|units?|times|plus|minus|divided|multiply|area|perimeter|volume|speed|distance|rate|increase|decrease|[+×÷]|half of|quarter of|double|triple|square root|squared|cubed|remainder|quotient|product of", re.I)
_STRONG_LOGIC = re.compile(r"logic|deductive|constraint|puzzle|riddle|clues?|truth[- ]?teller|liar|knight|knave|arrangement|seating|each.{0,45}different|who owns|owns the|satisfy all", re.I)
_WEAK_LOGIC = re.compile(r"adjacent|(?:directly |immediately )?(?:left|right) of|older than|younger than|first to last|finish(?:es|ed)?.{0,20}(?:before|after|first|last)|who (?:finished|came|won|placed|ranked|owns|is next)\b", re.I)


def classify(prompt: str) -> str:
    p = prompt.strip()
    for category, pattern in _EXPLICIT.items():
        if pattern.search(p):
            return category

    literal_code = bool(_STRONG_CODE.search(p))
    code_context = literal_code or bool(_WEAK_CODE.search(p))
    wants_write = bool(_WRITE.search(p))
    wants_fix = bool(_FIX.search(p))
    if code_context:
        if wants_write and not wants_fix:
            return "codegen"
        if wants_fix or literal_code:
            return "debug"
        if wants_write:
            return "codegen"

    has_math = bool(_DIGIT.search(p) and _MATH.search(p))
    strong_logic = bool(_STRONG_LOGIC.search(p))
    has_logic = strong_logic or bool(_WEAK_LOGIC.search(p))
    if has_math and has_logic:
        return "logic" if strong_logic else "math"
    if has_math:
        return "math"
    if has_logic:
        return "logic"
    return "factual"


_SYSTEM = {
    "factual": "Answer every part accurately and directly. Use essential context only. Stay under 100 words unless the task requires more. Do not invent facts.",
    "math": "Solve carefully. Give at most two compact calculation steps, then write exactly 'Answer: <final>' with units when relevant.",
    "sentiment": "Use the labels requested by the task. Otherwise choose Positive, Negative, Neutral, or Mixed. Give the label and one brief reason.",
    "summary": "Output only the summary. Obey every sentence, word, bullet, style, and length constraint exactly. Add no outside facts.",
    "ner": "Extract all and only the requested named entities. Preserve their exact spelling. Follow the requested format; otherwise use 'TYPE: Entity', one per line.",
    "debug": "State the bug in one short sentence, then give the corrected minimal implementation in the original language. Preserve the intended behavior.",
    "logic": "Use every constraint. Give at most two compact deductions, then write exactly 'Answer: <final>'.",
    "codegen": "Return only correct, minimal, self-contained code in the requested language. Handle all stated edge cases. No explanation unless requested.",
}

_MAX_TOKENS = {
    "factual": 120,
    "math": 120,
    "sentiment": 48,
    "summary": 160,
    "ner": 128,
    "debug": 240,
    "logic": 128,
    "codegen": 320,
}


def _summary_cap(prompt: str) -> int:
    low = prompt.lower()
    m = re.search(r"(?:exactly|in|under|at most|no more than)\s+(\d+)\s+words?", low)
    if m:
        words = max(1, min(int(m.group(1)), 250))
        return max(24, min(160, math.ceil(words * 1.55) + 12))
    m = re.search(r"(?:exactly|in|under|at most)\s+(\d+)\s+sentences?", low)
    if m:
        return max(48, min(160, 35 + 55 * max(1, min(int(m.group(1)), 6))))
    if re.search(r"\b(?:one|single|1)[ -]?sentence\b", low):
        return 96
    return 160


def render(category: str, prompt: str) -> tuple[list[dict[str, str]], int]:
    category = category if category in CATEGORIES else "factual"
    system = "You are a precise benchmark assistant. Answer in English. No preamble. Do not reveal private reasoning. " + _SYSTEM[category]
    cap = _summary_cap(prompt) if category == "summary" else _MAX_TOKENS[category]
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}], cap


def _strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.I | re.S)
    text = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.I | re.S)
    return text.strip()


def postprocess(category: str, text: str) -> str:
    text = _strip_reasoning(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if category == "sentiment":
        low = text.lower()
        for label in ("mixed", "positive", "negative", "neutral"):
            if re.search(rf"\b{label}\b", low):
                compact = re.sub(r"\s+", " ", text)
                return compact[:220]
    if category == "codegen":
        m = re.fullmatch(r"```[a-zA-Z0-9_+\-.#]*\s*\n(.*?)\n```", text, flags=re.S)
        if m:
            return m.group(1).strip()
    return text


def format_violation(category: str, prompt: str, answer: str) -> str | None:
    """Return a short repair instruction only for objective format failures."""
    if not answer.strip():
        return "The answer was empty. Produce the requested answer now."
    if category != "summary":
        return None
    low = prompt.lower()
    m = re.search(r"exactly\s+(\d+)\s+words?", low)
    if m:
        wanted = int(m.group(1))
        got = len(re.findall(r"\b[\w'-]+\b", answer))
        if got != wanted:
            return f"Rewrite the summary using exactly {wanted} words. The previous answer used {got}. Output only the corrected summary."
    if re.search(r"\b(?:exactly\s+)?(?:one|single|1)[ -]?sentence\b", low):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
        if len(sentences) != 1:
            return "Rewrite as exactly one sentence. Output only the corrected summary."
    return None
