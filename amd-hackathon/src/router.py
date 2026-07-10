"""Conservative task router for the 8 Track 1 categories."""
from __future__ import annotations

import re

Category = str

EXPLICIT = {
    "sentiment": re.compile(r"\bsentiment\b|\bpolarity\b|classify.{0,60}\b(positive|negative|neutral|mixed)\b|\bpositive[,/ ]+negative[,/ ]+(or )?neutral\b|\b(review|feedback).{0,60}\b(sentiment|tone)\b", re.I),
    "summary": re.compile(r"summari[sz]e|summary|tl;?dr|condense|shorten|recap|gist|exactly \d+ sentence|one sentence|single sentence|in \d+ words|in \d+ bullets", re.I),
    "ner": re.compile(r"named entit|\bNER\b|entities and their types|extract.{0,90}(person|people|organi[sz]ation|company|location|city|country|date|entity|entities)|identify.{0,90}(person|people|organi[sz]ation|company|location|date|entities)", re.I),
}

STRONG_CODE = re.compile(r"```|\bdef\s+\w+|#include|console\.log|System\.out|\bSELECT\b.{0,80}\bFROM\b|\bclass\s+\w+|\bimport\s+\w+|\)\s*\{", re.I)
WEAK_CODE = re.compile(r"\bcode\b|\bsnippet\b|\bfunction\b|\bmethod\b|\bclass\b|\bprogram\b|\bscript\b|\balgorithm\b|\bregex\b|\bSQL\b|\bpython\b|\bjavascript\b|\bjava\b|\bc\+\+\b", re.I)
WRITE = re.compile(r"\b(write|implement|create|build|generate|develop|complete|define|provide)\b", re.I)
FIX = re.compile(r"\bfix\b|\bdebug\b|\bbug\b|\bbuggy\b|\bbroken\b|wrong|incorrect|not working|fails?|error|exception|should.{0,60}but|find and fix|what'?s wrong", re.I)

DIGIT = re.compile(r"\d")
MATH_SIGNAL = re.compile(r"calculate|compute|evaluate|solve|how (many|much|old|far|fast|long)|percent|percentage|%|average|mean|sum|total|remain|remaining|left|sold|discount|profit|ratio|interest|cost|price|items?|units?|times|plus|minus|divided|multiply|area|perimeter|volume|speed|distance|rate|increase|decrease|[+×÷]", re.I)
LOGIC_SIGNAL = re.compile(r"logic|deductive|constraint|puzzle|riddle|clue|truth|liar|knight|knave|arrangement|seating|ranking|order|each.{0,45}different|who owns|owns the|left of|right of|older than|younger than|first to last|satisfy", re.I)


def classify(prompt: str) -> Category | None:
    p = prompt.strip()
    for cat, rx in EXPLICIT.items():
        if rx.search(p):
            return cat

    is_code = bool(STRONG_CODE.search(p)) or bool(WEAK_CODE.search(p) and (WRITE.search(p) or FIX.search(p)))
    if is_code:
        if FIX.search(p):
            return "debug"
        if WRITE.search(p):
            return "codegen"
        return "debug"

    has_math = bool(DIGIT.search(p)) and bool(MATH_SIGNAL.search(p))
    has_logic = bool(LOGIC_SIGNAL.search(p))
    if has_logic and not (has_math and not re.search(r"clue|constraint|logic|puzzle|each.{0,45}different|who owns|arrangement|seating", p, re.I)):
        return "logic"
    if has_math:
        return "math"
    if has_logic:
        return "logic"
    return "factual"


def fallback_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Return one letter only."},
        {"role": "user", "content": "A factual, B math, C sentiment, D summary, E NER, F debugging, G logic, H code generation.\nTask: " + prompt},
    ]


def parse_fallback_letter(text: str) -> str:
    letter = (text or "").strip().upper()[:1]
    return {
        "A": "factual", "B": "math", "C": "sentiment", "D": "summary", "E": "ner", "F": "debug", "G": "logic", "H": "codegen",
    }.get(letter, "factual")
