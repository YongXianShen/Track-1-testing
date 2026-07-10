"""Robust router for Track 1 task categories.

The hidden set is only 19 tasks, so a single misroute can cost one full 5.26%.
Rules are intentionally conservative: explicit operations first, code before math,
and math only when there is both a number and a math signal.
"""
from __future__ import annotations

import re

Category = str

_EXPLICIT = {
    "sentiment": re.compile(
        r"\bsentiment\b"
        r"|\b(classify|label|categorize|categorise|determine|identify|rate|analyze|analyse)\b.{0,55}\b(positive|negative|neutral|mixed)\b"
        r"|\bpositive[,/ ]+negative[,/ ]+(or +)?neutral\b"
        r"|\bpositive or negative\b"
        r"|\bhow (do|does|would) (you|people|customers|users|they|the (customer|reviewer|user)) feel\b"
        r"|\b(classify|determine|identify|what is|what'?s|analyze|analyse|judge|assess).{0,35}\b(tone|mood|emotion|sentiment)\b",
        re.I,
    ),
    "summary": re.compile(
        r"\bsummari[sz]e\b|\bsummary\b|\btl;?dr\b|\bcondense\b|\bshorten\b|\bcompress\b|\brecap\b|\bgist\b"
        r"|\bin (one|two|three|four|five|\d+) (sentence|sentences|word|words|bullet|bullets|line|lines)\b"
        r"|\b(one|single|1)[ -]?(sentence|line)\b",
        re.I,
    ),
    "ner": re.compile(
        r"named entit|\bNER\b"
        r"|\b(extract|identify|find|list|tag|label|pull out|pick out|highlight)\b.{0,90}\b(persons?|people|organi[sz]ations?|orgs?\b|compan(y|ies)|locations?|places?|cit(y|ies)|countr(y|ies)|dates?|entit(y|ies))\b"
        r"|\b(persons?|people|organi[sz]ations?|companies|locations?|dates?)\b.{0,70}\bnamed\b"
        r"|\bwho (is|are) (mentioned|named|listed|referenced)\b",
        re.I,
    ),
}

_STRONG_CODE = re.compile(
    r"```|\bdef \w|=>|print\(|console\.log|System\.out|#include|\bimport \w|\)\s*\{|\bSELECT\b[^\n]{0,80}\bFROM\b",
    re.I,
)
_WEAK_CODE = re.compile(
    r"\bcode\b|\bsnippet\b|\bfunction\b|\breturn\b|\bclass \w|\bmethod\b|\bvariable\b|\bloop\b|\bregex\b|\bscript\b|\balgorithm\b|\bsort\b|\bSQL\b|\bquery\b|\bin (python|java|javascript|typescript|c\+\+|go|rust|ruby)\b",
    re.I,
)
_WRITE = re.compile(r"\b(write|implement|create|build|generate|develop|complete|define)\b|\bgive me\b|\bprovide\b", re.I)
_STRONG_FIX = re.compile(
    r"\bfix\b|\bdebug\b|\bbugs?\b|\bbuggy\b|\bbroken\b|off-by-one|doesn'?t work|not working|never (terminates|returns|works|ends|stops)"
    r"|should\b.{0,60}\bbut\b|\bspot the (problem|bug|error|issue)\b|\bwhat'?s? (is )?wrong\b|\bthrows?\b.{0,30}\b(error|exception)\b|\braises?\b.{0,30}\b(error|exception)\b"
    r"|why (does|do|is|are|doesn'?t)\b.{0,45}\b(work|returns?|crash(es)?|fails?|raises?|throws?|breaks?|output|print|behave|terminate|loop|error|exception)\b",
    re.I,
)
_FIX = re.compile(_STRONG_FIX.pattern + r"|\bcorrect(ed)?\b|\bincorrect\b|\bcrash(es|ed)?\b|\bfails?\b|\berror\b", re.I)

_DIGIT = re.compile(r"\d")
_MATH_SIGNAL = re.compile(
    r"\bcalculate\b|\bcompute\b|\bconvert\b|\bevaluate\b|\bsolve\b|\bhow (many|much|old|far|fast|long)\b"
    r"|\bpercent(age)?\b|%|\baverage of\b|\bmean of\b|\bsum of\b|\btotal\b|\bcompound\b|\binterest\b|\bdiscount\b|\bprofit\b|\bsales tax\b"
    r"|\bper (hour|day|week|month|year|item|unit|person)\b|\bspeed\b|\bratio\b|\bproportion\b|\btimes\b|\bdivided?\b|\bmultipl\b|\bplus\b|\bminus\b"
    r"|\barea\b|\bperimeter\b|\bvolume\b|[+×÷]"
    r"|\bhalf of\b|\bquarter of\b|\bdozen\b|\bdouble[sd]?\b|\btriple[sd]?\b|\btwice\b|\bsquare root\b|\bsquared\b|\bcubed\b|\bremainder\b|\bquotient\b|\bproduct of\b"
    r"|\badds? (up )?to\b|\bdiffer by\b|\bincreases? by\b|\bdecreases? by\b|\bgrows? by\b|\bremain(s|ing)?\b|\bleft\b|\bsold\b",
    re.I,
)

_STRONG_LOGIC = re.compile(
    r"\b(lying|liar|knight|knave)\b|truth[- ]?tell|\bclues?\b|\bconstraints?\b|\bdeduce\b|\bseat(ed|ing)\b|\barrangement\b|\beach\b.{0,50}\bdifferent\b|\blogic puzzle\b",
    re.I,
)
_NOT_TRIVIA = r"(?!.{0,35}\b(mountain|animal|country|countries|ocean|river|planet|building|city|structure|lake|desert|bird|fish|tree|species|continent|state|volcano|waterfall)\b)"
_WEAK_LOGIC = re.compile(
    r"\brow of\b|\bin a row\b|\bsits? in\b|\bstand(s|ing)? in\b|\badjacent\b|\b(directly |immediately )?(left|right) of\b"
    r"|\bfinish(es|ed)?\b.{0,25}\b(before|after|first|last)\b|\bwho (finished|came|won|wins|placed|ranked|owns|is next)\b(?!\s+to\b)"
    r"|\bwho is (the )?" + _NOT_TRIVIA + r"(tallest|shortest|oldest|youngest|fastest|slowest|biggest|smallest|first|last)\b"
    r"|\bmore than\b.{0,35}\bless than\b|\bfirst to last\b|\brank\b.{0,45}\b(first|last|order|from|to)\b|\b(all|some|no)\b \w+ \bare\b|\bowns? the\b",
    re.I,
)

LETTER_TO_CATEGORY = {
    "A": "factual",
    "B": "math",
    "C": "sentiment",
    "D": "summary",
    "E": "ner",
    "F": "debug",
    "G": "logic",
    "H": "codegen",
}


def classify(prompt: str) -> Category | None:
    for category, rx in _EXPLICIT.items():
        if rx.search(prompt):
            return category

    is_code = bool(_STRONG_CODE.search(prompt)) or bool(_WEAK_CODE.search(prompt) and (_WRITE.search(prompt) or _FIX.search(prompt)))
    if is_code:
        if _WRITE.search(prompt) and not _STRONG_FIX.search(prompt):
            return "codegen"
        if _FIX.search(prompt):
            return "debug"
        if _WRITE.search(prompt):
            return "codegen"
        return "debug"

    has_math = bool(_DIGIT.search(prompt)) and bool(_MATH_SIGNAL.search(prompt))
    strong_logic = bool(_STRONG_LOGIC.search(prompt))
    has_logic = strong_logic or bool(_WEAK_LOGIC.search(prompt))

    if has_math and has_logic:
        return "logic" if strong_logic else None
    if has_math:
        return "math"
    if has_logic:
        return "logic"
    return "factual"


def fallback_messages(prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "Classify into one letter only: A factual, B math, C sentiment, D summary, "
                "E named-entity extraction, F debugging, G logic puzzle, H code generation.\n"
                f"Task: {prompt[:420]}\nLetter:"
            ),
        }
    ]

_LETTER = re.compile(r"\b([A-H])\b")


def parse_fallback_letter(text: str) -> Category:
    match = _LETTER.search((text or "").upper())
    return LETTER_TO_CATEGORY[match.group(1)] if match else "factual"
