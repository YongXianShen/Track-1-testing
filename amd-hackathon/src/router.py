"""Conservative router for the eight Track 1 task categories.

V15 keeps V12's proven regex-first design but fixes several high-risk misroutes:
- "write a function with error handling" is code generation, not debugging;
- "positive or negative" is sentiment even when the word sentiment is absent;
- generic factual uses of words such as "order" are not logic puzzles;
- math still requires both a number and an arithmetic signal.
"""
from __future__ import annotations

import re

Category = str

EXPLICIT = {
    "sentiment": re.compile(
        r"\bsentiment\b|\bpolarity\b"
        r"|\b(classify|label|categor(?:ize|ise)|determine|identify|analy[sz]e|judge|assess)\b.{0,50}\b(positive|negative|neutral|mixed|tone|mood|emotion)\b"
        r"|\bpositive\s+(?:or|/)\s+negative\b"
        r"|\bpositive[,/ ]+negative[,/ ]+(?:or\s+)?neutral\b"
        r"|\bhow (?:does|do|would) (?:the )?(?:customer|reviewer|user|people|they) feel\b"
        r"|\b(?:satisfied|happy|pleased)\s+(?:or|/)\s+(?:dissatisfied|unhappy|disappointed)\b",
        re.I,
    ),
    "summary": re.compile(
        r"summari[sz]e|summary|tl;?dr|condense|shorten|compress|recap|gist"
        r"|\b(?:one|single|1)[ -]?(?:sentence|line)\b"
        r"|\bin (?:one|two|three|four|five|\d+) (?:sentences?|words?|bullets?|lines?)\b",
        re.I,
    ),
    "ner": re.compile(
        r"named entit|\bNER\b|entities and their types"
        r"|\b(?:extract|identify|find|list|tag|label|name)\b.{0,90}\b(?:persons?|people|organi[sz]ations?|companies|locations?|places?|cities|countries|dates?|entities)\b"
        r"|\bwho (?:is|are) (?:mentioned|named|listed|referenced)\b",
        re.I,
    ),
}

STRONG_CODE = re.compile(
    r"```|\bdef\s+\w+|=>|print\(|console\.log|System\.out|#include|\bimport\s+\w+"
    r"|\)\s*\{|\bSELECT\b.{0,80}\bFROM\b|\bclass\s+\w+",
    re.I,
)
WEAK_CODE = re.compile(
    r"\bcode\b|\bsnippet\b|\bfunction\b|\bmethod\b|\bclass\b|\bprogram\b|\bscript\b"
    r"|\balgorithm\b|\bregex\b|\bSQL\b|\bpython\b|\bjavascript\b|\btypescript\b"
    r"|\bjava\b|\bc\+\+\b|\bgo\b|\brust\b|\breturn\b|\bloop\b|\bquery\b",
    re.I,
)
WRITE = re.compile(r"\b(write|implement|create|build|generate|develop|complete|define|provide|show)\b", re.I)
STRONG_FIX = re.compile(
    r"\bfix\b|\bdebug\b|\bbugs?\b|\bbuggy\b|\bbroken\b|off-by-one"
    r"|doesn'?t work|not working|never (?:terminates|returns|works|ends|stops)"
    r"|should\b.{0,55}\bbut\b|find and fix|spot the (?:problem|bug|error|issue)"
    r"|what'?s? (?:is )?wrong|what is wrong|traceback"
    r"|why (?:does|do|is|are|doesn'?t)\b.{0,45}\b(?:work|return|crash|fail|raise|throw|break|output|print|behave|terminate|loop|error|exception)",
    re.I,
)
WEAK_FIX = re.compile(r"\bincorrect\b|\bcrash(?:es|ed)?\b|\bfails?\b|\berror\b|\bexception\b", re.I)

DIGIT = re.compile(r"\d")
MATH_SIGNAL = re.compile(
    r"calculate|compute|evaluate|solve|convert|how (?:many|much|old|far|fast|long)"
    r"|percent(?:age)?|%|average|mean|sum|total|remain|remaining|left|sold|discount"
    r"|profit|ratio|interest|cost|price|items?|units?|times|plus|minus|divided|multiply"
    r"|area|perimeter|volume|speed|distance|rate|increase|decrease|[+×÷]"
    r"|half of|quarter of|double|triple|square root|squared|cubed|remainder|quotient|product of",
    re.I,
)
STRONG_LOGIC = re.compile(
    r"logic|deductive|constraint|puzzle|riddle|clues?|truth[- ]?teller|liar|knight|knave"
    r"|arrangement|seating|each.{0,45}different|who owns|owns the|satisfy all",
    re.I,
)
WEAK_LOGIC = re.compile(
    r"adjacent|(?:directly |immediately )?(?:left|right) of|older than|younger than"
    r"|first to last|finish(?:es|ed)?.{0,20}(?:before|after|first|last)"
    r"|who (?:finished|came|won|placed|ranked|owns|is next)\b|rank.{0,35}(?:first|last|from|to)",
    re.I,
)


def classify(prompt: str) -> Category | None:
    p = prompt.strip()
    for category, rx in EXPLICIT.items():
        if rx.search(p):
            return category

    literal_code = bool(STRONG_CODE.search(p))
    code_context = literal_code or bool(WEAK_CODE.search(p))
    wants_write = bool(WRITE.search(p))
    strong_fix = bool(STRONG_FIX.search(p))
    weak_fix = bool(WEAK_FIX.search(p))

    if code_context:
        # A request to write new code remains codegen even if it mentions
        # "errors" or "exceptions" as features. Only explicit broken-code
        # language overrides that intent.
        if wants_write and not strong_fix:
            return "codegen"
        if strong_fix:
            return "debug"
        if literal_code and weak_fix:
            return "debug"
        if wants_write:
            return "codegen"
        if literal_code:
            return "debug"

    has_math = bool(DIGIT.search(p)) and bool(MATH_SIGNAL.search(p))
    strong_logic = bool(STRONG_LOGIC.search(p))
    has_logic = strong_logic or bool(WEAK_LOGIC.search(p))

    if has_math and has_logic:
        return "logic" if strong_logic else "math"
    if has_math:
        return "math"
    if has_logic:
        return "logic"
    return "factual"


def fallback_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Return one letter only."},
        {
            "role": "user",
            "content": "A factual, B math, C sentiment, D summary, E NER, F debugging, G logic, H code generation.\nTask: " + prompt[:500],
        },
    ]


def parse_fallback_letter(text: str) -> str:
    match = re.search(r"\b([A-H])\b", (text or "").upper())
    return {
        "A": "factual",
        "B": "math",
        "C": "sentiment",
        "D": "summary",
        "E": "ner",
        "F": "debug",
        "G": "logic",
        "H": "codegen",
    }.get(match.group(1) if match else "", "factual")
