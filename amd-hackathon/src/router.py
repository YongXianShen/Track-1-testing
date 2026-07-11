"""Precision router for the eight Track 1 categories.

The rules are intentionally conservative: explicit operations first, code before
math/logic, and factual as the safe default. No model call is needed for normal routing.
"""
from __future__ import annotations

import re

Category = str

# Explicit operation categories.  These are checked before code/math/logic because
# their instruction verbs are normally unambiguous.
_SENTIMENT = re.compile(
    r"\bsentiment\b(?!\s+analysis\b(?!\s+(?:on|of)\b))"
    r"|\b(?:classify|label|categor(?:y|ize|ise)|determine|identify|rate|assess|analy[sz]e)\b"
    r".{0,55}\b(?:positive|negative|neutral|mixed)\b"
    r"|\bpositive\s*(?:,|/|or)\s*negative(?:\s*(?:,|/|or)\s*(?:neutral|mixed))?\b"
    r"|\b(?:tone|mood|emotion)\s+of\s+(?:this|the following|the)\s+"
    r"(?:review|comment|text|message|feedback|passage|paragraph)\b",
    re.I,
)
_SUMMARY = re.compile(
    r"\bsummari[sz]e\b|\bsummary\b|\btl;?dr\b|\bcondense\b|\bshorten\b|\brecap\b|\bgist\b"
    r"|\bboil\b.{0,25}\bdown\b|\b(?:one|single|1|two|2|three|3|\d+)\s*[- ]?"
    r"(?:sentence|word|bullet|line)s?\b",
    re.I,
)
_NER = re.compile(
    r"named entit|\bNER\b"
    r"|\b(?:extract|identify|find|list|tag|label|pull out|highlight)\b.{0,90}"
    r"\b(?:persons?|people|organi[sz]ations?|orgs?|companies|locations?|places?|cities|countries|dates?|entities)\b"
    r"|\bwho (?:is|are) (?:mentioned|named|listed|referenced)\b",
    re.I,
)

# Code is recognized only from literal code or a code artifact paired with a
# write/fix intent, preventing conceptual questions about functions from misrouting.
_STRONG_CODE = re.compile(
    r"```|\bdef\s+\w+|=>|print\s*\(|console\.log|System\.out|#include|\bimport\s+\w+"
    r"|\)\s*\{|\bSELECT\b[^\n]{0,80}\bFROM\b|\bclass\s+\w+\s*[:{]",
    re.I,
)
_WEAK_CODE = re.compile(
    r"\bcode\b|\bsnippet\b|\bfunction\b|\bmethod\b|\bscript\b|\bprogram\b|\bimplementation\b"
    r"|\bregex\b|\bquery\b|\bSQL\b|\bAPI endpoint\b|\bparser\b|\bdecorator\b"
    r"|\bin\s+(?:python|java|javascript|typescript|c\+\+|go|rust|ruby)\b",
    re.I,
)
_WRITE = re.compile(
    r"\b(?:write|implement|create|build|generate|develop|complete)\b"
    r"|\b(?:give|provide|show)\s+(?:me\s+)?(?:a|an|the)?\s*(?:\w+\s+){0,2}"
    r"(?:function|method|script|program|class|snippet|implementation|regex|query|endpoint|parser|code)\b",
    re.I,
)
_STRONG_FIX = re.compile(
    r"\bfix\b|\bdebug\b|\bbugs?\b|\bbuggy\b|\bbroken\b|off[- ]by[- ]one|doesn'?t work|not working"
    r"|never (?:terminates|returns|works|ends|stops)|\bshould\b.{0,55}\bbut\b"
    r"|\b(?:spot|find) the (?:problem|bug|error|issue)\b|\bwhat(?:'s| is) wrong\b"
    r"|\bcorrected (?:version|implementation|code)\b"
    r"|\b(?:this|my|below|following|the (?:code|function|script|program|snippet|loop|method|query))\b"
    r".{0,60}\b(?:throws?|raises?|crashes?|fails?)\b",
    re.I,
)
_FIX = re.compile(_STRONG_FIX.pattern + r"|\bincorrect\b|\berror\b|\bexception\b", re.I)

_DIGIT = re.compile(r"\d")
_MATH_SIGNAL = re.compile(
    r"\b(?:calculate|compute|evaluate|convert|solve)\b|\bhow (?:many|much|far|fast|long)\b"
    r"|\bpercent(?:age)?\b|%|\baverage of\b|\bmean of\b|\bsum of\b|\btotal\b"
    r"|\bdiscount\b|\bprofit\b|\bsales tax\b|\binterest\b|\bratio\b|\bproportion\b"
    r"|\bper (?:hour|day|week|month|year|item|unit|person)\b|\bspeed\b|\bdistance\b"
    r"|\b(?:times|plus|minus|divided|multiply|product|quotient|remainder)\b|[+×÷]"
    r"|\b(?:area|perimeter|volume|square root|squared|cubed)\b"
    r"|\b(?:increases?|decreases?|grows?|drops?) by\b|\byears? (?:older|younger)\b|\bin \d+ years?\b",
    re.I,
)
_STRONG_LOGIC = re.compile(
    r"\b(?:logic puzzle|deductive|clues?|constraints?|liar|knight|knave|truth[- ]?teller)\b"
    r"|\bseating\b|\barrangement\b|\beach\b.{0,45}\bdifferent\b|\bwho owns\b",
    re.I,
)
_WEAK_LOGIC = re.compile(
    r"\b(?:directly |immediately )?(?:left|right) of\b|\badjacent\b"
    r"|\b(?:sits?|stands?|placed|arranged|lined? up)\s+in\s+(?:a|the)\s+(?:row|line|queue)\b"
    r"|\b(?:older|younger|taller|shorter|faster|slower) than\b"
    r"|\bfirst to last\b|\brank\b.{0,35}\b(?:first|last|order)\b"
    r"|\b(?:all|some|no)\s+\w+\s+are\b",
    re.I,
)


def classify(prompt: str) -> Category | None:
    p = (prompt or "").strip()
    if _SENTIMENT.search(p):
        return "sentiment"
    if _SUMMARY.search(p):
        return "summary"
    if _NER.search(p):
        return "ner"

    code_like = bool(_STRONG_CODE.search(p)) or bool(_WEAK_CODE.search(p) and (_WRITE.search(p) or _FIX.search(p)))
    if code_like:
        # A request to write new code stays codegen unless it explicitly describes
        # an existing broken artifact.
        if _WRITE.search(p) and not _STRONG_FIX.search(p):
            return "codegen"
        if _FIX.search(p):
            return "debug"
        return "debug" if _STRONG_CODE.search(p) else "codegen"

    has_math = bool(_DIGIT.search(p)) and bool(_MATH_SIGNAL.search(p))
    strong_logic = bool(_STRONG_LOGIC.search(p))
    has_logic = strong_logic or bool(_WEAK_LOGIC.search(p))

    if has_math and has_logic:
        # Comparative/order constraints are logic even when numbers appear.
        # Otherwise arithmetic intent is the safer interpretation.
        return "logic" if (strong_logic or _WEAK_LOGIC.search(p)) else "math"
    if has_math:
        return "math"
    if has_logic:
        return "logic"
    return "factual"


def fallback_messages(prompt: str) -> list[dict[str, str]]:
    return [{
        "role": "user",
        "content": (
            "Return one letter only: A factual, B math, C sentiment, D summary, "
            "E NER, F debugging, G logic, H code generation.\nTask: " + prompt[:500]
        ),
    }]


def parse_fallback_letter(text: str) -> str:
    match = re.search(r"\b([A-H])\b", (text or "").upper())
    return {
        "A": "factual", "B": "math", "C": "sentiment", "D": "summary",
        "E": "ner", "F": "debug", "G": "logic", "H": "codegen",
    }.get(match.group(1) if match else "", "factual")
