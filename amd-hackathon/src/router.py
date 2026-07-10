"""Deterministic routing for the eight Track 1 task categories.

The router is intentionally conservative: explicit task wording wins, code is
recognized before numeric reasoning, and everything ambiguous falls back to a
strong general model rather than a fragile local guess.
"""
from __future__ import annotations

import re

Category = str

_SENTIMENT = re.compile(
    r"\bsentiment\b|\bpolarity\b|classif(?:y|ication).{0,80}"
    r"\b(?:positive|negative|neutral|mixed)\b|"
    r"\b(?:review|feedback)\b.{0,80}\b(?:sentiment|tone)\b",
    re.I | re.S,
)
_SUMMARY = re.compile(
    r"summari[sz]e|\bsummary\b|\btl;?dr\b|\bcondense\b|\bshorten\b|"
    r"\brecap\b|\bgist\b|exactly\s+\d+\s+sentences?|\bone sentence\b|"
    r"\bsingle sentence\b|\bin\s+\d+\s+words?\b|\bin\s+\d+\s+bullets?\b",
    re.I,
)
_NER = re.compile(
    r"named entit|\bNER\b|entities?\s+and\s+(?:their\s+)?types?|"
    r"(?:extract|identify|list).{0,100}\b(?:entities?|persons?|people|"
    r"organi[sz]ations?|companies|locations?|cities|countries|dates?)\b",
    re.I | re.S,
)

_CODE_BLOCK = re.compile(
    r"```|\bdef\s+[A-Za-z_]\w*\s*\(|#include\s*[<\"]|"
    r"\b(?:public|private|protected)\s+(?:static\s+)?(?:class|void|int|String)\b|"
    r"\bclass\s+[A-Za-z_]\w*\s*[:{]|\bSELECT\b.{0,120}\bFROM\b|"
    r"console\.log\s*\(|System\.out\.|=>\s*[{(]?",
    re.I | re.S,
)
_CODE_TOPIC = re.compile(
    r"\bcode\b|\bsnippet\b|\bfunction\b|\bmethod\b|\bprogram\b|"
    r"\bscript\b|\balgorithm\b|\bregex\b|\bSQL\b|\bPython\b|"
    r"\bJavaScript\b|\bTypeScript\b|\bJava\b|\bC\+\+\b|\bRust\b",
    re.I,
)
_DEBUG = re.compile(
    r"\bdebug\b|\bbug(?:gy)?\b|\bbroken\b|\bfix\b|\bincorrect\b|"
    r"\bwrong\b|not working|\bfails?\b|\berror\b|\bexception\b|"
    r"find\s+and\s+fix|should.{0,70}\bbut\b|what(?:'s| is) wrong",
    re.I | re.S,
)
_GENERATE = re.compile(
    r"\bwrite\b|\bimplement\b|\bcreate\b|\bbuild\b|\bgenerate\b|"
    r"\bdefine\b|\bprovide\b|\bcomplete\b",
    re.I,
)

_LOGIC = re.compile(
    r"\blogic(?:al)?\b|\bdeductive\b|\bconstraint\b|\bpuzzle\b|\briddle\b|"
    r"\bclues?\b|\btruth(?:ful)?\b|\bliars?\b|\bknights?\b|\bknaves?\b|"
    r"\bseating\b|\barrangement\b|each.{0,50}\bdifferent\b|\bwho owns\b|"
    r"\bowns the\b|\bleft of\b|\bright of\b|\bolder than\b|\byounger than\b|"
    r"\bmust be true\b|\bcan(?:not|'t) be true\b|satisf(?:y|ies).{0,50}conditions?",
    re.I | re.S,
)
_DIGIT = re.compile(r"\d")
_MATH = re.compile(
    r"\bcalculate\b|\bcompute\b|\bevaluate\b|\bsolve\b|\bhow (?:many|much)\b|"
    r"\bpercent(?:age)?\b|%|\baverage\b|\bmean\b|\bsum\b|\btotal\b|"
    r"\bremain(?:ing)?\b|\bleft\b|\bsold\b|\bdiscount\b|\bprofit\b|"
    r"\bratio\b|\binterest\b|\bcost\b|\bprice\b|\barea\b|\bperimeter\b|"
    r"\bvolume\b|\bspeed\b|\bdistance\b|\brate\b|\bincrease\b|\bdecrease\b|"
    r"[+×÷=]",
    re.I,
)


def classify(prompt: str) -> Category:
    text = prompt.strip()

    if _SENTIMENT.search(text):
        return "sentiment"
    if _SUMMARY.search(text):
        return "summary"
    if _NER.search(text):
        return "ner"

    code_context = bool(_CODE_BLOCK.search(text)) or bool(_CODE_TOPIC.search(text))
    if code_context and _DEBUG.search(text):
        return "debug"
    if code_context and _GENERATE.search(text):
        return "codegen"
    if _CODE_BLOCK.search(text):
        return "debug"

    # Logic clues take precedence over incidental numbers in a puzzle.
    if _LOGIC.search(text):
        return "logic"
    if _DIGIT.search(text) and _MATH.search(text):
        return "math"

    return "factual"
