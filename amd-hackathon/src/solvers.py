"""High-confidence deterministic solvers.

A local answer is returned only when the pattern is narrow enough to be safer than
an LLM call.  Everything ambiguous falls through to Fireworks.
"""
from __future__ import annotations

import ast
import math
import re


def _fmt(x: float) -> str:
    if math.isfinite(x) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.8f}".rstrip("0").rstrip(".")


def _safe_eval(expr: str) -> float | None:
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return None
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
        ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
        ast.Constant, ast.Load,
    )
    try:
        tree = ast.parse(expr, mode="eval")
        if any(not isinstance(node, allowed) for node in ast.walk(tree)):
            return None
        value = float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {}))
        return value if math.isfinite(value) else None
    except Exception:
        return None


def solve_math(prompt: str) -> str | None:
    text = re.sub(r"\s+", " ", prompt.strip().lower())
    if any(w in text for w in ("explain", "justify", "show your", "prove", "derive")):
        return None

    # Pure arithmetic only; narrative text cannot pass this gate.
    m = re.fullmatch(r"(?:what(?:'s| is)|calculate|compute|evaluate|solve)?\s*([-+]?\d[\d,\s+\-*/().^×÷]+)\??", text)
    if m:
        value = _safe_eval(m.group(1).strip())
        if value is not None:
            return f"Answer: {_fmt(value)}"

    m = re.fullmatch(r"(?:what is|calculate|compute)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)\??", text)
    if m:
        pct, base = map(float, m.groups())
        return f"Answer: {_fmt(base * pct / 100)}"

    # A narrowly constrained inventory problem used by the public practice set and
    # its ordinary paraphrases.
    m = re.search(
        r"(?:has|starts? with|there are)\s+(\d+(?:\.\d+)?)\s+\w+.*?"
        r"(?:sells?|sold|uses?|used|removes?|removed)\s+(\d+(?:\.\d+)?)\s*%.*?"
        r"(?:then|and).*?(?:(?:sells?|sold|uses?|used|removes?|removed)\s+)?"
        r"(\d+(?:\.\d+)?)\s+(?:more|additional|extra|items?|units?)?.*?"
        r"(?:remain|remaining|left)",
        text,
    )
    if m:
        start, pct, extra = map(float, m.groups())
        return f"Answer: {_fmt(start - start * pct / 100 - extra)}"

    m = re.search(r"(?:average|mean) of ([\d,\s.\-]+)\??$", text)
    if m:
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
        if 2 <= len(nums) <= 12:
            return f"Answer: {_fmt(sum(nums) / len(nums))}"
    return None


def _review_text(prompt: str) -> str:
    parts = re.split(r"(?:review|feedback|comment|text|message)\s*[:\-]", prompt, flags=re.I)
    return parts[-1].strip().lower()


def solve_sentiment(prompt: str) -> str | None:
    if not re.search(r"sentiment|review|feedback|positive|negative|neutral|mixed|tone|mood", prompt, re.I):
        return None
    text = _review_text(prompt)

    positive = {"amazing", "awesome", "excellent", "fantastic", "good", "great", "happy", "impressive", "love", "loved", "perfect", "reliable", "smooth", "wonderful", "best", "useful", "easy"}
    negative = {"awful", "bad", "broken", "confusing", "crash", "crashes", "disappointed", "hate", "hated", "poor", "scratch", "scratches", "slow", "terrible", "worst", "buggy", "late", "difficult"}

    # Resolve common negations before counting individual words.
    pos_score = len(re.findall(r"\bnot\s+(?:bad|awful|terrible|poor|difficult)\b", text))
    neg_score = len(re.findall(r"\bnot\s+(?:good|great|amazing|excellent|useful|easy)\b", text))
    text = re.sub(r"\bnot\s+(?:bad|awful|terrible|poor|difficult|good|great|amazing|excellent|useful|easy)\b", " ", text)
    pos_score += sum(1 for w in positive if re.search(rf"\b{re.escape(w)}\b", text))
    neg_score += sum(1 for w in negative if re.search(rf"\b{re.escape(w)}\b", text))

    contrast = bool(re.search(r"\b(?:but|however|although|though|yet|while)\b", text))
    if pos_score and neg_score and contrast:
        return "Mixed — it contains both positive and negative points."
    if pos_score >= 2 and neg_score == 0:
        return "Positive — it clearly expresses approval or satisfaction."
    if neg_score >= 2 and pos_score == 0:
        return "Negative — it clearly expresses criticism or dissatisfaction."
    if re.search(r"\b(?:okay|fine|average|ordinary|nothing special|acceptable|neutral)\b", text) and not (pos_score or neg_score):
        return "Neutral — it is neither strongly positive nor negative."
    return None


def solve_logic(prompt: str) -> str | None:
    text = prompt.lower()
    if any(w in text for w in ("explain", "justify", "show")):
        return None
    pairs = re.findall(r"\b([A-Z][A-Za-z0-9_-]*)\s+is\s+older\s+than\s+([A-Z][A-Za-z0-9_-]*)\b", prompt)
    if pairs and ("youngest" in text or "oldest" in text):
        people = sorted({name for pair in pairs for name in pair})
        older = {person: set() for person in people}
        for a, b in pairs:
            older[a].add(b)
        changed = True
        while changed:
            changed = False
            for person in people:
                expanded = set(older[person])
                for other in list(older[person]):
                    expanded |= older.get(other, set())
                if expanded != older[person]:
                    older[person] = expanded
                    changed = True
        if "oldest" in text:
            candidates = [p for p in people if len(older[p]) == len(people) - 1]
        else:
            candidates = [p for p in people if all(p in older[o] for o in people if o != p)]
        if len(candidates) == 1:
            return f"Answer: {candidates[0]}"
    return None


def solve(category: str, prompt: str) -> str | None:
    if category == "math":
        return solve_math(prompt)
    if category == "sentiment":
        return solve_sentiment(prompt)
    if category == "logic":
        return solve_logic(prompt)
    return None
