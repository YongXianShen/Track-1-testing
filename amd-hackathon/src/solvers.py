"""Very narrow local solvers for zero-token wins. Avoid broad guesses."""
from __future__ import annotations

import ast
import math
import re


def _fmt(x: float) -> str:
    if math.isfinite(x) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.8f}".rstrip("0").rstrip(".")


def _safe_eval(expr: str) -> float | None:
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    if not re.fullmatch(r"[\d\s+\-*/().%*]+", expr):
        return None
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Constant, ast.Load)
    try:
        tree = ast.parse(expr, mode="eval")
        if any(not isinstance(n, allowed) for n in ast.walk(tree)):
            return None
        return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {}))
    except Exception:
        return None


def solve_math(prompt: str) -> str | None:
    text = re.sub(r"\s+", " ", prompt.strip().lower())
    if any(w in text for w in ("explain", "justify", "show", "prove", "derive")):
        return None
    m = re.fullmatch(r"(?:what is|calculate|compute|evaluate)?\s*([-+]?\d[\d\s+\-*/().%^×÷]+)\??", text)
    if m:
        val = _safe_eval(m.group(1).strip())
        if val is not None:
            return _fmt(val)
    # Starts with N, removes/sells p%, then M more; ask remain.
    m = re.search(
        r"(?:has|starts? with|there are)\s+(\d+(?:\.\d+)?)\s+\w+.*?"
        r"(?:sells?|sold|uses?|used|removes?|removed)\s+(\d+(?:\.\d+)?)\s*%.*?"
        r"(?:then|and).*?(?:(?:sells?|sold|uses?|used|removes?|removed)\s+)?(\d+(?:\.\d+)?)\s+(?:more|additional|extra|items?|units?)?.*?"
        r"(?:remain|remaining|left)", text)
    if m:
        start, pct, extra = map(float, m.groups())
        return f"Answer: {_fmt(start - start * pct / 100 - extra)}"
    m = re.search(r"(?:average|mean) of ([\d,\s.\-]+)\??$", text)
    if m:
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
        if 2 <= len(nums) <= 10:
            return f"Answer: {_fmt(sum(nums) / len(nums))}"
    return None


def solve_sentiment(prompt: str) -> str | None:
    if not re.search(r"sentiment|review|feedback|positive|negative|neutral|mixed", prompt, re.I):
        return None
    text = re.split(r"review\s*[:\-]|feedback\s*[:\-]|text\s*[:\-]", prompt, flags=re.I)[-1].lower()
    pos_words = {"amazing","awesome","excellent","fantastic","fast","good","great","happy","impressive","love","loved","perfect","reliable","smooth","wonderful","best","useful","easy"}
    neg_words = {"awful","bad","broken","confusing","crash","crashes","disappointed","hate","hated","poor","scratch","scratches","slow","terrible","worst","buggy","late","difficult"}
    pos = sum(1 for w in pos_words if re.search(rf"\b{re.escape(w)}\b", text))
    neg = sum(1 for w in neg_words if re.search(rf"\b{re.escape(w)}\b", text))
    if pos and neg:
        return "Mixed — it contains both positive and negative points."
    if pos and not neg:
        return "Positive — it expresses approval or satisfaction."
    if neg and not pos:
        return "Negative — it expresses criticism or dissatisfaction."
    if re.search(r"\b(okay|fine|average|ordinary|nothing special|acceptable)\b", text):
        return "Neutral — it is neither strongly positive nor negative."
    return None


def solve_logic(prompt: str) -> str | None:
    text = prompt.lower()
    if any(w in text for w in ("explain", "justify", "show")):
        return None
    pairs = re.findall(r"\b([A-Z][A-Za-z0-9_-]*)\s+is\s+older\s+than\s+([A-Z][A-Za-z0-9_-]*)\b", prompt)
    if pairs and ("youngest" in text or "oldest" in text):
        people = sorted({x for p in pairs for x in p})
        older = {p: set() for p in people}
        for a, b in pairs:
            older[a].add(b)
        changed = True
        while changed:
            changed = False
            for p in people:
                new = set(older[p])
                for q in list(older[p]):
                    new |= older.get(q, set())
                if new != older[p]:
                    older[p] = new
                    changed = True
        cands = [p for p in people if (all(p in older[o] for o in people if o != p) if "youngest" in text else len(older[p]) == len(people) - 1)]
        if len(cands) == 1:
            return f"Answer: {cands[0]}"
    return None


def solve(category: str, prompt: str) -> str | None:
    if category == "math":
        return solve_math(prompt)
    if category == "sentiment":
        return solve_sentiment(prompt)
    if category == "logic":
        return solve_logic(prompt)
    return None
