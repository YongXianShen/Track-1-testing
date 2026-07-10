"""Conservative deterministic solvers. Return None unless the interpretation is unambiguous."""
from __future__ import annotations

import ast
import math
import re


def _fmt(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _eval(expr: str) -> float | None:
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    if not re.fullmatch(r"[\d\s+\-*/().%]+", expr):
        return None
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Constant, ast.Load)
    try:
        tree = ast.parse(expr, mode="eval")
        if any(not isinstance(node, allowed) for node in ast.walk(tree)):
            return None
        return float(eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, {}))
    except Exception:
        return None


def solve_math(prompt: str) -> str | None:
    text = re.sub(r"\s+", " ", prompt.strip().lower())
    if re.search(r"\b(explain|justify|prove|derive|show your work)\b", text):
        return None

    m = re.fullmatch(r"(?:what is|calculate|compute|evaluate)?\s*([-+]?\d[\d\s+\-*/().^×÷%]+)\??", text)
    if m:
        value = _eval(m.group(1).strip())
        if value is not None:
            return f"Answer: {_fmt(value)}"

    m = re.search(r"(?:what is|calculate)?\s*(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)", text)
    if m:
        pct, base = map(float, m.groups())
        return f"Answer: {_fmt(base * pct / 100)}"

    m = re.search(r"(?:average|mean) of\s+([\d,\s.\-]+)\??$", text)
    if m:
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
        if 2 <= len(nums) <= 20:
            return f"Answer: {_fmt(sum(nums) / len(nums))}"

    m = re.search(
        r"(?:has|starts? with|there are)\s+(\d+(?:\.\d+)?)\s+\w+.*?"
        r"(?:sells?|sold|uses?|used|removes?|removed)\s+(\d+(?:\.\d+)?)\s*%.*?"
        r"(?:then|and).*?(?:(?:sells?|sold|uses?|used|removes?|removed)\s+)?(\d+(?:\.\d+)?)\s+(?:more|additional|extra|items?|units?)?.*?"
        r"(?:remain|remaining|left)",
        text,
    )
    if m:
        start, pct, extra = map(float, m.groups())
        return f"Answer: {_fmt(start - start * pct / 100 - extra)}"

    m = re.search(r"(?:price|cost)\s+(?:is\s+)?\$?(\d+(?:\.\d+)?).*?(\d+(?:\.\d+)?)\s*%\s+(?:discount|off).*?(?:final|new|sale)\s+(?:price|cost)", text)
    if m:
        price, pct = map(float, m.groups())
        return f"Answer: {_fmt(price * (1 - pct / 100))}"
    return None


def solve_sentiment(prompt: str) -> str | None:
    if not re.search(r"sentiment|review|feedback|positive|negative|neutral|mixed|polarity", prompt, re.I):
        return None
    if re.search(r"sarcasm|sarcastic|irony|ironic|custom label|labels? are|choose from", prompt, re.I):
        return None
    text = re.split(r"review\s*[:\-]|feedback\s*[:\-]|text\s*[:\-]", prompt, flags=re.I)[-1].lower()
    positive = {"amazing", "awesome", "excellent", "fantastic", "good", "great", "happy", "impressive", "love", "loved", "perfect", "reliable", "smooth", "wonderful", "best", "useful", "easy", "pleased", "satisfied"}
    negative = {"awful", "bad", "broken", "confusing", "crash", "crashes", "disappointed", "hate", "hated", "poor", "scratch", "scratches", "slow", "terrible", "worst", "buggy", "late", "difficult", "frustrating", "annoying", "unreliable"}
    words = re.findall(r"[a-z]+(?:n't)?", text)
    pos = neg = 0
    for i, word in enumerate(words):
        polarity = 1 if word in positive else -1 if word in negative else 0
        if not polarity:
            continue
        window = words[max(0, i - 3):i]
        if any(w in {"not", "never", "no", "hardly"} or w.endswith("n't") for w in window):
            polarity *= -1
        pos += polarity > 0
        neg += polarity < 0
    if pos and neg:
        return "Mixed — it contains both positive and negative points."
    if pos >= 2 and not neg:
        return "Positive — it expresses clear approval or satisfaction."
    if neg >= 2 and not pos:
        return "Negative — it expresses clear criticism or dissatisfaction."
    return None


def solve(category: str, prompt: str) -> str | None:
    if category == "math":
        return solve_math(prompt)
    if category == "sentiment":
        return solve_sentiment(prompt)
    return None
