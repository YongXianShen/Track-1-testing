"""High-precision, zero-token solvers.

Only use a local result when the parser proves it understands the whole task.
Unknown or ambiguous tasks are deliberately escalated to Fireworks.
"""
from __future__ import annotations

import ast
import math
import re

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Constant, ast.Load,
)


def _format_number(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _evaluate(expression: str) -> float | None:
    expression = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    if not re.fullmatch(r"[\d\s+\-*/().%]+", expression):
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        if any(not isinstance(node, _ALLOWED_NODES) for node in ast.walk(tree)):
            return None
        value = eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, {})
        value = float(value)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def solve_math(prompt: str) -> str | None:
    compact = re.sub(r"\s+", " ", prompt.strip())
    low = compact.lower()
    # Do not locally answer tasks that explicitly require reasoning/explanation.
    if re.search(r"\b(?:explain|justify|prove|derive|show (?:your )?work)\b", low):
        return None

    # Pure arithmetic expression. Full-match prevents accidental partial parsing.
    match = re.fullmatch(
        r"(?:what is|calculate|compute|evaluate|solve)?\s*"
        r"([-+]?\d[\d\s+\-*/().%^×÷]*)\s*\??",
        low,
    )
    if match:
        value = _evaluate(match.group(1).strip())
        if value is not None:
            return _format_number(value)

    # Generic inventory pattern: start with N, sell/use P%, then X more.
    match = re.search(
        r"(?:has|starts? with|there are)\s+(\d+(?:\.\d+)?)\s+.*?"
        r"(?:sells?|sold|uses?|used|removes?|removed)\s+(\d+(?:\.\d+)?)\s*%"
        r".*?(?:then|and)\s+.*?(?:sells?|sold|uses?|used|removes?|removed)?\s*"
        r"(\d+(?:\.\d+)?)\s+(?:more|additional|extra)\b.*?"
        r"(?:remain|remaining|left)",
        low,
        flags=re.S,
    )
    if match:
        initial, percent, extra = map(float, match.groups())
        result = initial - initial * percent / 100.0 - extra
        return f"Answer: {_format_number(result)}"

    return None


def solve(category: str, prompt: str) -> str | None:
    if category == "math":
        return solve_math(prompt)
    return None
