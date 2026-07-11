"""High-confidence deterministic solvers for Track 1 V17.4.

Only narrow, fully parsed tasks are answered locally. Ambiguous prompts fall
through to Fireworks. The solvers are generic and do not depend on task IDs.
"""
from __future__ import annotations

import ast
import math
import re


def _fmt(x: float) -> str:
    if math.isfinite(x) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.8f}".rstrip("0").rstrip(".")


def _money(x: float) -> str:
    return f"${x:.2f}"


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


def _parse_fraction(value: str) -> float | None:
    value = value.strip().replace(" ", "")
    unicode_fractions = {
        "½": 1 / 2, "⅓": 1 / 3, "⅔": 2 / 3, "¼": 1 / 4,
        "¾": 3 / 4, "⅕": 1 / 5, "⅖": 2 / 5, "⅗": 3 / 5,
        "⅘": 4 / 5, "⅙": 1 / 6, "⅚": 5 / 6, "⅛": 1 / 8,
        "⅜": 3 / 8, "⅝": 5 / 8, "⅞": 7 / 8,
    }
    if value in unicode_fractions:
        return unicode_fractions[value]
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return float(value)
    m = re.fullmatch(r"(\d+)/(\d+)", value)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    return None


def _solve_stock_sequence(prompt: str) -> str | None:
    """Solve a fully described stock sequence in textual order.

    Supported operations are percentage sales/removals, fixed sales/removals,
    and fixed restocks/additions. It only activates for stock/unit questions and
    requires at least two recognized operations.
    """
    text = re.sub(r"\s+", " ", prompt.lower().replace(",", "")).strip()
    if not re.search(r"\b(?:remain|remaining|left|end of|final stock|stock remains?)\b", text):
        return None
    if not re.search(r"\b(?:stock|units?|items?|products?|books?|tickets?)\b", text):
        return None

    start_match = re.search(
        r"\b(?:starts?|begins?)\s+with\s+(\d+(?:\.\d+)?)"
        r"|\b(?:initially\s+has|has|there are)\s+(\d+(?:\.\d+)?)\s+(?:units?|items?|products?|books?|tickets?)\b",
        text,
    )
    if not start_match:
        return None
    start = float(next(group for group in start_match.groups() if group is not None))
    tail = text[start_match.end():]

    operation_rx = re.compile(
        r"(?P<out_pct>\b(?:sells?|sold|uses?|used|removes?|removed|loses?|lost|ships?|shipped)\s+"
        r"(?P<out_pct_n>\d+(?:\.\d+)?)\s*(?:%|\bpercent\b))"
        r"|(?P<in_fixed>\b(?:restocks?|restocked|adds?|added|receives?|received|gains?|gained|purchases?|purchased)\s+"
        r"(?P<in_fixed_n>\d+(?:\.\d+)?)\s+(?:units?|items?|products?|books?|tickets?)\b)"
        r"|(?P<out_fixed>\b(?:sells?|sold|uses?|used|removes?|removed|loses?|lost|ships?|shipped)\s+"
        r"(?P<out_fixed_n>\d+(?:\.\d+)?)\s+(?:units?|items?|products?|books?|tickets?)\b)",
        re.I,
    )

    operations: list[tuple[int, str, float]] = []
    shorthand_used = False
    for match in operation_rx.finditer(tail):
        if match.group("out_pct_n") is not None:
            operations.append((match.start(), "out_pct", float(match.group("out_pct_n"))))
        elif match.group("in_fixed_n") is not None:
            operations.append((match.start(), "in_fixed", float(match.group("in_fixed_n"))))
        elif match.group("out_fixed_n") is not None:
            operations.append((match.start(), "out_fixed", float(match.group("out_fixed_n"))))

    # Practice-style shorthand: "sells 15% and then 60 more". Do not use this
    # shortcut when any restock/addition wording is present.
    if len(operations) == 1 and operations[0][1] == "out_pct" and not re.search(
        r"\b(?:restock|add|receive|gain|purchase)", tail
    ):
        shorthand = re.search(
            r"(?:%|\bpercent\b).{0,45}\b(?:then|and)\s+(\d+(?:\.\d+)?)\s+"
            r"(?:more|additional|extra)(?:\s+(?:units?|items?|products?))?",
            tail[operations[0][0]:],
        )
        if shorthand:
            operations.append((len(tail), "out_fixed", float(shorthand.group(1))))
            shorthand_used = True

    if len(operations) < 2:
        return None

    current = start
    expressions = [_fmt(start)]
    for _, kind, value in sorted(operations):
        if kind == "out_pct":
            amount = current * value / 100
            current -= amount
            expressions.append(f"- {_fmt(amount)}")
        elif kind == "in_fixed":
            current += value
            expressions.append(f"+ {_fmt(value)}")
        else:
            current -= value
            expressions.append(f"- {_fmt(value)}")

    if current < -1e-9:
        return None
    if shorthand_used:
        return f"Answer: {_fmt(current)}"
    return f"{' '.join(expressions)} = {_fmt(current)}. Answer: {_fmt(current)} units."


def _solve_recipe_scaling(prompt: str) -> str | None:
    """Solve one-ingredient recipe scaling followed by a per-unit cost."""
    text = re.sub(r"\s+", " ", prompt.strip())
    amount_token = r"(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?|[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])"
    base = re.search(
        rf"(?P<amount>{amount_token})\s+cups?\s+(?:of\s+)?(?P<ingredient>[A-Za-z][A-Za-z -]{{0,35}}?)\s+for\s+"
        r"(?P<count>\d+(?:\.\d+)?)\s+(?P<unit>cookies?|servings?|portions?|people|muffins?|cakes?|pancakes?|biscuits?)\b",
        text,
        re.I,
    )
    if not base:
        return None
    amount = _parse_fraction(base.group("amount"))
    if amount is None:
        return None
    base_count = float(base.group("count"))
    unit = base.group("unit")

    target_matches = list(re.finditer(rf"(?:for|make|makes?|serve|serves?)\s+(\d+(?:\.\d+)?)\s+{re.escape(unit)}\b", text[base.end():], re.I))
    if not target_matches:
        # Handles "needed for 30 cookies" where other words appear before "for".
        target_matches = list(re.finditer(rf"for\s+(\d+(?:\.\d+)?)\s+{re.escape(unit)}\b", text[base.end():], re.I))
    cost_match = re.search(r"(?:costs?|priced? at)\s+\$?\s*(\d+(?:\.\d+)?)\s+per\s+cup\b", text, re.I)
    if not target_matches or not cost_match or base_count <= 0:
        return None

    target_count = float(target_matches[-1].group(1))
    cost_per_cup = float(cost_match.group(1))
    cups = amount * target_count / base_count
    total = cups * cost_per_cup
    return (
        f"{_fmt(amount)} × {_fmt(target_count)}/{_fmt(base_count)} = {_fmt(cups)} cups; "
        f"{_fmt(cups)} × {_money(cost_per_cup)} = {_money(total)}. "
        f"Answer: {_fmt(cups)} cups and {_money(total)}."
    )


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

    stock = _solve_stock_sequence(prompt)
    if stock:
        return stock

    recipe = _solve_recipe_scaling(prompt)
    if recipe:
        return recipe

    m = re.search(r"(?:average|mean) of ([\d,\s.\-]+)\??$", text)
    if m:
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
        if 2 <= len(nums) <= 12:
            return f"Answer: {_fmt(sum(nums) / len(nums))}"
    return None


def _review_original(prompt: str) -> str:
    quoted = re.findall(r"['\"]([^'\"]{8,})['\"]", prompt)
    if quoted:
        return quoted[-1].strip()
    parts = re.split(r"(?:review|feedback|comment|tweet|text|message)\s*[:\-]", prompt, flags=re.I)
    return parts[-1].strip().strip("'\"")


_POSITIVE_PATTERNS = (
    r"\bamazing\b", r"\bawesome\b", r"\bexcellent\b", r"\bfantastic\b",
    r"\bgood\b", r"\bgreat\b", r"\bhappy\b", r"\bimpressive\b",
    r"\blove(?:d)?\b", r"\bperfect(?:ly)?\b", r"\bflawless\b",
    r"\breliable\b", r"\bsmooth\b", r"\bwonderful\b", r"\bbest\b",
    r"\buseful\b", r"\beasy\b", r"\bworked?\b", r"\bresolved\b",
    r"\bresponsive\b", r"\bquick(?:ly)?\b", r"\bfast\b",
)
_NEGATIVE_PATTERNS = (
    r"\bawful\b", r"\bbad\b", r"\bbroken\b", r"\bconfusing\b",
    r"\bcrash(?:es|ed)?\b", r"\bdisappointed\b", r"\bhate(?:d)?\b",
    r"\bpoor\b", r"\bscratch(?:es|ed)?\b", r"\bslow\b", r"\bterrible\b",
    r"\bworst\b", r"\bbuggy\b", r"\blate\b", r"\bdifficult\b",
    r"\bdamaged\b", r"\bdented\b", r"\bmissing\b", r"\bcomplaint\b",
    r"\bfailed\b", r"\bproblem\b",
)


def _sentiment_scores(text: str) -> tuple[int, int]:
    low = text.lower()
    pos = len(re.findall(r"\bnot\s+(?:bad|awful|terrible|poor|difficult)\b", low))
    neg = len(re.findall(r"\bnot\s+(?:good|great|amazing|excellent|useful|easy)\b", low))
    low = re.sub(r"\bnot\s+(?:bad|awful|terrible|poor|difficult|good|great|amazing|excellent|useful|easy)\b", " ", low)
    pos += sum(bool(re.search(pattern, low)) for pattern in _POSITIVE_PATTERNS)
    neg += sum(bool(re.search(pattern, low)) for pattern in _NEGATIVE_PATTERNS)
    return int(pos), int(neg)


def _short_clause(text: str, limit: int = 64) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .,:;!?\"'")
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def solve_sentiment(prompt: str) -> str | None:
    if not re.search(r"sentiment|review|feedback|tweet|positive|negative|neutral|mixed|tone|mood", prompt, re.I):
        return None
    original = _review_original(prompt)
    if not original:
        return None

    split = re.search(r"\b(?:but|however|although|though|yet|while)\b", original, re.I)
    if split:
        left = original[:split.start()].strip(" ,;:")
        right = original[split.end():].strip(" ,;:")
        left_pos, left_neg = _sentiment_scores(left)
        right_pos, right_neg = _sentiment_scores(right)
        if left_neg > left_pos and right_pos > right_neg:
            return f"Mixed — negative: {_short_clause(left)}; positive: {_short_clause(right)}."
        if left_pos > left_neg and right_neg > right_pos:
            return f"Mixed — positive: {_short_clause(left)}; negative: {_short_clause(right)}."

    pos_score, neg_score = _sentiment_scores(original)
    if pos_score >= 2 and neg_score == 0:
        return f"Positive — {_short_clause(original, 120)}."
    if neg_score >= 2 and pos_score == 0:
        return f"Negative — {_short_clause(original, 120)}."
    if re.search(r"\b(?:okay|fine|average|ordinary|nothing special|acceptable|neutral)\b", original, re.I) and not (pos_score or neg_score):
        return f"Neutral — {_short_clause(original, 120)}."
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
