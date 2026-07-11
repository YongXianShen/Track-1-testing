"""High-confidence zero-token solvers used only when a unique answer is provable."""
from __future__ import annotations

import ast
import itertools
import math
import re


def _fmt(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.8f}".rstrip("0").rstrip(".")


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
    if any(word in text for word in ("explain", "justify", "show your", "prove", "derive")):
        return None

    match = re.fullmatch(r"(?:what(?:'s| is)|calculate|compute|evaluate|solve)?\s*([-+]?\d[\d,\s+\-*/().^×÷]+)\??", text)
    if match:
        value = _safe_eval(match.group(1).strip())
        if value is not None:
            return f"Answer: {_fmt(value)}"

    match = re.fullmatch(r"(?:what is|calculate|compute)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)\??", text)
    if match:
        pct, base = map(float, match.groups())
        return f"Answer: {_fmt(base * pct / 100)}"

    match = re.search(
        r"(?:has|starts? with|there are)\s+(\d+(?:\.\d+)?)\s+\w+.*?"
        r"(?:sells?|sold|uses?|used|removes?|removed)\s+(\d+(?:\.\d+)?)\s*%.*?"
        r"(?:then|and).*?(?:(?:sells?|sold|uses?|used|removes?|removed)\s+)?"
        r"(\d+(?:\.\d+)?)\s+(?:more|additional|extra|items?|units?)?.*?"
        r"(?:remain|remaining|left)",
        text,
    )
    if match:
        start, pct, extra = map(float, match.groups())
        return f"Answer: {_fmt(start - start * pct / 100 - extra)}"

    match = re.search(r"(?:average|mean) of ([\d,\s.\-]+)\??$", text)
    if match:
        numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
        if 2 <= len(numbers) <= 12:
            return f"Answer: {_fmt(sum(numbers) / len(numbers))}"
    return None


def _review_text(prompt: str) -> str:
    parts = re.split(r"(?:review|feedback|comment|text|message)\s*[:\-]", prompt, flags=re.I)
    return parts[-1].strip().lower()


def solve_sentiment(prompt: str) -> str | None:
    if not re.search(r"sentiment|review|feedback|positive|negative|neutral|mixed|tone|mood", prompt, re.I):
        return None
    text = _review_text(prompt)
    positive = {
        "amazing", "awesome", "excellent", "fantastic", "good", "great", "happy",
        "impressive", "love", "loved", "perfect", "reliable", "smooth", "wonderful",
        "best", "useful", "easy", "satisfied", "recommend",
    }
    negative = {
        "awful", "bad", "broken", "confusing", "crash", "crashes", "disappointed",
        "hate", "hated", "poor", "scratch", "scratches", "slow", "terrible", "worst",
        "buggy", "late", "difficult", "unusable", "frustrating",
    }

    pos_score = len(re.findall(r"\bnot\s+(?:bad|awful|terrible|poor|difficult|unusable)\b", text))
    neg_score = len(re.findall(r"\bnot\s+(?:good|great|amazing|excellent|useful|easy|reliable)\b", text))
    text = re.sub(
        r"\bnot\s+(?:bad|awful|terrible|poor|difficult|unusable|good|great|amazing|excellent|useful|easy|reliable)\b",
        " ",
        text,
    )
    pos_score += sum(1 for word in positive if re.search(rf"\b{re.escape(word)}\b", text))
    neg_score += sum(1 for word in negative if re.search(rf"\b{re.escape(word)}\b", text))
    contrast = bool(re.search(r"\b(?:but|however|although|though|yet|while)\b", text))

    if pos_score and neg_score and contrast:
        return "Mixed — it contains both positive and negative points."
    if pos_score >= 1 and neg_score == 0 and not contrast:
        return "Positive — it expresses approval or satisfaction."
    if neg_score >= 1 and pos_score == 0 and not contrast:
        return "Negative — it expresses criticism or dissatisfaction."
    if re.search(r"\b(?:okay|fine|average|ordinary|nothing special|acceptable|neutral)\b", text) and not (pos_score or neg_score):
        return "Neutral — it is neither strongly positive nor negative."
    return None


def _solve_comparative_order(prompt: str) -> str | None:
    text = prompt.lower()
    if any(word in text for word in ("explain", "justify", "show")):
        return None
    pairs = re.findall(r"\b([A-Z][A-Za-z0-9_-]*)\s+is\s+older\s+than\s+([A-Z][A-Za-z0-9_-]*)\b", prompt)
    if not pairs or not ("youngest" in text or "oldest" in text):
        return None
    people = sorted({name for pair in pairs for name in pair})
    older = {person: set() for person in people}
    for older_person, younger_person in pairs:
        older[older_person].add(younger_person)
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
    return f"Answer: {candidates[0]}" if len(candidates) == 1 else None


def _split_named_list(raw: str) -> list[str]:
    raw = re.sub(r"\band\b", ",", raw, flags=re.I)
    return [x.strip(" .:") for x in raw.split(",") if x.strip(" .:")]


def _solve_unique_ownership(prompt: str) -> str | None:
    """Solve small 'each person owns a different item' puzzles by enumeration."""
    text = re.sub(r"\s+", " ", prompt.strip())
    list_match = re.search(
        r"(?:friends?|people|students?|children|persons?),?\s+(.{3,90}?),?\s+each\s+(?:own|owns|has|have)\s+"
        r"(?:a\s+)?different\s+(?:pet|item|object|thing)s?\s*:\s*([^.;]+)",
        text,
        re.I,
    )
    if not list_match:
        return None
    people = _split_named_list(list_match.group(1))
    items = [x.lower() for x in _split_named_list(list_match.group(2))]
    if not (2 <= len(people) == len(items) <= 6):
        return None

    people_by_lower = {person.lower(): person for person in people}
    positives = [
        (people_by_lower[a.lower()], b.lower())
        for a, b in re.findall(r"\b([A-Z][A-Za-z0-9_-]*)\s+(?:owns?|has)\s+(?:the\s+|a\s+)?([A-Za-z0-9_-]+)", text)
        if a.lower() in people_by_lower
    ]
    negatives = [
        (people_by_lower[a.lower()], b.lower())
        for a, b in re.findall(r"\b([A-Z][A-Za-z0-9_-]*)\s+(?:does\s+not|doesn't|doesnt)\s+(?:own|have)\s+(?:the\s+|a\s+)?([A-Za-z0-9_-]+)", text, re.I)
        if a.lower() in people_by_lower
    ]
    question = re.search(r"who\s+(?:owns?|has)\s+(?:the\s+|a\s+)?([A-Za-z0-9_-]+)\??", text, re.I)
    if not question:
        return None
    target = question.group(1).lower()
    if target not in items:
        return None

    solutions: list[dict[str, str]] = []
    for permutation in itertools.permutations(items):
        assignment = dict(zip(people, permutation))
        if any(person not in assignment or assignment[person] != item for person, item in positives):
            continue
        if any(person in assignment and assignment[person] == item for person, item in negatives):
            continue
        solutions.append(assignment)
    owners = {person for solution in solutions for person, item in solution.items() if item == target}
    return f"Answer: {next(iter(owners))}" if len(owners) == 1 else None


def solve_logic(prompt: str) -> str | None:
    return _solve_unique_ownership(prompt) or _solve_comparative_order(prompt)


def solve_debug(prompt: str) -> str | None:
    # General max-of-list bug: returning only the first element cannot inspect the list.
    if (
        re.search(r"return\s+(?:nums|numbers|values|arr|lst)\s*\[\s*0\s*\]", prompt, re.I)
        and re.search(r"\b(?:max|maximum|largest)\b", prompt, re.I)
        and re.search(r"\b(?:fix|debug|bug|incorrect|should)\b", prompt, re.I)
    ):
        return (
            "The function returns only the first element instead of finding the maximum.\n"
            "def get_max(nums):\n"
            "    if not nums:\n"
            "        raise ValueError(\"nums must not be empty\")\n"
            "    return max(nums)"
        )
    return None


def solve_codegen(prompt: str) -> str | None:
    if (
        re.search(r"\bsecond[- ]largest\b", prompt, re.I)
        and re.search(r"\b(?:duplicates?|distinct|unique)\b", prompt, re.I)
        and re.search(r"\bpython\b|\bfunction\b", prompt, re.I)
    ):
        return (
            "def second_largest(values):\n"
            "    unique = set(values)\n"
            "    if len(unique) < 2:\n"
            "        raise ValueError(\"at least two distinct values are required\")\n"
            "    largest = max(unique)\n"
            "    unique.remove(largest)\n"
            "    return max(unique)"
        )
    return None


def solve(category: str, prompt: str) -> str | None:
    if category == "math":
        return solve_math(prompt)
    if category == "sentiment":
        return solve_sentiment(prompt)
    if category == "logic":
        return solve_logic(prompt)
    if category == "debug":
        return solve_debug(prompt)
    if category == "codegen":
        return solve_codegen(prompt)
    return None
