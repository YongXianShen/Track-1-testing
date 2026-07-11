"""No-paid-deploy model selection for Track 1 V17.3.

Every remote model is read from ALLOWED_MODELS. Gemma entries are deliberately
excluded so this build never requires an on-demand paid deployment. With the
published list, MiniMax handles non-code tasks and Kimi handles code tasks.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

BAD_RX = re.compile(r"embed|embedding|rerank|image|vision|audio|whisper|tts|guard|moderation|diffusion|flux|sdxl", re.I)
GEMMA_RX = re.compile(r"gemma", re.I)
CODE_RX = re.compile(r"code|coder|kimi", re.I)
REASON_RX = re.compile(r"\br1\b|qwq|reason|thinking|think|deepthink", re.I)
INSTRUCT_RX = re.compile(r"instruct|chat|it|assistant|turbo", re.I)
SIZE_RX = re.compile(r"(\d+(?:\.\d+|p\d+)?)\s*b\b", re.I)
MOE_RX = re.compile(r"(\d+)\s*x\s*(\d+)b\b", re.I)

UNKNOWN_SIZE = 12.0

@dataclass(frozen=True)
class ModelPlan:
    SMALL: str
    LANGUAGE: str
    REASON: str
    CODE: str
    FALLBACK: str

    def as_dict(self) -> dict[str, str]:
        return {
            "SMALL": self.SMALL,
            "LANGUAGE": self.LANGUAGE,
            "REASON": self.REASON,
            "CODE": self.CODE,
            "FALLBACK": self.FALLBACK,
        }


def allowed_models() -> list[str]:
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise ValueError("ALLOWED_MODELS is empty")
    usable: list[str] = []
    for m in models:
        if BAD_RX.search(m):
            continue
        if GEMMA_RX.search(m):
            continue
        usable.append(m)
    return usable or [m for m in models if not BAD_RX.search(m)] or models


def size_b(model_id: str) -> float:
    s = model_id.lower().replace("_", "-")
    moe = MOE_RX.findall(s)
    if moe:
        return max(float(a) * float(b) for a, b in moe)
    found = []
    for x in SIZE_RX.findall(s):
        try:
            found.append(float(x.replace("p", ".")))
        except Exception:
            pass
    return max(found) if found else UNKNOWN_SIZE


def _family_score(m: str, category: str) -> int:
    t = m.lower()
    score = 0

    # General safety and quality signals.
    score += min(int(size_b(m) * 6), 900)
    if INSTRUCT_RX.search(t):
        score += 70
    if REASON_RX.search(t) and category not in {"math", "logic"}:
        score -= 120

    # Category-specific family ranking. These are soft preferences; the allowed
    # list is still the source of truth.
    if category in {"debug", "codegen"}:
        if "kimi" in t: score += 1100
        if "coder" in t: score += 1000
        if "code" in t: score += 850
        if "qwen" in t: score += 520
        if "deepseek" in t: score += 480
        if "glm" in t: score += 360
        if "gpt-oss" in t or "oss" in t: score += 330
        if not CODE_RX.search(t): score -= 180
    elif category in {"math", "logic"}:
        if "minimax" in t or "m3" in t: score += 1050
        if "deepseek" in t: score += 980
        if "qwen" in t: score += 900
        if "glm" in t: score += 850
        if "gpt-oss" in t or "oss" in t: score += 760
        if "llama" in t: score += 650
        if REASON_RX.search(t): score += 450
        if CODE_RX.search(t): score -= 240
    elif category in {"factual", "summary", "ner"}:
        if "qwen" in t: score += 880
        if "glm" in t: score += 850
        if "gpt-oss" in t or "oss" in t: score += 830
        if "deepseek" in t: score += 810
        if "minimax" in t or "m3" in t: score += 760
        if "llama" in t: score += 720
        if CODE_RX.search(t): score -= 260
    elif category == "sentiment":
        # Sentiment is easy; prefer a capable non-code chat model, not necessarily largest.
        if "qwen" in t: score += 820
        if "glm" in t: score += 790
        if "gpt-oss" in t or "oss" in t: score += 760
        if "minimax" in t or "m3" in t: score += 740
        if "deepseek" in t: score += 720
        if CODE_RX.search(t): score -= 260
        score -= max(int(size_b(m) * 2), 0)  # gently prefer smaller for easy task
    else:
        if "qwen" in t or "glm" in t or "deepseek" in t or "gpt-oss" in t: score += 700

    return score


def best_for(models: list[str], category: str) -> str:
    return max(models, key=lambda m: (_family_score(m, category), -len(m)))


def build_plan() -> ModelPlan:
    models = allowed_models()
    non_code = [m for m in models if not CODE_RX.search(m)] or models
    code_models = [m for m in models if CODE_RX.search(m)] or models

    small = best_for(non_code, "sentiment")
    language = best_for(non_code, "summary")
    reason = best_for(non_code, "math")
    code = best_for(code_models, "codegen")
    fallback = best_for(non_code, "factual")
    return ModelPlan(SMALL=small, LANGUAGE=language, REASON=reason, CODE=code, FALLBACK=fallback)


def tier_for(category: str, plan: ModelPlan) -> str:
    if category == "sentiment":
        return "SMALL"
    if category in {"factual", "summary", "ner"}:
        return "LANGUAGE"
    if category in {"math", "logic"}:
        return "REASON"
    if category in {"debug", "codegen"}:
        return "CODE"
    return "FALLBACK"


def model_for(category: str, plan: ModelPlan) -> str:
    return getattr(plan, tier_for(category, plan))


def fallback_model(category: str, plan: ModelPlan) -> str:
    primary = model_for(category, plan)
    order = [plan.FALLBACK, plan.REASON, plan.LANGUAGE, plan.CODE, plan.SMALL]
    for m in order:
        if m != primary:
            return m
    return primary
