"""Dynamic no-paid-deploy model selection for Track 1.

Only model IDs supplied by ALLOWED_MODELS are used. Gemma remains disabled by
default because on-demand deployment may cost money.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

BAD_RX = re.compile(r"embed|embedding|rerank|image|vision|audio|whisper|tts|guard|moderation|diffusion|flux|sdxl", re.I)
GEMMA_RX = re.compile(r"gemma", re.I)
CODE_RX = re.compile(r"code|coder|kimi", re.I)
REASON_RX = re.compile(r"\br1\b|qwq|reason|thinking|think|deepthink", re.I)
INSTRUCT_RX = re.compile(r"instruct|chat|assistant|turbo|\bit\b", re.I)
SIZE_RX = re.compile(r"(\d+(?:\.\d+|p\d+)?)\s*b\b", re.I)
MOE_RX = re.compile(r"(\d+)\s*x\s*(\d+)b\b", re.I)
UNKNOWN_SIZE = 12.0


@dataclass(frozen=True)
class ModelPlan:
    SMALL: str
    FACTUAL: str
    LANGUAGE: str
    REASON: str
    CODE: str
    FALLBACK: str

    def as_dict(self) -> dict[str, str]:
        return {
            "SMALL": self.SMALL,
            "FACTUAL": self.FACTUAL,
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
    enable_gemma = os.environ.get("ENABLE_GEMMA", "0").strip().lower() in {"1", "true", "yes"}
    usable = [
        m for m in models
        if not BAD_RX.search(m) and (enable_gemma or not GEMMA_RX.search(m))
    ]
    return usable or [m for m in models if not BAD_RX.search(m)] or models


def size_b(model_id: str) -> float:
    s = model_id.lower().replace("_", "-")
    moe = MOE_RX.findall(s)
    if moe:
        return max(float(a) * float(b) for a, b in moe)
    found: list[float] = []
    for value in SIZE_RX.findall(s):
        try:
            found.append(float(value.replace("p", ".")))
        except ValueError:
            pass
    return max(found) if found else UNKNOWN_SIZE


def _family_score(model: str, category: str) -> int:
    t = model.lower()
    score = min(int(size_b(model) * 6), 900)
    if INSTRUCT_RX.search(t):
        score += 70
    if REASON_RX.search(t) and category not in {"math", "logic"}:
        score -= 120

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
    elif category == "factual":
        if "qwen" in t: score += 900
        if "glm" in t: score += 875
        if "gpt-oss" in t or "oss" in t: score += 860
        if "deepseek" in t: score += 825
        if "minimax" in t or "m3" in t: score += 770
        if "llama" in t: score += 720
        if CODE_RX.search(t): score -= 280
    elif category in {"summary", "ner"}:
        if "qwen" in t: score += 880
        if "glm" in t: score += 850
        if "gpt-oss" in t or "oss" in t: score += 830
        if "deepseek" in t: score += 810
        if "minimax" in t or "m3" in t: score += 760
        if "llama" in t: score += 720
        if CODE_RX.search(t): score -= 260
    elif category == "sentiment":
        if "qwen" in t: score += 820
        if "glm" in t: score += 790
        if "gpt-oss" in t or "oss" in t: score += 760
        if "minimax" in t or "m3" in t: score += 740
        if "deepseek" in t: score += 720
        if CODE_RX.search(t): score -= 260
        score -= max(int(size_b(model) * 2), 0)
    return score


def best_for(candidates: list[str], category: str) -> str:
    return max(candidates, key=lambda m: (_family_score(m, category), -len(m)))


def build_plan() -> ModelPlan:
    available = allowed_models()
    non_code = [m for m in available if not CODE_RX.search(m)] or available
    code_models = [m for m in available if CODE_RX.search(m)] or available
    return ModelPlan(
        SMALL=best_for(non_code, "sentiment"),
        FACTUAL=best_for(non_code, "factual"),
        LANGUAGE=best_for(non_code, "summary"),
        REASON=best_for(non_code, "math"),
        CODE=best_for(code_models, "codegen"),
        FALLBACK=best_for(non_code, "factual"),
    )


def model_for(category: str, plan: ModelPlan) -> str:
    if category == "sentiment":
        return plan.SMALL
    if category == "factual":
        return plan.FACTUAL
    if category in {"summary", "ner"}:
        return plan.LANGUAGE
    if category in {"math", "logic"}:
        return plan.REASON
    if category in {"debug", "codegen"}:
        return plan.CODE
    return plan.FALLBACK


def fallback_model(category: str, plan: ModelPlan) -> str:
    primary = model_for(category, plan)
    if category in {"debug", "codegen"}:
        order = [plan.REASON, plan.FACTUAL, plan.LANGUAGE, plan.SMALL]
    elif category in {"math", "logic"}:
        order = [plan.FACTUAL, plan.LANGUAGE, plan.CODE, plan.SMALL]
    else:
        order = [plan.FACTUAL, plan.REASON, plan.LANGUAGE, plan.CODE, plan.SMALL]
    return next((m for m in order if m != primary), primary)


def reasoning_effort_for(category: str, model: str) -> str | None:
    """Use low reasoning only where it may gain correctness without large output.

    The parameter is limited to GPT-OSS-like models known to support it. Other
    models receive no extra field, avoiding a failed/retried request.
    """
    t = model.lower()
    if category in {"math", "logic"} and ("gpt-oss" in t or "oss" in t):
        return "low"
    return None
