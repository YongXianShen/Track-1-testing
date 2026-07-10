"""Model selection optimized for benchmark accuracy, not dollar price.

Track 1 ranks by token count, not model size or API price. Therefore easy tasks
still use the strongest concise instruction model; code tasks use the strongest
coder. Gemma is excluded by default because it may require paid deployment.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

_UNSUITABLE = re.compile(
    r"embed|embedding|rerank|image|vision|audio|whisper|tts|guard|moderation|"
    r"diffusion|flux|sdxl",
    re.I,
)
_GEMMA = re.compile(r"gemma", re.I)
_CODE = re.compile(r"coder|code|kimi", re.I)
_REASONING_ONLY = re.compile(r"\br1\b|qwq|deepthink|reasoning-only", re.I)
_SIZE = re.compile(r"(?<!\d)(\d+(?:[._p]\d+)?)\s*b(?![a-z])", re.I)


@dataclass(frozen=True)
class ModelPlan:
    GENERAL: str
    REASON: str
    CODE: str
    BACKUP: str

    def as_dict(self) -> dict[str, str]:
        return {
            "GENERAL": self.GENERAL,
            "REASON": self.REASON,
            "CODE": self.CODE,
            "BACKUP": self.BACKUP,
        }


def _allowed() -> list[str]:
    raw = [item.strip() for item in os.environ.get("ALLOWED_MODELS", "").split(",") if item.strip()]
    if not raw:
        raise ValueError("ALLOWED_MODELS is empty")
    allow_gemma = os.environ.get("ENABLE_GEMMA", "0").lower() in {"1", "true", "yes"}
    filtered = [
        model for model in raw
        if not _UNSUITABLE.search(model) and (allow_gemma or not _GEMMA.search(model))
    ]
    return filtered or [m for m in raw if not _UNSUITABLE.search(m)] or raw


def _size(model: str) -> float:
    values: list[float] = []
    for raw in _SIZE.findall(model.lower().replace("-", "_")):
        try:
            values.append(float(raw.replace("_", ".").replace("p", ".")))
        except ValueError:
            pass
    return max(values, default=0.0)


def _quality(model: str, role: str) -> int:
    name = model.lower()
    score = min(int(_size(model) * 2), 500)

    # Prefer instruction/chat variants and avoid verbose reasoning-only variants.
    if re.search(r"instruct|chat|assistant|turbo|plus|pro", name):
        score += 90
    if _REASONING_ONLY.search(name):
        score -= 250

    if role == "code":
        if "kimi" in name and "code" in name: score += 1800
        elif "kimi" in name: score += 1550
        if "coder" in name: score += 1500
        if re.search(r"(^|[-_/])code($|[-_/])", name): score += 1200
        if "deepseek" in name: score += 750
        if "qwen" in name: score += 720
        if "gpt-oss" in name: score += 680
        if "glm" in name: score += 640
    elif role == "reason":
        if "deepseek" in name and ("v4" in name or "pro" in name): score += 1500
        elif "deepseek" in name: score += 1300
        if "qwen3" in name or "qwen-3" in name: score += 1450
        elif "qwen" in name: score += 1250
        if "gpt-oss-120b" in name or "gpt_oss_120b" in name: score += 1420
        elif "gpt-oss" in name: score += 1200
        if "glm-5" in name or "glm5" in name: score += 1400
        elif "glm" in name: score += 1220
        if "minimax" in name: score += 1120
        if _CODE.search(name): score -= 300
    else:  # strongest concise general instruction model
        if "gpt-oss-120b" in name or "gpt_oss_120b" in name: score += 1500
        elif "gpt-oss" in name: score += 1260
        if "qwen3" in name or "qwen-3" in name: score += 1460
        elif "qwen" in name: score += 1240
        if "glm-5" in name or "glm5" in name: score += 1430
        elif "glm" in name: score += 1230
        if "deepseek" in name and ("v4" in name or "pro" in name): score += 1400
        elif "deepseek" in name: score += 1210
        if "minimax" in name: score += 1110
        if "llama-4" in name: score += 1050
        if _CODE.search(name): score -= 350

    return score


def _best(models: list[str], role: str) -> str:
    return max(models, key=lambda model: (_quality(model, role), -len(model)))


def build_plan() -> ModelPlan:
    models = _allowed()
    non_code = [m for m in models if not _CODE.search(m)] or models
    code_models = [m for m in models if _CODE.search(m)] or models

    general = _best(non_code, "general")
    reason = _best(non_code, "reason")
    code = _best(code_models, "code")

    remaining = [m for m in non_code if m not in {general, reason}]
    backup = _best(remaining, "general") if remaining else (reason if reason != general else code)
    return ModelPlan(general, reason, code, backup)


def primary_for(category: str, plan: ModelPlan) -> str:
    if category in {"math", "logic"}:
        return plan.REASON
    if category in {"debug", "codegen"}:
        return plan.CODE
    return plan.GENERAL


def fallback_for(category: str, plan: ModelPlan) -> str:
    primary = primary_for(category, plan)
    for candidate in (plan.BACKUP, plan.GENERAL, plan.REASON, plan.CODE):
        if candidate != primary:
            return candidate
    return primary
