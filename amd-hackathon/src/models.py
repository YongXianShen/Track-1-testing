"""Model tiering for Fireworks ALLOWED_MODELS with Gemma-aware language routing."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

_MOE = re.compile(r"(\d+)\s*x\s*(\d+)b\b", re.I)
_SIZE = re.compile(r"(\d+(?:p\d+|\.\d+)?)b\b", re.I)
_REASONING = re.compile(r"\br1\b|\bo1\b|\bqwq\b|reasoning|thinking|deepthink|-think\b", re.I)
_CODE = re.compile(r"\bcode|coder|kimi", re.I)
_QUANT = re.compile(r"nvfp4|fp4|fp8|int4|int8|awq|gptq|gguf|-q[48]\b|[48]bit|bnb", re.I)
_BAD = re.compile(r"embed|embedding|rerank|image|vision|audio|whisper|tts|guard|moderation|diffusion|flux|sdxl", re.I)
_GEMMA = re.compile(r"gemma", re.I)
_INSTRUCT = re.compile(r"instruct|chat|it|assistant|turbo", re.I)

UNKNOWN_SIZE = 999.0

@dataclass(frozen=True)
class Tiers:
    SMALL: str
    MEDIUM: str
    LARGE: str
    CODE: str
    GEMMA: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {"SMALL": self.SMALL, "MEDIUM": self.MEDIUM, "LARGE": self.LARGE, "CODE": self.CODE, "GEMMA": self.GEMMA}


def allowed_models() -> list[str]:
    models = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
    if not models:
        raise ValueError("ALLOWED_MODELS is empty")
    usable = [m for m in models if not _BAD.search(m)]
    return usable or models


def size_b(model_id: str) -> float:
    text = model_id.lower()
    moe = _MOE.findall(text)
    if moe:
        return float(max(int(a) * int(b) for a, b in moe))
    sizes = [float(m.group(1).replace("p", ".")) for m in _SIZE.finditer(text)]
    return max(sizes) if sizes else UNKNOWN_SIZE


def is_reasoning(model_id: str) -> bool:
    return bool(_REASONING.search(model_id))


def is_code(model_id: str) -> bool:
    return bool(_CODE.search(model_id))


def is_gemma(model_id: str) -> bool:
    return bool(_GEMMA.search(model_id))


def _quality_key(model_id: str) -> tuple[float, int, int]:
    # Larger first; instruct/chat preferred; quantized slightly de-ranked.
    return (size_b(model_id) - (0.6 if _QUANT.search(model_id) else 0.0), 1 if _INSTRUCT.search(model_id) else 0, -len(model_id))


def _cheap_key(model_id: str) -> tuple[float, int, int]:
    # Smaller first for cheap classification/sentiment, but avoid reasoning models in cheap path.
    return (size_b(model_id) + (200 if is_reasoning(model_id) else 0), 0 if _QUANT.search(model_id) else 1, len(model_id))


def build_tiers() -> Tiers:
    models = allowed_models()
    code_models = [m for m in models if is_code(m)]
    general = [m for m in models if not is_code(m)] or models
    non_reasoning = [m for m in general if not is_reasoning(m)] or general

    cheap_pool = sorted(non_reasoning, key=_cheap_key)
    small = cheap_pool[0]
    medium = cheap_pool[len(cheap_pool) // 2]
    large = max(general, key=_quality_key)
    code = max(code_models, key=_quality_key) if code_models else medium

    gemmas = [m for m in general if is_gemma(m)]
    # Prefer a strong Gemma for language tasks. If only tiny Gemma exists, it can still
    # be good for sentiment/summary/NER, but factual will stay on LARGE unless GEMMA_FACTUAL=1.
    gemma = max(gemmas, key=_quality_key) if gemmas else None

    return Tiers(SMALL=small, MEDIUM=medium, LARGE=large, CODE=code, GEMMA=gemma)


def tier_for(category: str, tiers: Tiers) -> str:
    use_gemma_factual = os.environ.get("GEMMA_FACTUAL", "0").strip().lower() in {"1", "true", "yes"}
    if category in {"sentiment", "summary", "ner"} and tiers.GEMMA:
        return "GEMMA"
    if category == "factual" and tiers.GEMMA and (use_gemma_factual or size_b(tiers.GEMMA) >= 25):
        return "GEMMA"
    if category in {"debug", "codegen"}:
        return "CODE"
    if category in {"math", "logic", "factual"}:
        return "LARGE"
    return "SMALL"


def model_for_tier(tier: str, tiers: Tiers) -> str:
    value = getattr(tiers, tier)
    if value is None:
        return tiers.LARGE if tier == "GEMMA" else tiers.MEDIUM
    return value
