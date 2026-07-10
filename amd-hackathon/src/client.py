"""Minimal resilient Fireworks client.

Every remote request uses the harness-provided FIREWORKS_BASE_URL. A second model
is called only after an actual request failure or empty response.
"""
from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from openai import APIStatusError, AsyncOpenAI

_TIMEOUT = float(os.environ.get("CALL_TIMEOUT_SECONDS", "30"))
_RATE_RETRIES = int(os.environ.get("RATE_LIMIT_RETRIES", "3"))
_NORMAL_RETRIES = int(os.environ.get("RETRIES", "1"))

_client: AsyncOpenAI | None = None
_unsupported_effort: set[str] = set()


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"],
            max_retries=0,
        )
    return _client


async def close() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def _status(error: Exception) -> int | None:
    return error.status_code if isinstance(error, APIStatusError) else None


def _effort(model: str, category: str) -> str | None:
    # Low effort controls verbose reasoning on GPT-OSS without disabling it.
    override = os.environ.get("REASONING_EFFORT", "").strip().lower()
    if override:
        return override
    if "gpt-oss" in model.lower():
        return "low"
    return None


async def complete(
    model: str,
    category: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    rate_attempts = 0
    normal_attempts = 0

    while True:
        effort = None if model in _unsupported_effort else _effort(model, category)
        kwargs: dict[str, Any] = {"reasoning_effort": effort} if effort else {}
        try:
            response = await asyncio.wait_for(
                _get_client().chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    **kwargs,
                ),
                timeout=_TIMEOUT,
            )
            usage = response.usage
            return response.choices[0].message.content or "", {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            }
        except Exception as error:
            last_error = error
            status = _status(error)

            if effort and status is not None and 400 <= status < 500 and status != 429:
                _unsupported_effort.add(model)
                continue

            if status == 429:
                rate_attempts += 1
                if rate_attempts > _RATE_RETRIES:
                    break
                await asyncio.sleep(2.5 * rate_attempts + random.uniform(0.0, 1.0))
                continue

            if status is not None and 400 <= status < 500:
                break

            normal_attempts += 1
            if normal_attempts > _NORMAL_RETRIES:
                break
            await asyncio.sleep(1.0 * normal_attempts)

    raise last_error or RuntimeError("Fireworks completion failed")
