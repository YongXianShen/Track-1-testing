"""Async Fireworks client. All model calls go through FIREWORKS_BASE_URL."""
from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from openai import APIStatusError, AsyncOpenAI

RETRIES = int(os.environ.get("RETRIES", "2"))
RATE_LIMIT_RETRIES = int(os.environ.get("RATE_LIMIT_RETRIES", "4"))
CALL_TIMEOUT_SECONDS = float(os.environ.get("CALL_TIMEOUT_SECONDS", "25"))
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "none")

_client: AsyncOpenAI | None = None
_NO_EFFORT_PARAM: set[str] = set()


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"],
            max_retries=0,
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def _is_permanent(error: Exception) -> bool:
    return isinstance(error, APIStatusError) and error.status_code < 500 and error.status_code != 429


def _is_rate_limit(error: Exception) -> bool:
    return isinstance(error, APIStatusError) and error.status_code == 429


async def complete(model: str, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    failures = 0
    rate_hits = 0
    while True:
        sent_effort = bool(REASONING_EFFORT) and model not in _NO_EFFORT_PARAM
        kwargs = {"reasoning_effort": REASONING_EFFORT} if sent_effort else {}
        try:
            response = await asyncio.wait_for(
                get_client().chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                    **kwargs,
                ),
                timeout=CALL_TIMEOUT_SECONDS,
            )
            usage = response.usage
            return response.choices[0].message.content or "", {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            }
        except Exception as error:
            last_error = error
            if sent_effort and _is_permanent(error):
                _NO_EFFORT_PARAM.add(model)
                continue
            if _is_permanent(error):
                break
            if _is_rate_limit(error):
                rate_hits += 1
                if rate_hits > RATE_LIMIT_RETRIES:
                    break
                await asyncio.sleep(4 * rate_hits + random.uniform(0, 2.5))
                continue
            failures += 1
            if failures > RETRIES:
                break
            await asyncio.sleep(2 * failures - 1)
    raise last_error or RuntimeError("completion failed")
