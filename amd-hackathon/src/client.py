"""Fireworks async client. All calls go through FIREWORKS_BASE_URL."""
from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from openai import APIStatusError, AsyncOpenAI

CALL_TIMEOUT_SECONDS = float(os.environ.get("CALL_TIMEOUT_SECONDS", "24"))
RETRIES = int(os.environ.get("RETRIES", "1"))
RATE_LIMIT_RETRIES = int(os.environ.get("RATE_LIMIT_RETRIES", "3"))
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "none")

_client: AsyncOpenAI | None = None
_no_effort: set[str] = set()


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


def _permanent(error: Exception) -> bool:
    return isinstance(error, APIStatusError) and error.status_code < 500 and error.status_code != 429


def _rate_limit(error: Exception) -> bool:
    return isinstance(error, APIStatusError) and error.status_code == 429


async def complete(model: str, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, dict[str, Any]]:
    last: Exception | None = None
    normal_failures = 0
    rate_failures = 0
    while True:
        send_effort = bool(REASONING_EFFORT) and model not in _no_effort
        kwargs = {"reasoning_effort": REASONING_EFFORT} if send_effort else {}
        try:
            resp = await asyncio.wait_for(
                get_client().chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                    **kwargs,
                ),
                timeout=CALL_TIMEOUT_SECONDS,
            )
            usage = resp.usage
            return resp.choices[0].message.content or "", {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            }
        except Exception as err:
            last = err
            if send_effort and _permanent(err):
                _no_effort.add(model)
                continue
            if _permanent(err):
                break
            if _rate_limit(err):
                rate_failures += 1
                if rate_failures > RATE_LIMIT_RETRIES:
                    break
                await asyncio.sleep(3.5 * rate_failures + random.uniform(0, 1.5))
                continue
            normal_failures += 1
            if normal_failures > RETRIES:
                break
            await asyncio.sleep(1.2 * normal_failures)
    raise last or RuntimeError("completion failed")
