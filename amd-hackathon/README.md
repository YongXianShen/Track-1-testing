# Track 1 Stable Precision Router V15

This version is built directly from the proven V12 baseline. It keeps the same no-paid-deployment model plan and changes only conservative, testable areas.

## What changed from V12

- fixes code-generation prompts that mention **error handling** being misrouted as debugging;
- recognizes sentiment prompts such as **positive or negative**;
- avoids treating ordinary factual uses of **order** as logic puzzles;
- handles negation in the zero-token sentiment solver (`not good`, `not bad`);
- keeps factual answers complete (`under 120 words`);
- applies adaptive output caps for exact-length summaries;
- reduces code-generation cap from 620 to 520 tokens;
- still uses one Fireworks call per unsolved task and retries only after an empty/failed response.

## Cost

No Gemma deployment, local LLM, paid endpoint, or personal API key is required. The evaluation harness supplies `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`.

## Required contract

- read `/input/tasks.json`;
- write `/output/results.json`;
- use only models in `ALLOWED_MODELS`;
- route every Fireworks call through `FIREWORKS_BASE_URL`.

## Recommended environment

```text
ENABLE_GEMMA=0
ENABLE_LOCAL=1
CONCURRENCY=4
REASONING_EFFORT=none
```

## Test

```bash
python test_agent.py
```
