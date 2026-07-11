# Track 1 Stable Token Trim V17.2

A minimal token-efficiency refinement of the proven V17 result.

## What changed from V17

Only the repeated category instruction text was shortened. The router, deterministic
solvers, model selection, output caps, retries, fallbacks, concurrency, and result
schema are unchanged. This minimizes regression risk while reducing Fireworks input
tokens.

## Model plan for the published allowed list

- `minimax-m3`: factual, math, sentiment (when local solver is not certain), summary, NER, and logic
- `kimi-k2p7-code`: debugging and code generation
- Gemma models: never called, so no paid deployment is needed

All selected model IDs still come from `ALLOWED_MODELS` at runtime.

## Required runtime contract

- Reads `/input/tasks.json`
- Writes `/output/results.json`
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`
- Sends every Fireworks call through `FIREWORKS_BASE_URL`
- Uses only models present in `ALLOWED_MODELS`
- Internal deadline remains 8.5 minutes

## Local test

```bash
pip install -r requirements.txt
python -m pytest -q
```
