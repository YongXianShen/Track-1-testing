# Brocacho Precision Router V14

A lightweight Track 1 agent designed for high accuracy with low Fireworks token use.

## Strategy

- One remote call per non-local task.
- A fallback call occurs only after a failed or empty primary response.
- Only pure arithmetic and fully parsed percentage inventory problems are solved locally.
- The strongest allowed instruction model handles language tasks.
- A strong reasoning model handles math and logic.
- A dedicated coder handles debugging and code generation.
- Gemma is disabled by default, so no paid on-demand deployment is required.

## Required runtime contract

- Reads `/input/tasks.json`
- Writes `/output/results.json`
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`
- Uses only model IDs supplied through `ALLOWED_MODELS`

## Model reporting

The container prints `MODEL_PLAN` to stderr and writes `/output/model_usage.json`.
The exact model IDs depend on the evaluator's `ALLOWED_MODELS` value.

## Local test

```bash
pip install -r requirements.txt
python test_agent.py
```

## Docker test

```bash
docker build --platform linux/amd64 -t track1-v14 .
docker run --rm \
  -e FIREWORKS_API_KEY=dummy \
  -e FIREWORKS_BASE_URL=https://example.invalid/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/gpt-oss-120b \
  -v "$PWD/input:/input" \
  -v "$PWD/output:/output" \
  track1-v14
```
