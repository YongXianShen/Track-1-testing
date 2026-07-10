# Track 1 Gemma-aware Hybrid Router V11

This version is based on the stable tiered-router approach:

- `/input/tasks.json` -> `/output/results.json`
- reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS`
- uses Gemma only when it appears in `ALLOWED_MODELS`
- prefers Gemma for sentiment, summary, and NER; uses strong general models for factual/math/logic and code models for debug/codegen
- includes narrow zero-token local solvers for very simple math/sentiment/logic tasks

Recommended submission environment defaults:

```text
REASONING_EFFORT=none
ENABLE_LOCAL=1
CONCURRENCY=5
GEMMA_FACTUAL=0
```

Turn on `GEMMA_FACTUAL=1` only if the deployed Gemma model is strong, e.g. a 25B+ model, and the last run suggests factual tasks are failing.
