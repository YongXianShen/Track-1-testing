# Track 1 No-Paid-Gemma Router V12

This version is designed for teams that do **not** want to pay for on-demand Gemma deployment.

## Strategy

- Reads `/input/tasks.json`
- Writes `/output/results.json`
- Uses only `ALLOWED_MODELS`
- Does **not** select Gemma by default (`ENABLE_GEMMA=0`)
- Uses narrow local solvers for simple math/sentiment/logic to save tokens
- Uses one Fireworks call per unsolved task, with fallback only if the first call fails/returns empty

## Model plan

The actual model names depend on `ALLOWED_MODELS` at runtime. The code logs them to stderr as `MODEL_PLAN` and writes `/output/model_usage.json`.

Default routing:

- `sentiment` -> SMALL capable non-Gemma chat model
- `factual`, `summary`, `NER` -> LANGUAGE model, preferring Qwen / GLM / GPT-OSS / DeepSeek / Minimax
- `math`, `logic` -> REASON model, preferring Minimax / DeepSeek / Qwen / GLM / GPT-OSS
- `debug`, `codegen` -> CODE model, preferring Kimi / coder / code models

## Recommended environment

```text
ENABLE_GEMMA=0
ENABLE_LOCAL=1
CONCURRENCY=4
REASONING_EFFORT=none
```

Only set `ENABLE_GEMMA=1` if your team has already deployed Gemma and it appears in `ALLOWED_MODELS`.
