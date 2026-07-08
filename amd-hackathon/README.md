# AMD Developer Hackathon Track 1 Agent

Accuracy-first Track 1 container for the General-Purpose AI Agent task.

## Contract

- Reads tasks from `/input/tasks.json`
- Writes `/output/results.json`
- Output schema:

```json
[
  { "task_id": "t1", "answer": "..." }
]
```

## Runtime environment

The evaluation harness injects:

- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL`
- `ALLOWED_MODELS`

The code reads model IDs from `ALLOWED_MODELS` at runtime and routes all API calls through `FIREWORKS_BASE_URL`.

## Accuracy mode defaults

This version prioritizes passing the accuracy gate:

- local solvers are disabled by default because hidden prompts vary
- model ranking is dynamic based on the allowed model names
- consensus + review pass are enabled by default

Useful environment overrides:

```bash
ENABLE_LOCAL_SOLVERS=0
ENABLE_CONSENSUS=1
ENABLE_REVIEW_PASS=1
MAX_CONCURRENCY=6
TASK_TIMEOUT_SECONDS=85
API_TIMEOUT_SECONDS=65
GLOBAL_TIMEOUT_SECONDS=560
```

After passing the accuracy gate, token efficiency can be improved by setting:

```bash
ENABLE_CONSENSUS=0
ENABLE_REVIEW_PASS=0
```

## Local test

```bash
python test_agent.py
```

## Build

```bash
docker buildx build --platform linux/amd64 -t your-registry/amd-track1:latest .
```
