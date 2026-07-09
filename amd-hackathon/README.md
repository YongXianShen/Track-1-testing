# AMD Hackathon Track 1 Agent

V7 is based on the most stable Fireworks-first version, with a few conservative local deterministic solvers for simple arithmetic, clear sentiment, and exact simple code/logic patterns.

## Contract

- Reads tasks from `/input/tasks.json`
- Writes results to `/output/results.json`
- Output schema: `[ { "task_id": ..., "answer": "..." } ]`
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` from the runtime environment
- Calls only models listed in `ALLOWED_MODELS`

## Useful environment knobs

Defaults are safe for hidden evaluation:

```bash
ENABLE_SAFE_LOCAL=1
ENABLE_REVIEW_PASS=0
MODEL_STRATEGY=stable
MAX_CONCURRENCY=2
```

If score drops, try the safest Fireworks-only mode by setting:

```bash
ENABLE_SAFE_LOCAL=0
MODEL_STRATEGY=stable
```

If model ranking seems bad, try:

```bash
MODEL_STRATEGY=first
```

## Docker image

Build linux/amd64 and push to GHCR, for example:

```bash
docker buildx build --platform linux/amd64 -t ghcr.io/yongxianshen/track-1-testing:latest --push .
```


## V10 note
This version keeps the V7 stable runtime, but prefers deployed Gemma models only for factual, summarisation, NER, and sentiment tasks when Gemma appears in `ALLOWED_MODELS`. Math, logic, debugging, and code generation keep the V7 ranking.
