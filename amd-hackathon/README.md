# AMD Hackathon Track 1 Agent - V3 Hybrid Accuracy

This container reads tasks from `/input/tasks.json` and writes `/output/results.json`.

V3 changes:
- restores high-confidence local solvers for easy math/sentiment/debug/logic/code patterns;
- uses Fireworks for anything uncertain;
- lowers default concurrency to reduce rate-limit/time pressure;
- disables consensus/review passes by default to avoid overwriting correct answers and timing out;
- keeps all model IDs from `ALLOWED_MODELS` and all calls through `FIREWORKS_BASE_URL`.

Recommended first submission env defaults:

```bash
ENABLE_LOCAL_SOLVERS=1
ENABLE_CONSENSUS=0
ENABLE_REVIEW_PASS=0
MAX_CONCURRENCY=3
```

If this passes the accuracy gate, then tune token usage. If it still fails and logs show no timeouts/rate limits, try a second run with:

```bash
ENABLE_REVIEW_PASS=1
ENABLE_CONSENSUS=0
MAX_CONCURRENCY=2
```
