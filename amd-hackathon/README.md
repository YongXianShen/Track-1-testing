# AMD Hackathon Track 1 Agent - V4 Stable Accuracy

This version is a stable Fireworks-first agent for Track 1.

Key defaults:
- Reads `/input/tasks.json`
- Writes `/output/results.json`
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`
- Uses only models from `ALLOWED_MODELS`
- Disables risky local shortcut solvers
- Uses low concurrency to avoid rate-limit/time pressure
- Produces only `{task_id, answer}` in results

Docker image after GitHub Actions build:

```text
ghcr.io/yongxianshen/track-1-testing:latest
```

After passing the accuracy gate, optimize token usage later.
