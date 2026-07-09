# AMD Hackathon Track 1 Agent - V5 Accuracy Repair

This version keeps the stable Fireworks-first approach from V4, but adds:

- stricter task-specific prompts
- better model ranking for current model families
- verification pass for math, logic, NER, debug, code, and exact-format summaries
- safer code-block cleanup for code tasks
- very narrow exact local rules only for obvious sample-style tasks

Required runtime contract:

- read `/input/tasks.json`
- write `/output/results.json`
- read `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`
- use only models from `ALLOWED_MODELS`

Docker image after GitHub Actions build:

```text
ghcr.io/yongxianshen/track-1-testing:latest
```

After passing the accuracy gate, try setting `ENABLE_VERIFY_PASS=0` to reduce token usage.
