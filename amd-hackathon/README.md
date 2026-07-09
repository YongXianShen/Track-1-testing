# AMD Hackathon Track 1 Agent - V6 Model-Ranking Repair

V6 is a safer accuracy repair version after V4/V5 stayed around the same accuracy.

Changes:

- keeps Fireworks-first answering
- improves model ranking for current Fireworks model names such as DeepSeek v4, GLM 5, Qwen 3.7 Plus, Kimi K2.7 Code, MiniMax M3, and GPT-OSS
- uses Kimi/code models first for code generation and debugging
- uses DeepSeek/GLM/Qwen-style models first for math, logic, factual, summary, NER, and sentiment
- disables the full verification pass by default because it did not improve the previous score and can rewrite correct answers
- keeps only very narrow exact local rules for obvious practice-style prompts
- broadens category detection for NER and sentiment variants

Required runtime contract:

- read `/input/tasks.json`
- write `/output/results.json`
- read `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`
- use only models from `ALLOWED_MODELS`

Docker image after GitHub Actions build:

```text
ghcr.io/yongxianshen/track-1-testing:latest
```

Optional switches:

```bash
ENABLE_VERIFY_PASS=0   # default; recommended for this V6 test
ENABLE_VERIFY_PASS=1   # only try this if V6 is still below the gate and you want a repair pass
ENABLE_EXACT_LOCAL=1   # default; only very narrow exact rules
MAX_CONCURRENCY=3      # default
```
