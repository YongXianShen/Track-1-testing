# Track 1 Stable Lean V18

A conservative optimization of the proven V17 agent.

## What changed from V17

- Keeps the V17 dynamic Fireworks model strategy.
- Uses a separate strongest factual model role.
- Removes routine LLM routing calls; regex routing is deterministic by default.
- Adds general zero-token solvers for narrow arithmetic, clear sentiment, small ownership puzzles, a max-list debug pattern, and second-largest-distinct code generation.
- Uses shorter category prompts and lower output budgets.
- Uses `reasoning_effort=low` only for GPT-OSS math/logic tasks.
- Makes a fallback call only when the answer is empty or fails category validation.
- Enforces exact summary word/sentence constraints when safely possible.
- Gemma stays disabled by default, so no paid deployment is required.

## Runtime contract

- Reads `/input/tasks.json`
- Writes `/output/results.json`
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`
- Uses only models supplied in `ALLOWED_MODELS`
- All remote inference goes through `FIREWORKS_BASE_URL`

## Local test

```bash
pip install -r requirements.txt
python -m pytest -q
```
