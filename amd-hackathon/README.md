# Track 1 Judge-Aware Lean V17.4

A conservative update to the proven V17.2 router. It uses only models supplied
through `ALLOWED_MODELS`, excludes paid on-demand Gemma deployments, and keeps
MiniMax M3 for non-code tasks plus Kimi K2P7 Code for code tasks.

The update adds generic zero-token solvers for ordered stock changes and recipe
scaling, improves mixed-sentiment reasons so both sides are explicitly cited,
and tightens category prompts around the public judging principles.

## Required runtime contract

- Reads `/input/tasks.json`
- Writes `/output/results.json`
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`
- Routes every Fireworks call through `FIREWORKS_BASE_URL`
- Produces one `{ "task_id", "answer" }` object per input task

## Expected model plan for the published list

- `minimax-m3`: factual, math, sentiment, summary, NER, logic
- `kimi-k2p7-code`: debugging and code generation
- Gemma models: not called; no paid deployment required
