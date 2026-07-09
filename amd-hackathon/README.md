# AMD Developer Hackathon Track 1 Agent - V9 Friend-Style Routing

This container reads `/input/tasks.json` and writes `/output/results.json`.

V9 changes compared with the previous submitted build:

- Uses safer local deterministic solvers only for obvious arithmetic, sentiment, simple logic, and simple code/debug prompts.
- Uses stronger model ranking based on the higher-scoring reference approach:
  - code/debug: prefer Kimi/code/coder models
  - math/logic: prefer Minimax M3 / reasoning models
  - factual, summarisation, sentiment, NER: prefer Gemma 4 31B IT / strong chat models
- Enables review pass by default.
- Enables consensus for hard categories by default when time allows.
- Sends `reasoning_effort="none"` only to Minimax/M3-style models, with fallback if rejected.

Environment variables used by the judge:

- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL`
- `ALLOWED_MODELS`

Useful tuning flags:

- `ENABLE_SAFE_LOCAL=0` disables local deterministic solvers.
- `ENABLE_REVIEW_PASS=0` disables verification pass.
- `ENABLE_CONSENSUS=0` disables second-model consensus.
- `MODEL_STRATEGY=first` uses the allowed-model list order instead of ranked routing.

Docker image should be built for `linux/amd64` and remain under the 10GB compressed limit.
