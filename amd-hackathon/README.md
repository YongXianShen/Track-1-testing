# Brocacho Track 1 — Ultra-Local V17.6

Aggressive token-efficiency experiment built from the 94.7% / 5,027-token V17.4 baseline.

## Strategy

- High-confidence deterministic solvers answer suitable factual, math, sentiment, summary, NER, and small logic tasks locally.
- Remaining non-code tasks are sent together in one `minimax-m3` request.
- Remaining debug/code-generation tasks are sent together in one `kimi-k2p7-code` request.
- Missing or malformed batch answers fall back to the proven per-task route.
- Gemma is excluded, so no paid on-demand deployment is needed.

## Required environment

- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL`
- `ALLOWED_MODELS`

The harness supplies these values. All remote calls use `FIREWORKS_BASE_URL`, and selected models come only from `ALLOWED_MODELS`.

## Container contract

- Reads `/input/tasks.json`
- Writes `/output/results.json`
- Linux `amd64`
- Exit code `0` on success

## Important

This is an aggressive experiment intended to approach sub-1,000 scored tokens. Preserve the V17.4 image/commit before replacing `latest`; hidden-set accuracy cannot be guaranteed.


## V17.6 ultra-local mode

- Solves broad factual, math, sentiment, summary, NER, common code, and simple logic tasks locally.
- Sends at most three unresolved tasks to compact MiniMax/Kimi batches.
- Uses no Gemma deployment and no personal paid API.
- Default remote budget: `REMOTE_TASK_BUDGET=3`.
