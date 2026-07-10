# Brocacho Track 1 — Stable Plus V17

A no-paid-deployment hybrid AI router for Track 1 of the AMD Developer Hackathon: ACT II.

## Runtime contract

- Reads `/input/tasks.json`
- Writes `/output/results.json`
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` from the environment
- Uses only models supplied through `ALLOWED_MODELS`
- Does not require Gemma deployment or a personal API key

## Strategy

- High-confidence deterministic local solvers for selected tasks
- One Fireworks call for other tasks
- Fallback only when the first response fails, is empty, or is truncated
- Dynamic model routing for language, reasoning, and code tasks

## Docker image

`ghcr.io/yongxianshen/track-1-testing:latest`

The GitHub Actions workflow automatically rebuilds and pushes the Linux AMD64 image whenever `amd-hackathon/**` or the workflow file changes on `main`.
