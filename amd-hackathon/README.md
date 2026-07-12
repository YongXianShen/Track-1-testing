# Track 1 V17.10 — Phi-4 Batched Zero-Token Agent

A fully local Track 1 agent using bundled Phi-4-mini-instruct IQ4_XS.

## Runtime fix versus V17.9

V17.9 generated one model response per unresolved task and timed out on the 2-vCPU grader. V17.10:

- loads Phi-4 once;
- runs deterministic Python solvers first;
- sends all remaining non-code tasks in one compact JSON batch;
- sends remaining code tasks in one compact JSON batch;
- makes no repair generations;
- writes a result for every task before exit;
- makes zero Fireworks calls.

## Contract

- Reads `/input/tasks.json`
- Writes `/output/results.json`
- Linux/amd64 Docker image
- No Fireworks dependency or paid deployment

## Default runtime settings

- Context: 4096
- Threads: 2
- Batch size: 128
- General output cap: 720 tokens
- Code output cap: 640 tokens
- Internal deadline: 450 seconds
