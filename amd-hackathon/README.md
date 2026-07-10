# Track 1 V16 — Local Zero-Token Router

A fully local Track 1 agent. It makes **zero calls** to `FIREWORKS_BASE_URL`, so its recorded Fireworks token usage is zero.

## Model

Default bundled model:

- **Qwen3.5-2B**, `Q4_K_M` GGUF quantization
- Runtime: official `llama.cpp` server
- Model is downloaded while the Docker image is built and stored inside the image
- No paid deployment, personal API key, Ollama, or network access is required at evaluation time

Qwen3.5-2B was chosen over a 3B–4B model because the grader has only 4 GB RAM, 2 vCPU, and a 10-minute limit. The 4-bit model leaves more memory and time for the agent while still supporting all eight task categories.

## Pipeline

```text
/input/tasks.json
      ↓
regex category router
      ↓
high-confidence exact solver ── yes → answer locally
      ↓ no
bundled Qwen3.5-2B via llama.cpp
      ↓
format validation / short reasoning check
      ↓
/output/results.json
```

## Build

```bash
docker build --platform linux/amd64 -t track1-local-v16 .
```

The build downloads about 1.3 GB of model weights. The final image remains below the hackathon's 10 GB compressed limit.

## Local test

```bash
python3 test_agent.py
mkdir -p input output
# place tasks at input/tasks.json
docker run --rm \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  track1-local-v16
```

## Output

```json
[
  {"task_id":"t1","answer":"..."}
]
```

`/output/model_usage.json` records:

```json
{"fireworks_calls":0,"fireworks_tokens":0,"local_model":"Qwen3.5-2B-Q4_K_M"}
```

## Important

A 100% score cannot be guaranteed because the 19 evaluation tasks are hidden and the LLM judge is not perfectly deterministic. This version is designed for zero scored tokens and broad capability; it does not contain hidden answers or hardcoded evaluation data.
