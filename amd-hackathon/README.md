# Track 1 V17.9 — Phi-4 Zero-Token Guarded Agent

Fully local Track 1 experiment. It makes **zero Fireworks API calls**.

## Model

- Microsoft Phi-4-mini-instruct, 3.8B parameters
- GGUF IQ4_XS quantization (~2.22 GB)
- `llama-cpp-python`, CPU-only, 2 threads
- Bundled into the Docker image during build

## Pipeline

1. Read `/input/tasks.json`.
2. Classify each task into the eight Track 1 categories.
3. Use generic deterministic solvers when they can produce an exact answer.
4. Use the bundled local model for unresolved tasks.
5. Validate strict sentiment, summary, NER, math, logic, and code formats.
6. Write `/output/results.json`.

No hidden task IDs, prompt hashes, answer cache, Fireworks calls, or paid deployment are used.

## Runtime

Designed for `linux/amd64`, 4 GB RAM, 2 vCPU, and the 10-minute limit. The model is downloaded at Docker **build** time, never at evaluation runtime.

## Local smoke test

```bash
docker build --platform linux/amd64 -t track1-v179 .
mkdir -p input output
# place tasks.json in ./input
docker run --rm --platform linux/amd64 --memory=4g --cpus=2 \
  -v "$PWD/input:/input:ro" -v "$PWD/output:/output" track1-v179
cat output/results.json
```
