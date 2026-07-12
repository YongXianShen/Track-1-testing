# Track 1 Llama-Hybrid V17.7

A conservative hybrid version built from V17.5.

## Local model

- **Model:** Meta Llama 3.2 3B Instruct
- **Quantization:** Q4_K_M GGUF (4-bit)
- **Runtime:** llama-cpp-python / llama.cpp, CPU only
- **Bundled at Docker build:** yes
- **Runtime download:** no
- **Paid deployment:** no

The model is downloaded during the Docker build from the public GGUF repository
`bartowski/Llama-3.2-3B-Instruct-GGUF`. Use remains subject to the Meta Llama
3.2 Community License and Acceptable Use Policy.

## Safe default routing

1. Deterministic Python solvers run first.
2. Llama runs locally for `sentiment,summary,ner`.
3. Invalid local answers fall back to the proven Fireworks batch route.
4. MiniMax M3 handles remaining non-code tasks.
5. Kimi K2P7 Code handles remaining code tasks.

This default is designed to reduce Fireworks tokens without replacing every
high-accuracy remote answer with a 3B local model.

## Optional modes

```text
LLAMA_CATEGORIES=sentiment,summary,ner,factual
```

adds factual tasks to local inference for a more aggressive token reduction.

```text
LOCAL_ONLY=1
LLAMA_CATEGORIES=factual,math,sentiment,summary,ner,logic,debug,codegen
```

uses zero Fireworks calls, but is experimental and is not recommended when the
accuracy gate matters.

## Contract

- Reads `/input/tasks.json`
- Writes `/output/results.json`
- Writes optional diagnostics to `/output/model_usage.json`
- Default internal deadline: 8.5 minutes
- Intended platform: `linux/amd64`, 4 GB RAM, 2 vCPU

## Local smoke test

```bash
docker build --platform linux/amd64 -t track1-llama-v177 .
mkdir -p input output
# place tasks.json in ./input
docker run --rm --memory=4g --cpus=2 \
  -e FIREWORKS_API_KEY=dummy \
  -e FIREWORKS_BASE_URL=http://127.0.0.1:9/v1 \
  -e ALLOWED_MODELS=minimax-m3,kimi-k2p7-code \
  -v "$PWD/input:/input" -v "$PWD/output:/output" \
  track1-llama-v177
```

For a true hybrid test, use valid Fireworks variables. For a local-only test,
set `LOCAL_ONLY=1` and no Fireworks variables are required.
