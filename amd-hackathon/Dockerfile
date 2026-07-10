# Fully local Track 1 image: no Fireworks calls at runtime.
# The official llama.cpp server image supports linux/amd64.
FROM ghcr.io/ggml-org/llama.cpp:server

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG MODEL_REPO=unsloth/Qwen3.5-2B-GGUF
ARG MODEL_FILE=Qwen3.5-2B-Q4_K_M.gguf
RUN mkdir -p /models \
    && curl -fL --retry 6 --retry-delay 3 \
       "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}?download=true" \
       -o /models/model.gguf \
    && test -s /models/model.gguf

WORKDIR /app/track1
COPY . /app/track1

ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    LOCAL_MODEL_PATH=/models/model.gguf \
    LOCAL_MODEL_NAME=Qwen3.5-2B-Q4_K_M \
    LOCAL_THREADS=2 \
    LOCAL_CONTEXT=4096 \
    DEADLINE_SECONDS=510 \
    VERIFY_REASONING=1 \
    OMP_NUM_THREADS=2 \
    MALLOC_ARENA_MAX=2 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "/app/track1/agent.py"]
