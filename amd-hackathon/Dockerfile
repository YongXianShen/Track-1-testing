FROM --platform=$TARGETPLATFORM python:3.11-slim

ARG MODEL_URL="https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-IQ4_XS.gguf?download=true"
ARG MODEL_SHA256=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AGENT_VERSION="V17.10-Phi4-Batched-ZeroToken" \
    LOCAL_MODEL_PATH=/models/Phi-4-mini-instruct-IQ4_XS.gguf \
    LOCAL_MODEL_CTX=4096 \
    LOCAL_MODEL_THREADS=2 \
    LOCAL_MODEL_BATCH=128 \
    LOCAL_GENERAL_MAX_TOKENS=720 \
    LOCAL_CODE_MAX_TOKENS=640 \
    DEADLINE_SECONDS=450

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN (pip install --no-cache-dir --only-binary=:all: \
          --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
          llama-cpp-python==0.3.33 \
    || (apt-get update \
        && apt-get install -y --no-install-recommends build-essential cmake ninja-build \
        && CMAKE_ARGS="-DGGML_NATIVE=OFF" FORCE_CMAKE=1 \
           pip install --no-cache-dir llama-cpp-python==0.3.33 \
        && apt-get purge -y build-essential cmake ninja-build \
        && apt-get autoremove -y \
        && rm -rf /var/lib/apt/lists/*))

RUN mkdir -p /models \
    && curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 \
      "$MODEL_URL" -o "$LOCAL_MODEL_PATH" \
    && test "$(stat -c%s "$LOCAL_MODEL_PATH")" -gt 2100000000 \
    && if [ -n "$MODEL_SHA256" ]; then echo "$MODEL_SHA256  $LOCAL_MODEL_PATH" | sha256sum -c -; fi

COPY src ./src
RUN mkdir -p /input /output
ENTRYPOINT ["python", "-m", "src.main"]
