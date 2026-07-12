FROM --platform=$TARGETPLATFORM python:3.11-slim

ARG MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true"
ARG MODEL_SHA256=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLAMA_MODEL_PATH=/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    LLAMA_CTX=2048 \
    LLAMA_THREADS=2 \
    LLAMA_BATCH=128 \
    LLAMA_CATEGORIES=sentiment,summary,ner

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Prefer the maintained CPU wheel. If the wheel index is temporarily unavailable,
# compile a generic x86-64 build instead of failing the submission build.
RUN pip install --no-cache-dir "openai>=1.35.0,<2" \
    && (pip install --no-cache-dir --only-binary=:all: \
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
    && curl -fL --retry 5 --retry-delay 4 --connect-timeout 30 \
      "$MODEL_URL" -o "$LLAMA_MODEL_PATH" \
    && test "$(stat -c%s "$LLAMA_MODEL_PATH")" -gt 1500000000 \
    && if [ -n "$MODEL_SHA256" ]; then echo "$MODEL_SHA256  $LLAMA_MODEL_PATH" | sha256sum -c -; fi

COPY src ./src
RUN mkdir -p /input /output
ENTRYPOINT ["python", "-m", "src.main"]
