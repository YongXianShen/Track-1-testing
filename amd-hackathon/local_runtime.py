"""Start one bundled llama.cpp server and call it through localhost only."""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class LocalRuntime:
    def __init__(self) -> None:
        self.host = "127.0.0.1"
        self.port = int(os.environ.get("LOCAL_LLM_PORT", "8080"))
        self.base = f"http://{self.host}:{self.port}"
        self.model_path = os.environ.get("LOCAL_MODEL_PATH", "/models/model.gguf")
        self.server_bin = os.environ.get("LLAMA_SERVER_BIN", "/app/llama-server")
        self.threads = max(1, int(os.environ.get("LOCAL_THREADS", "2")))
        self.context = max(2048, int(os.environ.get("LOCAL_CONTEXT", "4096")))
        self.process: subprocess.Popen[bytes] | None = None
        self.log_file = None

    def start(self, timeout: float = 150.0) -> None:
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"Bundled model missing: {self.model_path}")
        self.log_file = open("/tmp/llama-server.log", "wb")
        command = [
            self.server_bin,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.context),
            "--threads", str(self.threads),
            "--threads-batch", str(self.threads),
            "--batch-size", "128",
            "--ubatch-size", "64",
            "--parallel", "1",
        ]
        self.process = subprocess.Popen(command, stdout=self.log_file, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.process.returncode}; see /tmp/llama-server.log")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception as exc:  # server is still loading
                last_error = str(exc)
            time.sleep(1)
        raise TimeoutError(f"Local model did not become ready: {last_error}")

    def complete(self, messages: list[dict[str, str]], max_tokens: int, timeout: float = 80.0) -> str:
        payload: dict[str, Any] = {
            "model": "local",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 7,
            "stream": False,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        }
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Local model HTTP {exc.code}: {detail}") from exc
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        return content if isinstance(content, str) else str(content)

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.log_file:
            self.log_file.close()
