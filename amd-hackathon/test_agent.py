import json, os, pathlib, subprocess, sys, tempfile

practice = [
    {"task_id":"practice-01","prompt":"What is the capital of Australia, and what body of water is it near?"},
    {"task_id":"practice-02","prompt":"A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?"},
    {"task_id":"practice-03","prompt":"Classify the sentiment of this review: The battery life is great, but the screen scratches too easily."},
]

def test_contract_writes_schema():
    with tempfile.TemporaryDirectory() as td:
        inp = pathlib.Path(td)/"tasks.json"
        out = pathlib.Path(td)/"results.json"
        inp.write_text(json.dumps(practice), encoding="utf-8")
        env = os.environ.copy()
        env.update({
            "INPUT_PATH": str(inp),
            "OUTPUT_PATH": str(out),
            "FIREWORKS_API_KEY": "dummy",
            "FIREWORKS_BASE_URL": "http://localhost:1/v1",
            "ALLOWED_MODELS": "accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/minimax-m3,accounts/fireworks/models/qwen3p7-plus,accounts/fireworks/models/gpt-oss-120b",
            "DEADLINE_SECONDS": "1",
            "CALL_TIMEOUT_SECONDS": "0.5",
        })
        subprocess.run([sys.executable, "-m", "src.main"], cwd=pathlib.Path(__file__).parent, env=env, timeout=8)
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) == len(practice)
        assert all(set(x) == {"task_id", "answer"} for x in data)
