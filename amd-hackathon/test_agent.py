import json
import os
import pathlib
import subprocess
import sys
import tempfile

from src import prompts, router, solvers


def test_router_regressions():
    cases = {
        "Write a Python function with error handling that parses an integer.": "codegen",
        "This Python function is broken and raises an exception. Find and fix it: def f(x): return x[0]": "debug",
        "Is this review positive or negative? The setup was effortless and reliable.": "sentiment",
        "What is the order of operations in arithmetic?": "factual",
        "Three people sit in a row. Ana is left of Ben. Who is first?": "logic",
        "A shop has 240 items and sells 15%. How many remain?": "math",
        "Summarize this paragraph in exactly one sentence: Example text.": "summary",
        "Extract all people and locations from: Ana visited Paris.": "ner",
    }
    for text, expected in cases.items():
        actual = router.classify(text)
        assert actual == expected, (text, expected, actual)


def test_local_sentiment_negation():
    assert solvers.solve_sentiment("Classify sentiment: This is not good.").startswith("Negative")
    assert solvers.solve_sentiment("Classify sentiment: This is not bad.").startswith("Positive")
    assert solvers.solve_sentiment("Classify sentiment: Great battery, but the screen scratches.").startswith("Mixed")


def test_adaptive_summary_caps():
    _, one_sentence = prompts.render("summary", "Summarize in exactly one sentence: text")
    _, ten_words = prompts.render("summary", "Summarize in exactly 10 words: text")
    assert one_sentence <= 110
    assert ten_words <= 40


def test_contract_writes_schema():
    practice = [
        {"task_id": "practice-01", "prompt": "What is the capital of Australia, and what body of water is it near?"},
        {"task_id": "practice-02", "prompt": "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?"},
        {"task_id": "practice-03", "prompt": "Classify the sentiment of this review: The battery life is great, but the screen scratches too easily."},
    ]
    with tempfile.TemporaryDirectory() as td:
        inp = pathlib.Path(td) / "tasks.json"
        out = pathlib.Path(td) / "results.json"
        inp.write_text(json.dumps(practice), encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "INPUT_PATH": str(inp),
                "OUTPUT_PATH": str(out),
                "FIREWORKS_API_KEY": "dummy",
                "FIREWORKS_BASE_URL": "http://localhost:1/v1",
                "ALLOWED_MODELS": "accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/minimax-m3,accounts/fireworks/models/qwen3p7-plus,accounts/fireworks/models/gpt-oss-120b",
                "DEADLINE_SECONDS": "1",
                "CALL_TIMEOUT_SECONDS": "0.5",
            }
        )
        subprocess.run([sys.executable, "-m", "src.main"], cwd=pathlib.Path(__file__).parent, env=env, timeout=8)
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) == len(practice)
        assert all(set(x) == {"task_id", "answer"} for x in data)


if __name__ == "__main__":
    test_router_regressions()
    test_local_sentiment_negation()
    test_adaptive_summary_caps()
    test_contract_writes_schema()
    print("All V15 tests passed")
