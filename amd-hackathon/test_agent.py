import json
import os
import pathlib
import subprocess
import sys
import tempfile

from src import models, prompts, router, solvers

ROUTING_CASES = {
    "What is sentiment analysis in NLP?": "factual",
    "Classify this review as positive, negative, or neutral: It works well.": "sentiment",
    "Summarize this passage in one sentence: Long text here.": "summary",
    "Extract all people, organizations, locations and dates from the sentence.": "ner",
    "Explain what a Python function is.": "factual",
    "Write a Python function that returns the second-largest distinct value.": "codegen",
    "This function should return the maximum but returns nums[0]. Find and fix it.": "debug",
    "A store has 240 items and sells 15%, then 60 more. How many remain?": "math",
    "Three people each own a different pet. Who owns the cat?": "logic",
    "What is the order of operations?": "factual",
    "Who is the oldest person alive?": "factual",
}


def test_router_cases():
    for text, expected in ROUTING_CASES.items():
        assert router.classify(text) == expected, (text, router.classify(text), expected)


def test_local_solvers():
    assert solvers.solve("math", "What is 17 * 8?") == "Answer: 136"
    assert solvers.solve("math", "What is 15% of 240?") == "Answer: 36"
    assert solvers.solve("math", "A store has 240 items. It sells 15% and then 60 more. How many remain?") == "Answer: 144"
    assert solvers.solve("sentiment", "Classify the sentiment of this review: The battery is great, but the screen scratches easily.").startswith("Mixed")
    assert solvers.solve("sentiment", "Classify: The product is good.") is None  # ambiguous/low confidence -> model


def test_dynamic_caps():
    _, one_sentence = prompts.render("summary", "Summarize in exactly one sentence: text")
    _, ten_words = prompts.render("summary", "Summarize in exactly 10 words: text")
    _, simple_code = prompts.render("codegen", "Write a Python function to add two numbers.")
    assert ten_words < one_sentence <= 170
    assert simple_code <= 360


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
        env.update({
            "INPUT_PATH": str(inp),
            "OUTPUT_PATH": str(out),
            "FIREWORKS_API_KEY": "dummy",
            "FIREWORKS_BASE_URL": "http://localhost:1/v1",
            "ALLOWED_MODELS": "accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/minimax-m3,accounts/fireworks/models/qwen3p7-plus,accounts/fireworks/models/gpt-oss-120b",
            "DEADLINE_SECONDS": "1",
            "CALL_TIMEOUT_SECONDS": "0.5",
        })
        subprocess.run([sys.executable, "-m", "src.main"], cwd=pathlib.Path(__file__).parent, env=env, timeout=8, check=False)
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) == len(practice)
        assert all(set(item) == {"task_id", "answer"} for item in data)


def test_published_allowed_model_plan(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_MODELS",
        "minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4",
    )
    plan = models.build_plan()
    assert plan.SMALL == "minimax-m3"
    assert plan.LANGUAGE == "minimax-m3"
    assert plan.REASON == "minimax-m3"
    assert plan.CODE == "kimi-k2p7-code"
    assert plan.FALLBACK == "minimax-m3"
    assert all("gemma" not in model.lower() for model in plan.as_dict().values())
