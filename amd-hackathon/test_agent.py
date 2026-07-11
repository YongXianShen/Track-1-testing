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
    assert solvers.solve("sentiment", "Classify the sentiment: The product is great.").startswith("Positive")
    assert solvers.solve("sentiment", "Classify the sentiment: The product is not good.").startswith("Negative")
    assert solvers.solve("sentiment", "Classify the sentiment of this review: The battery is great, but the screen scratches easily.").startswith("Mixed")


def test_logic_and_code_solvers():
    puzzle = (
        "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. "
        "Sam does not own the bird. Jo owns the dog. Who owns the cat?"
    )
    assert solvers.solve("logic", puzzle) == "Answer: Sam"
    debug = "This function should return the max but has a bug: def get_max(nums): return nums[0]. Find and fix it."
    assert "return max(nums)" in solvers.solve("debug", debug)
    code = "Write a Python function that returns the second-largest number in a list, handling duplicates correctly."
    assert "def second_largest" in solvers.solve("codegen", code)


def test_dynamic_caps_and_validation():
    _, factual = prompts.render("factual", "Explain recursion.")
    _, one_sentence = prompts.render("summary", "Summarize in exactly one sentence: text")
    _, ten_words = prompts.render("summary", "Summarize in exactly 10 words: text")
    _, simple_code = prompts.render("codegen", "Write a Python function to add two numbers.")
    assert factual == 160
    assert ten_words < one_sentence <= 145
    assert simple_code <= 275
    fixed = prompts.postprocess("summary", "First sentence. Second sentence.", "Summarize in exactly one sentence")
    assert fixed == "First sentence; Second sentence."
    assert prompts.is_usable("sentiment", "Positive — clear approval.", "")
    assert not prompts.is_usable("sentiment", "It seems favorable.", "")


def test_model_plan():
    old = os.environ.get("ALLOWED_MODELS")
    os.environ["ALLOWED_MODELS"] = ",".join([
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/qwen3p7-plus",
        "accounts/fireworks/models/gpt-oss-120b",
    ])
    try:
        plan = models.build_plan()
        assert plan.CODE
        assert plan.FACTUAL
        assert models.reasoning_effort_for("math", "accounts/fireworks/models/gpt-oss-120b") == "low"
        assert models.reasoning_effort_for("factual", plan.FACTUAL) is None
    finally:
        if old is None:
            os.environ.pop("ALLOWED_MODELS", None)
        else:
            os.environ["ALLOWED_MODELS"] = old


def test_contract_writes_schema():
    practice = [
        {"task_id": "practice-01", "prompt": "What is the capital of Australia, and what body of water is it near?"},
        {"task_id": "practice-02", "prompt": "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?"},
        {"task_id": "practice-03", "prompt": "Classify the sentiment of this review: The battery life is great, but the screen scratches too easily."},
        {"task_id": "practice-07", "prompt": "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?"},
        {"task_id": "practice-08", "prompt": "Write a Python function that returns the second-largest number in a list, handling duplicates correctly."},
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
        assert all(item["answer"] for item in data[1:])
