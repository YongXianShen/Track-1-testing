from __future__ import annotations

import os

from src import batching, models, router, solvers


def category(prompt: str) -> str:
    return router.classify(prompt) or "factual"


def test_public_style_local_handlers() -> None:
    samples = {
        "rgb": "Name the three primary colors in the RGB color model and briefly explain why displays use RGB instead of RYB.",
        "ml": "What is the difference between machine learning and deep learning? Briefly explain how each works.",
        "memory": "Explain the difference between RAM and ROM in a computer. What is each type used for?",
        "stock": "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?",
        "recipe": "A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is needed for 30 cookies? If sugar costs $2.40 per cup, what is the total cost of sugar for 30 cookies?",
        "sentiment": "Classify the sentiment as Positive, Negative, or Neutral and give a one-sentence reason: 'The box was damaged, but the device worked perfectly and support resolved the issue quickly.'",
        "summary": "Summarize the following passage in exactly two sentences: 'A new system improves diagnosis. It finds patterns clinicians may miss. However, privacy and bias remain concerns. Regulation is still catching up.'",
        "ner": "Extract all named entities and label each as PERSON, ORGANIZATION, LOCATION, or DATE: 'On March 15 2023, Sundar Pichai announced that Google would open a lab in Zurich with ETH Zurich.'",
        "logic": "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?",
    }
    answers = {name: solvers.solve(category(prompt), prompt) for name, prompt in samples.items()}
    assert all(answers.values()), answers
    assert "1672" in answers["stock"]
    assert "$4.50" in answers["recipe"]
    assert answers["sentiment"].startswith("Mixed")
    assert len([x for x in answers["summary"].split(". ") if x]) == 2
    assert "Sundar Pichai — PERSON" in answers["ner"]
    assert "ETH Zurich — ORGANIZATION" in answers["ner"]
    assert "Sam owns the cat" in answers["logic"]


def test_batch_parser() -> None:
    categories = {"a": "factual", "b": "codegen"}
    text = '{"a":"Red, green, and blue.","b":"def f(x):\\n    return x"}'
    got = batching.parse_answers(text, set(categories), categories)
    assert got["a"] == "Red, green, and blue."
    assert got["b"].startswith("def f")


def test_allowed_plan() -> None:
    os.environ["ALLOWED_MODELS"] = ",".join([
        "minimax-m3",
        "kimi-k2p7-code",
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it-nvfp4",
    ])
    plan = models.build_plan()
    assert plan.REASON == "minimax-m3"
    assert plan.LANGUAGE == "minimax-m3"
    assert plan.CODE == "kimi-k2p7-code"
    assert "gemma" not in " ".join(plan.as_dict().values()).lower()


if __name__ == "__main__":
    test_public_style_local_handlers()
    test_batch_parser()
    test_allowed_plan()
    print("batch-lean tests passed")
