from exact import solve
from tasking import classify, postprocess

CASES = {
    "What is the capital of Australia?": "factual",
    "A store has 240 items and sells 15%, then 60 more. How many remain?": "math",
    "Classify the sentiment: It is good but very slow.": "sentiment",
    "Summarize this in exactly one sentence: text": "summary",
    "Extract all named entities and their types from: Maria joined Acme in Berlin.": "ner",
    "This Python function has a bug: def f(x): return x[0]. Fix it.": "debug",
    "Three people each own a different pet. Who owns the cat?": "logic",
    "Write a Python function that returns the second-largest distinct number.": "codegen",
}

for prompt, expected in CASES.items():
    got = classify(prompt)
    assert got == expected, (prompt, expected, got)

assert solve("math", "What is 12 * (3 + 2)?") == "Answer: 60"
assert solve("math", "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?") == "Answer: 144"
assert postprocess("codegen", "```python\ndef f():\n    return 1\n```").startswith("def f")
print("local unit tests passed")
