from src import local, prompts, router

PRACTICE = {
    "factual": "What is the capital of Australia, and what body of water is it near?",
    "math": "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?",
    "sentiment": "Classify the sentiment of this review: The battery life is great, but the screen scratches too easily.",
    "summary": "Summarize the following in exactly one sentence: A long sample paragraph.",
    "ner": "Extract all named entities and their types from: Maria Sanchez joined Fireworks AI in Berlin last March.",
    "debug": "This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it.",
    "logic": "Three friends each own a different pet. Jo owns the dog. Who owns the cat?",
    "codegen": "Write a Python function that returns the second-largest number in a list, handling duplicates correctly.",
}

for expected, prompt in PRACTICE.items():
    actual = router.classify(prompt)
    assert actual == expected, (expected, actual, prompt)

assert local.solve("math", PRACTICE["math"]) == "Answer: 144"
assert local.solve("sentiment", PRACTICE["sentiment"]) is None

messages, limit = prompts.render("summary", PRACTICE["summary"])
assert len(messages) == 2
assert limit == 105
print("All local tests passed.")
