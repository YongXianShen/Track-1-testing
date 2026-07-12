from src import local_llama


def test_default_categories():
    assert local_llama.enabled_for("sentiment")
    assert local_llama.enabled_for("summary")
    assert local_llama.enabled_for("ner")
    assert not local_llama.enabled_for("math")


def test_validation():
    assert local_llama.validate("sentiment", "Classify sentiment", "Mixed — late delivery, but the item works well.")
    assert not local_llama.validate("sentiment", "Classify sentiment", "It is okay")
    assert local_llama.validate("ner", "Extract entities", "Google — ORGANIZATION\nZurich — LOCATION")
    assert local_llama.validate("summary", "Summarize in exactly two sentences", "One point. Another point.")
    assert not local_llama.validate("summary", "Summarize in exactly two sentences", "Only one sentence.")
