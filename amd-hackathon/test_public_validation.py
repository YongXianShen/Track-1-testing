"""Retired public judging examples: local checks for generic solver behavior."""
from src import prompts, router, solvers


def test_public_inventory_sequence():
    prompt = (
        "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. "
        "In Q2 it restocks 800 units. In Q3 it sells 640 units. "
        "How many units remain at the end of Q3?"
    )
    answer = solvers.solve("math", prompt)
    assert answer and "1672" in answer, answer


def test_public_recipe_scaling():
    prompt = (
        "A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is "
        "needed for 30 cookies? If sugar costs $2.40 per cup, what is the total cost?"
    )
    answer = solvers.solve("math", prompt)
    assert answer and "1.875" in answer and "$4.50" in answer, answer


def test_public_mixed_sentiment_mentions_both_sides():
    prompt = (
        "Classify the sentiment as Positive, Negative, or Neutral and give a one-sentence reason: "
        "'The product arrived two days late and the packaging was damaged, but the item worked "
        "perfectly and customer support resolved my complaint within an hour.'"
    )
    answer = solvers.solve("sentiment", prompt)
    assert answer and answer.startswith("Mixed")
    assert "negative:" in answer and "positive:" in answer


def test_summary_caps_are_tight_but_safe():
    _, two_sentences = prompts.render("summary", "Summarize in exactly two sentences: text")
    _, three_bullets = prompts.render(
        "summary", "Summarize in exactly three bullet points, each no longer than 15 words: text"
    )
    assert 80 <= two_sentences <= 130
    assert 80 <= three_bullets <= 150


def test_public_categories_route_correctly():
    assert router.classify("Explain the difference between RAM and ROM and what each is used for.") == "factual"
    assert router.classify("Summarize this passage in exactly three bullet points, each under 15 words.") == "summary"
    assert router.classify("Extract all named entities and label PERSON, ORGANIZATION, LOCATION, or DATE.") == "ner"
