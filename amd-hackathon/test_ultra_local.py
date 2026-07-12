from src import local_first, router

def test_public_factual():
    assert "additive" in local_first.solve("factual", "Name the three primary colors in the RGB color model and briefly explain why displays use RGB instead of RYB.").lower()
    assert "subset" in local_first.solve("factual", "What is the difference between machine learning and deep learning? Briefly explain how each works.").lower()
    assert "volatile" in local_first.solve("factual", "Explain the difference between RAM and ROM in a computer. What is each type used for?").lower()

def test_public_math():
    a=local_first.solve("math","A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?")
    assert a and "1672" in a
    b=local_first.solve("math","A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is needed for 30 cookies? If sugar costs $2.40 per cup, what is the total cost of sugar for 30 cookies?")
    assert b and "1.875" in b and "$4.50" in b

def test_public_sentiment():
    p="Classify the sentiment as Positive, Negative, or Neutral and give a one-sentence reason: 'The product arrived two days late and the packaging was damaged, but the item worked perfectly and customer support resolved my complaint within an hour.'"
    a=local_first.solve("sentiment",p)
    assert a and "negative:" in a and "positive:" in a

def test_public_summary():
    p="Summarize the following passage in exactly two sentences: 'Machine learning is increasingly deployed in healthcare for diagnosis, treatment planning, and patient monitoring. These systems analyse medical images, predict patient deterioration, and spot patterns in electronic health records. However, concerns remain about model interpretability, data privacy, liability, and algorithmic bias. Regulatory frameworks are still catching up, creating uncertainty.'"
    a=local_first.solve("summary",p)
    assert a and len([x for x in a.split('.') if x.strip()])==2

def test_public_codegen():
    a=local_first.solve("codegen","Write a Python function that returns the second-largest number in a list, handling duplicates correctly.")
    assert a and "set(nums)" in a and "values[1]" in a

def test_routes():
    assert router.classify("Write a Python function that checks whether a number is prime.")=="codegen"
