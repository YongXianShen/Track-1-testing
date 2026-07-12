import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import local_model, router, solvers


def test_no_fireworks_dependency():
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src").glob("*.py"))
    assert "AsyncOpenAI" not in source
    assert "FIREWORKS_BASE_URL" not in source


def test_public_math_patterns():
    p1 = "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?"
    assert "1,672" in (solvers.solve("math", p1) or "") or "1672" in (solvers.solve("math", p1) or "")
    p2 = "A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is needed for 30 cookies? If sugar costs $2.40 per cup, what is the total cost?"
    answer = solvers.solve("math", p2) or ""
    assert "1.875" in answer and "4.50" in answer


def test_format_validators():
    ok, _ = local_model.validate("sentiment", "The box was damaged, but the device is flawless.", "Mixed — the box was damaged, but the device itself is flawless.")
    assert ok
    ok, _ = local_model.validate("summary", "Summarize in exactly two sentences: x", "First sentence. Second sentence.")
    assert ok
    ok, _ = local_model.validate("ner", "On March 15 2023, Alex joined Acme in Zurich.", "March 15 2023 — DATE\nAlex — PERSON\nAcme — ORGANIZATION\nZurich — LOCATION")
    assert ok


if __name__ == "__main__":
    test_no_fireworks_dependency()
    test_public_math_patterns()
    test_format_validators()
    print("zero-token tests passed")
