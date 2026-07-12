import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import local_model


def test_no_fireworks_reference():
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src").glob("*.py"))
    assert "AsyncOpenAI" not in source
    assert "FIREWORKS_BASE_URL" not in source


def test_batch_payload_parsing(monkeypatch):
    class FakeModel:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": '{"a":"Answer: 4","b":"Positive — clear praise."}'}}]}

    monkeypatch.setattr(local_model, "_MODEL", FakeModel())
    result = local_model.answer_batch([
        {"id": "a", "category": "math", "prompt": "2+2"},
        {"id": "b", "category": "sentiment", "prompt": "Great product"},
    ])
    assert result["a"] == "Answer: 4"
    assert result["b"].startswith("Positive")


if __name__ == "__main__":
    test_no_fireworks_reference()
    print("batched zero-token tests passed")
