# Track 1 Stable Plus V17

A conservative refinement of the proven V12 hybrid router.

## Strategy

- High-confidence deterministic arithmetic/sentiment/logic answers use zero Fireworks tokens.
- Every other task uses one category-appropriate model from `ALLOWED_MODELS`.
- Gemma stays disabled by default; no paid deployment is needed.
- A second call occurs only if the first answer is empty or truncated.
- Short adaptive output budgets reduce token use without compressing factual answers below the proven completeness range.

## Required runtime contract

- Reads `/input/tasks.json`
- Writes `/output/results.json`
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`
- Uses only models supplied in `ALLOWED_MODELS`

## Local test

```bash
pip install -r requirements.txt
python -m pytest -q
```
