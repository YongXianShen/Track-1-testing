# Track 1 V17.8 — Qwen2.5 3B Hybrid

Conservative hybrid built from the proven V17.5 batch router.

- Local model: Qwen2.5-3B-Instruct, GGUF Q4_K_M (4-bit)
- Default local categories: sentiment and summarisation
- Remote fallback: MiniMax M3 for general/reasoning, Kimi K2P7 Code for code
- Fireworks calls always use the injected `FIREWORKS_BASE_URL`
- No paid local deployment is required; weights are bundled at Docker build time

The default is deliberately conservative. To test more categories locally, set
`LOCAL_MODEL_CATEGORIES=sentiment,summary,ner,factual`. This can reduce tokens but may reduce accuracy.

Input: `/input/tasks.json`  
Output: `/output/results.json`
