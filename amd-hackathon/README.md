# Track 1 V17.8 — Gemma 2 2B Hybrid

Conservative hybrid built from the proven V17.5 batch router.

- Local model: Gemma-2-2B-IT, GGUF Q4_K_M (4-bit)
- Default local categories: sentiment and summarisation
- Remote fallback: MiniMax M3 for general/reasoning, Kimi K2P7 Code for code
- Fireworks calls always use the injected `FIREWORKS_BASE_URL`
- No paid deployment is required; weights are bundled at Docker build time

Gemma weights are license-controlled. If the Docker build receives HTTP 401/403,
accept the Gemma license on Hugging Face and supply a free `HF_TOKEN` as a build argument.
No paid model deployment is required.

Input: `/input/tasks.json`  
Output: `/output/results.json`
