# Testing NARE (Neuro-Adaptive-Reasoning-Engine)

## Devin Secrets Needed
- `GEMINI_API_KEY` — Required for all LLM-dependent tests (REM Sleep, Meta-Abduction LLM, SLOW/HYBRID routing, Sleep Phase crystallization). Get from https://aistudio.google.com/apikey

## Project Structure
- `nare/agent.py` — Core agent with 4-way routing (REFLEX/FAST/HYBRID/SLOW), REM Sleep, Sleep Phase
- `nare/llm.py` — Gemini API integration, skill generation, `repair_skill()`, stress tests
- `nare/memory.py` — Episodic + semantic + factual (RAG) memory with FAISS
- `nare/meta_abduction.py` — Cross-domain meta-rule discovery with LLM principle generation
- `nare/metrics.py` — MetricsTracker (routing, cost, convergence, stability)
- `nare/graph_memory.py` — Associative graph memory with Hebbian strengthening
- `nare/rl_retriever.py` — RL-based contextual bandit retriever
- `nare/neural_memory.py` — Titans/MIRAS neural memory with surprise gating
- `nare/sandbox.py` — AST-validated sandbox for skill execution
- `main.py` — CLI entrypoint (demo/interactive/benchmark modes)

## Running the App
```bash
cd /home/ubuntu/repos/Neuro-Adaptive-Reasoning-Engine
pip install -r requirements.txt
python main.py              # demo mode (default)
python main.py interactive  # REPL mode
python main.py benchmark    # run benchmarks
python main.py --query "What is 2+2?"  # single query
```

## Testing Approach
This is a Python library/CLI, NOT a web app. All testing is done via shell commands — no browser recording needed.

### Constructor Notes
- `NAREProductionAgent()` takes **no arguments**. It uses `MemorySystem()` default `persist_dir="memory_store"`.
- To isolate tests, clean `memory_store/` before creating the agent: `rm -rf memory_store`
- Do NOT pass `persist_dir` to `NAREProductionAgent()` — it will raise `TypeError`.

### Offline-Testable (no API key needed)
- Memory system: add/retrieve episodes, deduplication, forgetting
- Graph memory: add edges, Hebbian strengthening, multi-hop BFS, synaptic downscaling
- RL Retriever: value updates, reranking, epsilon decay
- Neural Memory: forward pass, surprise metric, consolidation
- Meta-Abduction (offline clustering): feature extraction, structural clustering (but LLM principle generation needs API key)
- MetricsTracker: record/compute all 4 metrics
- Sandbox: AST validation

### LLM-Dependent (needs GEMINI_API_KEY)
- Full agent.solve() flow (SLOW/HYBRID/FAST routing)
- Sleep Phase crystallization (skill generation)
- REM Sleep (stress test generation + `repair_skill()`)
- Meta-Abduction LLM principle generation (`_llm_generate_principle()`)
- Tree-of-Thoughts candidate generation and scoring

### Key Test Patterns

**FAST CACHE test**: Solve same query twice. First goes SLOW, second hits FAST CACHE (0 tokens). Verify `metrics.history` has 2 entries.

**REM Sleep repair test**: Call `repair_skill()` directly with a deliberately broken skill (trigger always False, execute returns wrong answer). Verify returned code is different and has valid structure.

**Meta-Abduction test**: Create 2+ skills with `python_code`, call `analyze_skills()`. Verify `abstract_pattern` is LLM-generated (NOT the template "Problems involving...").

**Amortization test**: Solve a math query via SLOW path, then solve the same query — should return via FAST in <0.01s with 0 tokens.

## Known Behaviors (Not Bugs)
- Keyword extraction in meta-abduction filters out words < 4 characters (e.g., "sum" = 3 chars is excluded)
- Free-tier Gemini API has rate limits (~15 RPM). Tests with multiple LLM calls may need `time.sleep(15)` between calls.
- The `gemma-3-27b-it` model might not always be available. If 429 errors occur, wait and retry.
- REFLEX path requires mature skills (maturity >= 3) accumulated over multiple sleep cycles — cannot be tested in a single session without mocking.

## Cleanup
After testing, clean up persisted state:
```bash
rm -rf memory_store/
```
