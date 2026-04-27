# Testing NARE (Neuro-Adaptive-Reasoning-Engine)

## Overview
NARE is a Python cognitive architecture with 4 routing paths (FAST, REFLEX, HYBRID, SLOW). All testing is shell-based (no GUI). Tests require a live Gemini API key for end-to-end verification.

## Devin Secrets Needed
- `GEMINI_API_KEY` — Gemini API key from https://aistudio.google.com/apikey (org-scoped)

## Setup
```bash
cd /home/ubuntu/repos/Neuro-Adaptive-Reasoning-Engine
pip install -r requirements.txt
echo "GEMINI_API_KEY=${GEMINI_API_KEY}" > .env
```

## Testing Approach

### Offline Tests (no API key needed)
- **Module imports**: `python -c "from nare.agent import NAREProductionAgent"`
- **Memory system**: Create `MemorySystem`, test `add_episode()` / `retrieve_episodes()` with numpy arrays
- **Sandbox security**: Verify AST validation blocks dangerous code
- **CLI**: `python main.py --help` should show 3 modes (demo, interactive, benchmark)
- **Missing API key error**: Remove `.env`, run `python main.py demo` — should exit code 1 with helpful message

### Online Tests (API key required)
- **Embedding API**: `get_embedding('test')` should return 3072-dim vector
- **SLOW path**: First query goes through full Chain-of-Thought generation (~20s)
- **FAST path**: Exact repeat of same query hits cache (~0.01s)
- **HYBRID path**: Similar-but-different query triggers delta reasoning (~2s)
- **REFLEX path**: Requires mature skills from Sleep Phase crystallization — hard to trigger in a single test session. Verify return dict structure via source analysis instead.

### Adversarial Bug Verification
For each bug fix, verify BOTH that the fix works AND that the old code would crash:
- `__builtins__` fix: In script context, `__builtins__` is a module (not dict). Test `isinstance(__builtins__, dict)` returns False.
- FAISS 1D fix: Pass `np.random.randn(3072)` (1D) to memory methods. Confirm `faiss.normalize_L2(1d_array)` raises `tuple index out of range`.
- REFLEX return keys: Check both REFLEX and REFLEX_PROVISIONAL return dicts contain all 6 keys.

## Common Pitfalls
- **`load_dotenv()` finds `.env` relative to `main.py`'s directory**, not cwd. To test missing-API-key behavior, you must move/rename the `.env` file, not just unset the env var.
- **SLOW path takes ~20s per query** — set generous timeouts for subprocess tests.
- **REFLEX path** requires Sleep Phase to crystallize skills first. In a fresh test session with no stored memory, you won't see REFLEX routing. Don't treat this as a test failure.
- **Memory deduplication**: If you add the same episode twice, you'll see `[Memory] Deduplication triggered` — this is expected behavior, not an error.
- **No CI pipeline exists** for this repo. CodeRabbit is the only check (bot review, not blocking).

## Key Files
- `main.py` — CLI entry point (demo/interactive/benchmark modes)
- `nare/agent.py` — Main agent with 4-way routing
- `nare/memory.py` — FAISS-based memory system
- `nare/llm.py` — Gemini API integration
- `nare/sandbox.py` — Safe code execution
- `benchmarks/` — 5 benchmark suites
