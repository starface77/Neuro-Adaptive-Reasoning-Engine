<p align="center">
  <img src="nare_banner.png" alt="VARE Banner" width="800"/>
</p>

<h1 align="center">VARE — Verified Amortized Reasoning Engine</h1>

<p align="center">
  <em>A Verified Code Synthesis Architecture with Episodic Memory<br/>for Amortized LLM Reasoning via Formal Verification</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT License"/></a>
  <a href="https://github.com/starface77/Neuro-Adaptive-Reasoning-Engine/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge" alt="PRs Welcome"/></a>
</p>

---

## Overview

**VARE** (Verified Amortized Reasoning Engine) is a cognitive architecture that combines LLM-based code synthesis with formal verification and episodic memory.

Three independent components:
1. **M_cache** — HNSW-backed episodic memory for instant retrieval of verified solutions
2. **G_θ** — Fixed-weight LLM generator with iterative self-refinement
3. **V_sandbox** — Formal verifier (Python AST + subprocess isolation)

> Familiar queries are served instantly from verified cache (FAST route). Novel queries go through a verified synthesis loop: generate -> sandbox verify -> refine (VERIFIED_RETRY). Background Library Learning clusters similar solutions into reusable compiled skills.

---

## Architecture

```
                    ┌────────────────────────────────────────────┐
                    │           2-WAY ROUTER                      │
                    │  ρ(x) = max cos_sim(E(x), M_cache)         │
                    │                                             │
                    │  ρ ≥ τ_fast  ──►  FAST (return cached)      │
                    │  ρ < τ_fast  ──►  VERIFIED_RETRY (synth)    │
                    └────────────────────────────────────────────┘
```

| Route | Condition | Cost |
|-------|-----------|------|
| **FAST** | `max_sim >= τ_fast` | 0 tokens, O(log N) |
| **VERIFIED_RETRY** | `max_sim < τ_fast` | N × LLM calls |

### Verified Synthesis MDP
- **State**: query + error history
- **Action**: generate candidate via G_θ
- **Transition**: execute in V_sandbox
- **Reward**: R(y) ∈ {0, 1}
- **Horizon**: max_retries

---

## Quickstart

```bash
git clone https://github.com/starface77/Neuro-Adaptive-Reasoning-Engine.git
cd Neuro-Adaptive-Reasoning-Engine
pip install -e .
cp .env.example .env  # add GEMINI_API_KEY
python main.py demo
```

---

## Project Structure

```
nare/
├── config.py              # VareConfig (routing, synthesis, memory, library)
├── core/
│   ├── agent.py           # VareAgent (main orchestrator)
│   ├── evolution.py       # EvolutionEngine (background compilation)
│   ├── library_learning.py
│   ├── router.py          # ReasoningRouter (legacy)
│   └── synthesis.py       # Verified synthesis utilities
├── memory/
│   ├── memory.py          # MemorySystem (HNSW + activation decay)
│   └── metrics.py         # MetricsTracker (Quality, Latency, Tokens)
├── reasoning/
│   ├── llm.py             # LLM API (Gemini/Anthropic)
│   ├── critic.py          # Solution critic
│   └── oracle.py          # Pluggable oracles
├── execution/
│   ├── sandbox.py         # AST sandbox
│   └── sandbox_subprocess.py
└── tools/
    ├── domain_detector.py
    ├── path_validator.py
    └── solve_context.py
```

## License

MIT
