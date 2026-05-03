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
  <a href="https://github.com/starface77/Neuro-Adaptive-Reasoning-Engine/stargazers"><img src="https://img.shields.io/github/stars/starface77/Neuro-Adaptive-Reasoning-Engine?style=for-the-badge&color=yellow" alt="Stars"/></a>
  <a href="https://github.com/starface77/Neuro-Adaptive-Reasoning-Engine/issues"><img src="https://img.shields.io/github/issues/starface77/Neuro-Adaptive-Reasoning-Engine?style=for-the-badge" alt="Issues"/></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#features">Features</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#russian">Русский</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**VARE** (Verified Amortized Reasoning Engine) is a cognitive architecture that combines LLM-based code synthesis with formal verification and episodic memory. It processes queries through three independent components:

1. **M_cache** — HNSW-backed episodic memory for instant retrieval of verified solutions
2. **G_θ** — Fixed-weight LLM generator (Gemma-3-27B) with iterative self-refinement
3. **V_sandbox** — Formal verifier (Python AST + subprocess isolation)

> **Key idea:** Familiar queries are served instantly from verified cache (FAST route). Novel queries go through a verified synthesis loop: generate → sandbox verify → refine until correct (VERIFIED_RETRY). Successfully verified solutions are cached for future use. Background Library Learning clusters similar solutions into reusable compiled skills.

---

<a name="architecture"></a>
## Architecture

```
                              ┌─────────────────────────────┐
                              │        Input Query          │
                              └──────────┬──────────────────┘
                                         │
                              ┌──────────▼──────────────────┐
                              │     Semantic Embedding       │
                              │   (gemini-embedding-001)     │
                              └──────────┬──────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────┐
                    │           2-WAY ROUTER                       │
                    │                                              │
                    │  ρ(x) = max cos_sim(E(x), M_cache)          │
                    │                                              │
                    │  ┌──────────┐  ρ≥τ_fast  ┌──────────────┐  │
                    │  │  FAST    ├────────────►│ Return cached │  │
                    │  │  ROUTE   │             │ answer / run  │  │
                    │  └──────────┘             │ compiled skill│  │
                    │       │ ρ<τ_fast          └──────────────┘  │
                    │  ┌────▼──────────┐        ┌──────────────┐  │
                    │  │  VERIFIED     ├───────►│ Generate →    │  │
                    │  │  RETRY        │        │ Verify →      │  │
                    │  │  (System 2)   │        │ Refine loop   │  │
                    │  └───────────────┘        └──────────────┘  │
                    └─────────────────────────────────────────────┘
                                         │
                              ┌──────────▼──────────────────┐
                              │    Store verified result     │
                              │    in M_cache (HNSW)         │
                              └──────────┬──────────────────┘
                                         │
                              ┌──────────▼──────────────────┐
                              │   Background: Library        │
                              │   Learning (cluster →        │
                              │   abstract → verify →        │
                              │   COMPILED_SKILL)            │
                              └──────────────────────────────┘
```

### Components

| Component | Role | Complexity |
|-----------|------|------------|
| **M_cache** | HNSW vector index of verified episodes + compiled skills | O(log N) search |
| **G_θ** | Fixed-weight LLM (Gemma-3-27B), generates candidates | Per-query cost |
| **V_sandbox** | AST validation + subprocess execution, binary R(y)∈{0,1} | Deterministic |

### Routing

| Route | Condition | Cost |
|-------|-----------|------|
| **FAST** | `max_sim >= τ_fast` | 0 tokens, O(log N) |
| **VERIFIED_RETRY** | `max_sim < τ_fast` | N × LLM calls (max H retries) |

### Verified Synthesis MDP

The synthesis loop is formalized as an MDP:
- **State** S: query x + error history E₁..ₖ₋₁
- **Action** A: generate candidate y_k ~ G_θ(y | x, E₁..ₖ₋₁)
- **Transition** T: execute y_k in V_sandbox
- **Reward** R: R(y_k) = 1 if passes, else 0
- **Horizon** H: max_retries

### Library Learning (Background)

Periodically clusters similar verified episodes, asks LLM to abstract a reusable function, verifies it against all cluster tasks, and stores as `COMPILED_SKILL` for instant FAST-route execution.

---

<a name="features"></a>
## Features

- **Verified Code Synthesis** — Every solution is sandbox-verified before caching
- **HNSW Episodic Memory** — O(log N) approximate nearest neighbour search
- **Activation-Based Forgetting** — Ebbinghaus-inspired exponential decay
- **Library Learning** — Automatic abstraction of recurring patterns into skills
- **AST Sandbox** — Secure execution with whitelist-based validation
- **Self-Refinement** — Error traces fed back to LLM for iterative correction
- **Oracle Protocol** — Pluggable ground-truth verification (SymPy, pytest, custom)

---

<a name="quickstart"></a>
## Quickstart

```bash
# 1. Clone
git clone https://github.com/starface77/Neuro-Adaptive-Reasoning-Engine.git
cd Neuro-Adaptive-Reasoning-Engine

# 2. Install
pip install -e .

# 3. Configure
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Run demo
python main.py demo

# 5. Interactive mode
python main.py interactive

# 6. Single query
python main.py --query "Find the next number: 3, 6, 9, 12, 15"
```

### Benchmarks

```bash
# Quick sanity check (6 tasks, ~5 min)
python main.py benchmark --benchmark quick

# Full evaluation (24 tasks, ~25 min)
python main.py benchmark --benchmark full
```

---

<a name="benchmarks"></a>
## Metrics

VARE tracks three composite metrics (per MemoryBench):

| Metric | Description |
|--------|-------------|
| **Quality** | Accuracy — fraction of verified solutions |
| **Latency** | Average response time (should decrease as memory grows) |
| **Tokens** | LLM token consumption per query (should decrease with FAST hits) |

---

<a name="russian"></a>
## Русский

**VARE** (Verified Amortized Reasoning Engine) — когнитивная архитектура, объединяющая синтез кода через LLM с формальной верификацией и эпизодической памятью.

Три компонента:
- **M_cache** — HNSW-граф эпизодической памяти для мгновенного извлечения проверенных решений
- **G_θ** — LLM-генератор с фиксированными весами и итеративным самоулучшением
- **V_sandbox** — формальный верификатор (AST + изолированный subprocess)

Два маршрута:
- **FAST** — при высоком сходстве (≥ τ_fast) мгновенный возврат из кэша
- **VERIFIED_RETRY** — цикл генерации → верификации → уточнения до корректного результата

Фоновый процесс Library Learning кластеризует решённые задачи, абстрагирует в функции, верифицирует и сохраняет как COMPILED_SKILL.

---

<a name="citation"></a>
## Citation

```bibtex
@software{vare2025,
  title   = {VARE: Verified Amortized Reasoning Engine},
  author  = {starface77},
  year    = {2025},
  url     = {https://github.com/starface77/Neuro-Adaptive-Reasoning-Engine}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
