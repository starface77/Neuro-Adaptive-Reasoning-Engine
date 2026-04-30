# VARE: Verified Amortized Reasoning Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**VARE (Verified Amortized Reasoning Engine)** is a production-grade reasoning system that combines three independently verified components:

1. **Verified Synthesis (VS)** - Iterative code generation with formal verification
2. **Episodic Memory** - HNSW-based caching for O(log N) retrieval
3. **Library Learning** - Automatic compilation of repeated patterns into reusable skills

By combining execution-based verification with semantic memory, VARE amortizes expensive reasoning: solve once with formal guarantees, reuse instantly for similar queries.

---

## 🎯 Core Architecture

### Three-Tier Routing

VARE routes queries through a hierarchical pipeline optimized for both correctness and efficiency:

**Layer -1: COMPILED_SKILL** (O(1))
- Pre-compiled functions for known patterns
- Zero LLM calls, instant execution
- Activated when pattern similarity ≥ 0.90

**Layer 0: FAST** (O(log N))
- HNSW vector cache of verified solutions
- Retrieves cached answers via cosine similarity
- Activated when similarity ≥ τ_fast (0.75)

**Layer 1: SLOW** (O(N))
- Verified Synthesis: iterative generation + formal verification
- Uses sandbox execution with oracle validation
- Saves successful solutions to memory

### Components

**M_cache (Memory System)**
- HNSW vector store with FAISS (O(log N) retrieval)
- Episodic memory with Ebbinghaus forgetting curve
- Compiled skills library for pattern reuse
- Thread-safe with RLock protection

**G_θ (LLM Generator)**
- Fixed-weight language model (Claude Sonnet 4.5 / Gemini)
- Generates code solutions with self-refinement
- No weight updates - purely inference

**V_sandbox (Formal Verifier)**
- Python AST + subprocess isolation
- Deterministic binary reward: R(y) ∈ {0,1}
- Prevents self-judging bias

### Memory Management

**Episodic Memory**
- Stores verified solutions with activation scores
- Each episode tracks: query, solution, score, embedding, timestamp
- Activation score boosted on access, decayed over time

**Forgetting Curve**
- Exponential decay: `s_i * exp(-Δt / S)`
- Episodes with `activation_score < 0.1` are pruned
- Prevents unbounded memory growth

**Library Learning (MVP)**
- Patterns used 3+ times → compiled as COMPILED_SKILL
- Extracts Python code blocks from solutions
- Stores with trigger embedding for fast retrieval
- Zero-token execution for compiled patterns

---

## 📊 MemoryBench Metrics

VARE tracks three key deltas (Δ) against baseline LLM:

**Quality (Δ)**: Accuracy improvement
- VARE accuracy vs baseline accuracy
- Measures correctness gains from verification

**Latency (Δ)**: Response time change
- % change in average response time
- Negative = faster (amortization working)

**Tokens (Δ)**: Token consumption change
- % change in average tokens per query
- Negative = cheaper (memory reuse working)

**Amortization Rate**: % queries served by O(1) paths
- FAST + REFLEX + COMPILED_SKILL routes
- Target: 30-40% after warm-up

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- API Key: Google Gemini or Anthropic Claude

```bash
git clone https://github.com/your-username/vare.git
cd vare
pip install -r requirements.txt
```

### Configuration
```bash
cp .env.example .env
# Add your API key to .env
```

### Basic Usage

```python
from nare.config import DEFAULT_CONFIG
from nare.agent import NAREProductionAgent

# Initialize agent with 384-dim embeddings (sentence-transformers)
agent = NAREProductionAgent(
    config=DEFAULT_CONFIG,
    persist_dir="./memory",
    embedding_dim=384
)

# Define oracle for verification
def oracle(query, answer):
    expected = "150"
    return (expected in answer, f"Expected {expected}")

# First query: SLOW path (Verified Synthesis)
query = "A train travels at 60 km/h for 2.5 hours. How far?"
result1 = agent.solve(query, oracle=oracle)
print(result1["route_decision"])  # Output: SLOW
print(result1["final_answer"])     # Output: 150

# Second query: FAST path (cached)
result2 = agent.solve(query, oracle=oracle)
print(result2["route_decision"])  # Output: FAST
print(result2["final_answer"])     # Output: 150 (instant)

# After 3+ similar queries: COMPILED_SKILL
result3 = agent.solve(query, oracle=oracle)
print(result3["route_decision"])  # Output: COMPILED_SKILL
```

### Running ARC-AGI Benchmark

```bash
# Run 10 tasks with Anthropic API
python benchmarks/nare_arc_full.py \
  --dataset data/arc-agi-2/training \
  --num-tasks 10 \
  --persist-dir memory_arc \
  --output results.json

# Check results
python -c "
import json
with open('results.json') as f:
    data = json.load(f)
    print(f'Accuracy: {data[\"accuracy\"]:.1%}')
    print(f'Amortization: {data[\"amortization_pct\"]:.1%}')
    print(f'Routing: {data[\"routing\"]}')
"
```

---

## 🧪 What's Working (2026-04-30)

✅ **Verified Synthesis** - Iterative code generation with oracle validation (8 attempts)  
✅ **Episodic Memory** - HNSW cache with O(log N) retrieval  
✅ **Forgetting Curve** - Exponential decay prevents memory bloat  
✅ **MemoryBench Metrics** - Quality/Latency/Tokens deltas vs baseline  
✅ **Library Learning MVP** - Simple pattern compilation (3+ uses → skill)

🚧 **In Progress**
- AST clustering for cross-task generalization
- Typed parameters for compiled skills
- Baseline comparison automation

📋 **Roadmap**
- DreamCoder-style DSL synthesis
- Multi-language sandbox (JS, Go, Rust)
- Distributed memory with Redis backend

---

## ❌ What We Removed (Honesty)

The following components were removed from the original NARE implementation:

- **Free Energy Principle (FEP)** - Pseudo-scientific decoration, no measurable impact
- **Tree-of-Thoughts (ToT)** - Terminological substitution for best-of-N sampling
- **Immune System** - Biological metaphor without formal grounding
- **Neural Memory** - Misleading name for standard vector store
- **HYBRID/REFLEX routes** - Kept for backward compatibility, but theory focuses on FAST/SLOW

VARE focuses on the **three components with empirical validation**:
1. Verified Synthesis (AlphaCode 2, OpenAI o1)
2. Episodic Memory (MemoryBank, Anthropic MCP)
3. Library Learning (DreamCoder DSL)

---

## ⚠️ Limitations

**Oracle Dependence**
- VARE requires oracle (ground truth) for verification
- Without oracle, falls back to best-of-N (less reliable)
- Not suitable for open-ended generation tasks

**Semantic Generalization**
- High similarity threshold (0.75-0.90) prevents false positives
- May miss structurally similar but numerically different problems
- Library Learning MVP doesn't generalize across tasks yet

**Sandbox Security**
- Current sandbox is subprocess-based (not Docker)
- Do not run on untrusted infrastructure
- Python-only (no JS/Go/Rust support yet)

**Memory Growth**
- Forgetting curve prevents unbounded growth
- But doesn't handle adversarial memory poisoning
- No distributed memory support yet

---

## 📚 References

**Verified Synthesis**
- AlphaCode 2 (DeepMind, 2023)
- OpenAI o1 (OpenAI, 2024)
- Reflexion (Shinn et al., 2023)

**Episodic Memory**
- MemoryBank (Zhong et al., 2024)
- Anthropic MCP (Anthropic, 2024)
- HNSW (Malkov & Yashunin, 2018)

**Library Learning**
- DreamCoder (Ellis et al., 2021)
- AlphaCode 2 DSL synthesis (DeepMind, 2023)

**Forgetting Curve**
- Ebbinghaus (1885) - Original forgetting curve
- ACT-R (Anderson, 1993) - Activation-based memory

---

## 🤝 Contributing

Contributions welcome. Please ensure:
- Unit tests pass (`pytest tests/`)
- No regression on ARC-AGI benchmark
- Code follows existing style (no reformatting)

## 📝 License

MIT License. See `LICENSE` for details.

---

## 🔬 Theory

For the formal mathematical framework, see `docs/theory.md`.

**TL;DR**: VARE = Verified Synthesis + Episodic Memory + Library Learning

No FEP. No ToT. No biological metaphors. Just three independently verified components working together.
