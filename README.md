# NARE: Neural Amortized Reasoning Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**NARE (Neural Amortized Reasoning Engine)** is a production-grade reasoning system that combines verified synthesis with semantic memory for software engineering tasks.

## 🎯 Core Architecture

NARE implements a clean 4-component architecture:

```
nare/
├── core/              # Core reasoning engine
│   ├── agent.py      # Main agent orchestration
│   ├── router.py     # Query routing logic
│   ├── synthesis.py  # Verified Synthesis loop
│   └── evolution.py  # Library learning
├── reasoning/         # LLM and reasoning
│   ├── llm.py        # LLM interface
│   ├── critic.py     # Solution critic
│   └── oracle.py     # Oracle builders
├── memory/            # Memory system
│   ├── memory.py     # HNSW-based cache
│   └── metrics.py    # Performance tracking
├── execution/         # Code execution
│   └── sandbox.py    # Secure sandbox
└── tools/             # Utilities
    ├── repo_manager.py
    └── solve_context.py
```

### Key Components

**1. Verified Synthesis (VS)**
- Iterative code generation with test-based verification
- MDP with binary reward: generate → execute → feedback loop
- Converges to first oracle-passing attempt (max 5-6 attempts)

**2. Semantic Memory**
- HNSW vector cache for O(log N) retrieval
- Episodic memory with activation scores
- Automatic forgetting curve prevents bloat

**3. Library Learning**
- Discovers reusable patterns through SEARCH (not extraction)
- Compiles repeated patterns into executable skills
- Holdout validation ensures generalization

**4. Adaptive Routing**
- DIRECT: Zero-shot for simple queries
- ANALYTIC: Chain-of-thought reasoning
- ADAPTIVE: Delta reasoning from cached solutions
- REACTIVE: Execute compiled skills
- SYNTHESIS: Program synthesis with verification

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Anthropic API key

```bash
git clone https://github.com/starface77/Neuro-Adaptive-Reasoning-Engine
cd Neuro-Adaptive-Reasoning-Engine
pip install -r requirements.txt
```

### Configuration

Create `.env` file:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional (defaults shown)
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

### Basic Usage

```python
from nare import NAREProductionAgent, DEFAULT_CONFIG

# Initialize agent
agent = NAREProductionAgent(
    config=DEFAULT_CONFIG,
    persist_dir="./memory",
    embedding_dim=384
)

# Define oracle for verification
def oracle(query, answer):
    expected = "150"
    return (expected in answer, f"Expected {expected}")

# Solve with verification
query = "A train travels at 60 km/h for 2.5 hours. How far?"
result = agent.solve(query, oracle=oracle)

print(result["route_decision"])  # ANALYTIC or SYNTHESIS
print(result["final_answer"])     # 150
```

---

## 📊 What's Working (2026-05-01)

✅ **Verified Synthesis** - Iterative generation with oracle feedback  
✅ **FAIL_TO_PASS Oracle** - Direct test execution without patch application  
✅ **Semantic Memory** - HNSW cache with FAISS  
✅ **Library Learning** - Rule discovery through search  
✅ **Adaptive Routing** - 5 modes (DIRECT/ANALYTIC/ADAPTIVE/REACTIVE/SYNTHESIS)  
✅ **Temperature Control** - 0.1 for code precision  
✅ **Recursive File Search** - Finds files like rst.py across entire repo  

🚧 **In Progress**
- Dependency installation for test execution
- Semantic file indexing for better retrieval
- Cross-task pattern generalization

---

## 🔧 Recent Improvements (2026-05-01)

**Architecture Reorganization**
- Clean separation: core/ reasoning/ memory/ execution/ tools/
- All imports fixed and tested
- No hardcoded API keys (uses .env only)

**File Retrieval Enhancement**
- Strategy 0: Recursive search by filename (finds rst.py, qdp.py)
- Strategy 1: Explicit path extraction
- Strategy 2: Keyword-based git grep
- Strategy 3: Error message parsing

**Oracle Improvements**
- Uses FAIL_TO_PASS tests directly (no patch application)
- Graceful fallback when dependencies missing
- Returns None (unavailable) vs False (failed)

**Verified Synthesis**
- Handles None oracle (skips retry)
- Improved system prompt for SYNTHESIS mode
- Better regex parsing for File: format

---

## ⚠️ Limitations

**Oracle Dependence**
- Requires test oracle for verification
- Without oracle, falls back to best-of-N
- Test dependencies must be installed

**File Retrieval**
- Recursive search helps but not perfect
- May find wrong files if names are ambiguous
- Semantic indexing would improve accuracy

**Sandbox Security**
- Subprocess-based (not Docker)
- Python-only (no JS/Go/Rust)
- Do not run on untrusted infrastructure

---

## 📚 References

**Verified Synthesis**
- AlphaCode 2 (DeepMind, 2023)
- OpenAI o1 (OpenAI, 2024)
- Reflexion (Shinn et al., 2023)

**Library Learning**
- DreamCoder (Ellis et al., 2021)
- AlphaCode 2 DSL synthesis

**Memory Systems**
- MemoryBank (Zhong et al., 2024)
- HNSW (Malkov & Yashunin, 2018)

---

## 🤝 Contributing

Contributions welcome. Please ensure:
- Unit tests pass
- No regression on benchmarks
- Follow existing code style

## 📝 License

MIT License. See `LICENSE` for details.

---

**NARE = Verified Synthesis + Semantic Memory + Library Learning**

No pseudo-science. No biological metaphors. Just three proven components working together.
