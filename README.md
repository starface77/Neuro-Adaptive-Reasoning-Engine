# NARE

<p align="center">
  <img src="assets/image.png" alt="NARE CLI Demo" width="800"/>
</p>

> **Neural Amortized Reasoning Engine** -- the AI coding agent that learns from experience.

NARE remembers every solved task, compiles repeated patterns into instant skills, and routes queries through a 4-tier reasoning pipeline. The result: common tasks become zero-cost, complex tasks get verified, and your assistant gets faster with every interaction.

---

## Quick Start

```bash
pip install narecli
export ANTHROPIC_API_KEY="your-key"
nare
```

No configuration files needed. Memory persists automatically in `.nare_memory/`.

---

## How It Works

### 4-Tier Routing

Every query passes through the reasoning router:

```
Query --> Router --> [FAST | REFLEX | HYBRID | SLOW]
```

| Tier | Tokens | Latency | Description |
|------|--------|---------|-------------|
| **FAST** | 0 | ~100ms | Exact episodic memory match (cosine similarity >= tau) |
| **REFLEX** | 0 | ~200ms | Pre-compiled skill or semantic rule match |
| **HYBRID** | Minimal | ~2s | Delta reasoning with memory context |
| **SLOW** | Full | ~5-15s | Verified synthesis with best-of-N generation |

As memory grows, more queries resolve through FAST/REFLEX, reducing cost over time.

### Amortized Reasoning

NARE tracks amortization in real time:

```
alpha_t = amortized_queries / total_queries
C_t = (1 - alpha_t) * C_llm + alpha_t * C_mem
```

Where `C_llm = 100` (full LLM cost) and `C_mem = 1` (memory retrieval cost). As alpha_t approaches 1.0, your effective cost approaches zero. View live metrics with `/metrics`.

### Episodic Memory

- **FAISS HNSW index** for sub-100ms similarity search
- Every successful task stored with embedding + full reasoning trace
- Trust coefficients (tau) decay over time for stale episodes
- Thread-safe persistence with automatic flush

### Crystallization (Library Learning)

When 3+ similar episodes accumulate:

1. **DBSCAN clustering** groups related episodes (eps=0.5 on L2-normalized vectors ~ cosine sim >= 0.875)
2. **LLM generates N candidate rules** with trigger/execute functions
3. **Sandbox compilation** validates each candidate
4. **Holdout evaluation** scores generalization (threshold: 0.6)
5. Best rule becomes a **compiled skill** (zero LLM cost on future matches)

Trigger manually with `/skills compile` or let the evolution engine run automatically.

### Verified Synthesis

When NARE generates new code (SLOW route):

1. Generate candidate solution
2. Execute in sandboxed environment
3. Validate with oracle (test runner, assertions, heuristics)
4. If failed: feed error back, retry with refinement prompt
5. Repeat up to N attempts (default: 8)

Only verified solutions are returned and cached.

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/agent` | Toggle autonomous agent mode (multi-step execution) |
| `/skills` | List compiled skills |
| `/skills compile` | Trigger manual crystallization |
| `/memory` | Show memory statistics |
| `/metrics` | Show routing metrics + amortization stats |
| `/mode` | Switch between code/architect/ask modes |
| `/autonomy` | Set autonomy level (supervised/assisted/autonomous) |
| `/status` | Show agent status |
| `/read <file>` | Add file to context |
| `/test <file>` | Run tests on file |
| `/diff` | Show git diff |
| `/undo` | Undo last file change |
| `/help` | Show all commands |

### Modes

- **Tab** to cycle between Code / Architect / Ask modes
- **Ctrl+L** to clear screen
- **Ctrl+D** to exit

---

## Architecture

```
+----------------------------------------------+
|          NAREProductionAgent                  |
+----------------------------------------------+
|                                              |
|  Memory System       Reasoning Router        |
|  +- Episodes         +- Intent Classifier    |
|  +- Skills           +- 4-Tier Routing       |
|  +- FAISS Index      +- Confidence Scoring   |
|  +- Trust Decay      +- Amortization Tracker |
|                                              |
|  Evolution Engine    Verified Synthesis       |
|  +- DBSCAN Cluster   +- Oracle Loop          |
|  +- Rule Discovery   +- Auto-Repair          |
|  +- Skill Compile    +- Critic Scoring       |
|  +- Quarantine       +- Sandbox Execution    |
+----------------------------------------------+
```

**Core modules:**

| Module | Path | Purpose |
|--------|------|---------|
| Agent | `nare/core/agent.py` | Main orchestrator, amortization stats |
| Memory | `nare/memory/engine.py` | FAISS index, episode/skill persistence |
| Router | `nare/core/routing/router.py` | 4-tier routing, intent classification |
| Evolution | `nare/core/evolution/engine.py` | Crystallization pipeline |
| Learning | `nare/core/evolution/learning.py` | Rule discovery via search |
| Synthesis | `nare/core/synthesis/engine.py` | Verified synthesis loop |
| CLI | `nare/cli/` | REPL, commands, display |

---

## Configuration

### Environment Variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."       # Required
export NARE_MEMORY_DIR="~/.nare/memory"     # Optional: custom memory location
export NARE_LOG_LEVEL="INFO"                # Optional: DEBUG, INFO, WARNING
```

### Programmatic API

```python
import asyncio
from nare.core.agent import NAREProductionAgent

async def main():
    agent = NAREProductionAgent()

    result = await agent.solve(
        query="Fix the bug in auth.py",
        working_dir="./my-project"
    )

    print(result["final_answer"])
    print(f"Route: {result['route_decision']}")
    print(f"Amortization: {result.get('amortization_ratio', 0):.1%}")

asyncio.run(main())
```

---

## Development

```bash
git clone https://github.com/Nare-Labs/NARE-CLI
cd NARE-CLI
pip install -e ".[embeddings]"
pytest tests/
```

---

## Honest Assessment

| Feature | Status | Notes |
|---------|--------|-------|
| Episodic Memory | Working | FAISS HNSW + thread-safe persistence |
| 4-Tier Routing | Working | FAST/REFLEX/HYBRID/SLOW with intent classification |
| Amortization Tracking | Working | Empirical + theoretical alpha_t, blended cost |
| Crystallization | Working | DBSCAN + LLM rule discovery + sandbox validation |
| Verified Synthesis | Working | Multi-attempt with oracle feedback |
| Autonomous Agent | Working | Multi-step tool execution with budget control |
| "Formal Verification" | Misleading | Runtime testing, not theorem proving |
| "Zero Latency" | Approximate | ~100ms for FAST route, not truly zero |

---

## License

Apache License 2.0 - see [LICENSE](LICENSE)

---

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

---

**Built by [Nare Labs](https://github.com/Nare-Labs)**
