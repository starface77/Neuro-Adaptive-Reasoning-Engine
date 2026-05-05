# NARE — Neural Amortized Reasoning Engine

[![License: MIT](https://img.shields.io/badge/license-Apache%20License%202.0-blue)](http://www.apache.org/licenses)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**NARE** is an AI reasoning engine that learns from experience. It caches solutions, compiles patterns into reusable skills, and gets faster over time — like a developer who remembers what worked before.

> **Status:** v0.2.0 — Production-ready core, evolving CLI. APIs may shift between minor releases.

---

## What makes NARE different?

Most AI coding assistants start from scratch every time. NARE **remembers**:

- **Semantic memory** — FAISS-backed cache of past solutions
- **Compiled skills** — Recurring patterns crystallize into executable code
- **Adaptive routing** — Cheap cached answers when possible, deep reasoning when needed
- **Verified synthesis** — Generate → test → critique → retry loop with oracle feedback

Think of it as an AI that builds its own library of solutions as it works.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Query                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │  Triage Agent  │  ← Classify intent (QUESTION/EXPLORE/EDIT)
            └────────┬───────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Adaptive Router     │  ← 5-tier decision tree
         └───────────┬───────────┘
                     │
        ┌────────────┼────────────┬────────────┬────────────┐
        │            │            │            │            │
        ▼            ▼            ▼            ▼            ▼
    DIRECT    COMPILED_SKILL   FAST       HYBRID       SLOW
    (chat)    (cached code)  (FAISS)  (FAISS+delta) (full LLM)
        │            │            │            │            │
        └────────────┴────────────┴────────────┴────────────┘
                                  │
                                  ▼
                          ┌───────────────┐
                          │ Memory System │  ← Episodes + Skills
                          └───────────────┘
```

### 5-Tier Routing

| Tier | When | Cost | Example |
|------|------|------|---------|
| **DIRECT** | Greetings, meta-questions | ~0 tokens | "привет", "what can you do?" |
| **COMPILED_SKILL** | Exact pattern match in skills | ~0 tokens | Recurring refactors, known fixes |
| **FAST** | Cached episode (similarity ≥ 0.85) | ~500 tokens | "fix auth bug" → cached solution |
| **HYBRID** | Cached + delta reasoning | ~2k tokens | Similar problem, different context |
| **SLOW** | Full reasoning + verification | ~10k+ tokens | Novel problems, complex edits |

**Key insight:** Most queries hit FAST or HYBRID after a few sessions. SLOW is expensive but teaches the system.

---

## Installation

```bash
git clone https://github.com/starface77/Neuro-Adaptive-Reasoning-Engine
cd Neuro-Adaptive-Reasoning-Engine
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- Anthropic API key (or compatible proxy)
- Optional: Local embeddings model

---

## Quick Start

### 1. Configure API

```bash
cp .env.example .env
# Edit .env:
ANTHROPIC_API_KEY=your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

**Using a proxy?** (e.g., local LLM gateway)
```bash
ANTHROPIC_BASE_URL=http://localhost:20128/v1
ANTHROPIC_MODEL=kr/claude-sonnet-4.5
```

### 2. Launch REPL

```bash
python -m nare.cli
```

```
◆ NARE  reasoning agent for software engineering
  NareCLI  /home/user/project
  Manual mode  ·  type /help for commands

> fix the login timeout bug
```

### 3. One-shot mode

```bash
python -m nare.cli "add type hints to utils.py"
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/status` | Session stats (tokens, memory, route distribution) |
| `/repo [path]` | Change working directory |
| `/files` | List files in context |
| `/read <path>` | Load file into context |
| `/clear` | Reset conversation |
| `/mode` | Cycle: Manual → Research → Autopilot |
| `/memory` | Inspect cached episodes and skills |
| `/diff` | Show uncommitted changes |
| `/commit [msg]` | Git commit with optional message |
| `/test` | Run project tests |
| `/bench <n>` | Run SWE-bench on n tasks |
| `/agent on\|off` | Toggle new agent loop (tool-calling) |
| `/exit` | Quit |

### Autonomy Modes

- **Manual** — Confirm every file write and shell command
- **Deep Research** — Auto-read files, confirm writes
- **Autopilot** — Full autonomy, only confirms destructive actions

---

## Programmatic API

```python
from nare import NAREProductionAgent, DEFAULT_CONFIG

agent = NAREProductionAgent(
    config=DEFAULT_CONFIG,
    persist_dir="./.nare_memory",
    embedding_dim=3072,
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

### Running SWE-bench

```bash
# Run on 30 tasks
python benchmarks/swe_bench_official.py --max-tasks 30

# Output: predictions.jsonl (official format)
```

---

## Limitations (Honest Section)

### What NARE does well:
✅ Repetitive refactors (gets faster over time)  
✅ Bug fixes with clear test cases  
✅ Code explanation and analysis  
✅ Incremental edits to existing code  

### What NARE struggles with:
❌ **Novel problems** — First attempt is slow (SLOW tier)  
❌ **Ambiguous requirements** — Needs clear oracles  
❌ **Large refactors** — Context window limits (working on it)  
❌ **Non-Python code** — Sandbox is Python-only  
❌ **Security** — Subprocess sandbox, not container-isolated  

### Known issues:
- File resolution can match wrong files (ambiguous names)
- Memory grows unbounded (need pruning strategy)
- No streaming UI for SLOW tier (shows spinner, then dumps result)
- Research agent incomplete (WebSearch integration TODO)

---

## Roadmap

**v0.3.0** (Next)
- [ ] Streaming UI for SLOW tier
- [ ] Memory pruning (LRU + activation decay)
- [ ] WebSearch integration in research agent
- [ ] Multi-file refactor support
- [ ] Docker sandbox (replace subprocess)

**v0.4.0**
- [ ] Multi-language support (JS, Go, Rust)
- [ ] Persistent task list (resume interrupted work)
- [ ] Skill marketplace (share compiled skills)
- [ ] Web UI (alternative to CLI)

---

## Contributing

PRs welcome! Focus areas:
- **Oracles** — New oracle types (linters, formatters, etc.)
- **Skills** — Pre-compiled skills for common tasks
- **Benchmarks** — More evaluation datasets
- **Docs** — Tutorials, examples, architecture deep-dives

```bash
# Run tests
pytest tests/

# Lint
ruff check nare/

# Format
ruff format nare/
```

---

## FAQ

**Q: How is this different from Cursor/Copilot/Aider?**  
A: NARE learns from experience. After solving a problem once, it caches the solution and gets faster. Most tools start from scratch every time.

**Q: Do I need a GPU?**  
A: No. Embeddings can run on CPU (slow) or via API (fast). LLM calls go to Anthropic API.

**Q: Can I use local LLMs?**  
A: Yes, via proxy. Set `ANTHROPIC_BASE_URL` to your local endpoint (e.g., Ollama, LM Studio).

**Q: Is my code sent to Anthropic?**  
A: Yes, if you use their API. Use a local proxy if you need privacy.

**Q: How much does it cost?**  
A: Depends on usage. FAST tier is ~free (cached). SLOW tier is ~$0.10-0.50 per complex task (Claude Sonnet 4).

**Q: Can I run this in production?**  
A: Core engine: yes. CLI: use at your own risk (subprocess sandbox is not production-grade).

---

## License

MIT — see [`LICENSE`](LICENSE)

---

## Credits

Built by github.com/starface77

Inspired by:
- [Voyager](https://github.com/MineDojo/Voyager) (skill library learning)
- [Reflexion](https://arxiv.org/abs/2303.11366) (self-critique loop)
- [MemGPT](https://github.com/cpacker/MemGPT) (memory management)

---

**Star this repo if you find it useful!** ⭐
