<div align="center">

# NARE CLI

**Neural Amortized Reasoning Engine**  
AI coding assistant that learns from experience and gets faster over time

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/narecli)](https://pypi.org/project/narecli/)

[Quick Start](#-quick-start) • [Features](#-features) • [Architecture](#-architecture) • [Benchmarks](#-benchmarks) • [Docker](#-docker)

</div>

---

## Why NARE?

Traditional AI assistants forget everything after each conversation. **NARE remembers, learns, and accelerates.**

| Traditional AI | NARE CLI |
|----------------|----------|
| ❌ Forgets solutions | ✅ Persistent memory |
| 🐌 Same speed always | ⚡ Gets faster with use |
| ❌ No validation | ✅ Verified synthesis |
| ❌ Static behavior | ✅ Compiles patterns into skills |

**The result:** Common tasks become instant. Complex tasks get verified. Your assistant evolves with your codebase.

---

## 🚀 Quick Start

### Installation

```bash
pip install narecli
```

### Setup

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"

# Start interactive mode
nare
```

### One-Shot Mode

```bash
nare "add type hints to utils.py"
```

### Example Session

```
> fix the bug in auth.py
  ● Intent: edit
  ◌ find_function(function_name='authenticate', file_path='auth.py')
  ● Found function 'authenticate' at line 45
  ◌ apply_hunks(...)
  ● Applied 1 hunk to auth.py
  
Fixed authentication bug in auth.py
  3.2k tokens  ·  4.1s
```

Next time you ask a similar question:

```
> fix the bug in payment.py
  ● Route: FAST (memory hit)
  
Fixed payment validation bug in payment.py
  0 tokens  ·  0.02s
```

---

## ✨ Features

### 🧠 Semantic Memory
- **Episodic storage**: Every solved task is remembered
- **FAISS-powered retrieval**: Sub-100ms similarity search
- **Automatic deduplication**: No redundant storage

### ⚡ 5-Tier Routing

```
User Query → Router → [FAST|REFLEX|COMPILED|HYBRID|SLOW]
```

1. **FAST** (0 tokens, <100ms): Exact memory match
2. **REFLEX** (0 tokens, <100ms): Pre-compiled skills
3. **COMPILED_SKILL** (0 tokens, <500ms): Pattern matching
4. **HYBRID** (minimal tokens): Memory + small edits
5. **SLOW** (full tokens): Verified synthesis with LLM

### 🔬 Verified Synthesis
- **Oracle-based validation**: Code is tested before application
- **Automatic repair**: Failed attempts trigger refinement
- **Confidence scoring**: Critic evaluates solution quality

### 📚 Library Learning
- **Pattern compilation**: Frequent tasks become instant skills
- **Background evolution**: Continuous optimization
- **Skill quarantine**: Invalid patterns are isolated

### 🛠️ Developer Tools
- **Token-efficient editing**: `find_function` → `apply_hunks` workflow
- **Batch operations**: Multi-file edits in one call
- **Git integration**: Safe branching and PR creation
- **Web search**: Real-time information retrieval

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  NAREProductionAgent                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ MemorySystem │  │ReasoningRouter│  │EvolutionEngine│ │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤ │
│  │ Episodes     │  │ Intent       │  │ Skill        │ │
│  │ Skills       │  │ Classifier   │  │ Compilation  │ │
│  │ FAISS Index  │  │ 5-Tier Route │  │ Background   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │VerifiedSynth │  │    Critic    │  │MetricsTracker│ │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤ │
│  │ Oracle Loop  │  │ Confidence   │  │ Performance  │ │
│  │ Auto-Repair  │  │ Scoring      │  │ Analytics    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Core Components

- **`nare/core/agent.py`**: Main facade coordinating all subsystems
- **`nare/memory/engine.py`**: Persistent storage with FAISS indexing
- **`nare/core/routing/router.py`**: 5-tier decision engine
- **`nare/core/evolution/engine.py`**: Background learning and skill compilation
- **`nare/core/synthesis/loop.py`**: Verified synthesis with oracle validation
- **`nare/cli/app.py`**: Rich terminal interface with live rendering

---

## 📊 Benchmarks

### ARC Challenge (Abstract Reasoning)

```python
# Run ARC benchmark
python benchmarks/nare_arc_full.py
```

**Results**: See `benchmarks/nare_arc_results.json`

### SWE-bench (Real-World Software Engineering)

```python
# Run SWE-bench pilot
python benchmarks/nare_swe_bench.py
```

**Dataset**: 50 real GitHub issues from popular Python repositories

### Performance Metrics

| Metric | Cold Start | After 100 Tasks |
|--------|------------|----------------|
| Avg Response Time | 4.2s | 0.8s |
| Token Usage | 100% | 23% |
| Memory Hit Rate | 0% | 67% |
| Skill Compilation | 0 | 34 |

---

## 🐳 Docker

### Quick Start

```bash
docker-compose up -d
docker exec -it nare-cli nare
```

### Build from Source

```bash
docker build -t nare-cli .
docker run -it \
  -e ANTHROPIC_API_KEY="your-key" \
  -v $(pwd):/workspace \
  nare-cli
```

---

## 🔧 Configuration

Create `.env` in your project root:

```bash
ANTHROPIC_API_KEY=your-key-here
NARE_MEMORY_DIR=~/.nare/memory  # Optional: custom memory location
NARE_LOG_LEVEL=INFO             # Optional: DEBUG, INFO, WARNING, ERROR
```

### Advanced Configuration

```python
from nare.config import NareConfig
from nare.core.agent import NAREProductionAgent

config = NareConfig(
    memory_threshold=0.85,        # Similarity threshold for memory hits
    max_synthesis_attempts=3,     # Max repair attempts in verified synthesis
    skill_compilation_min_uses=5, # Min uses before pattern compilation
    enable_background_evolution=True,
)

agent = NAREProductionAgent(config=config)
```

---

## 📖 Usage Examples

### Interactive Mode

```bash
nare
```

```
NARE CLI v0.2.4
Type 'help' for commands, 'exit' to quit

> add logging to database.py
> refactor UserService to use dependency injection
> write tests for the payment module
```

### Programmatic API

```python
import asyncio
from nare.core.agent import NAREProductionAgent

async def main():
    agent = NAREProductionAgent()
    
    result = await agent.solve(
        query="fix the bug in auth.py",
        working_dir="./my-project",
    )
    
    print(result["answer"])
    print(f"Route: {result['route']}")
    print(f"Tokens: {result['tokens_used']}")

asyncio.run(main())
```

### Compile Skills from History

```python
from nare.core.agent import NAREProductionAgent

agent = NAREProductionAgent()
agent.evolution.compile_skills(min_uses=3, max_skills=50)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_agent_loop.py

# Run with coverage
pytest --cov=nare tests/
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `pytest tests/`
5. **Commit**: `git commit -m 'Add amazing feature'`
6. **Push**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Setup

```bash
git clone https://github.com/Nare-Labs/NARE-CLI.git
cd NARE-CLI
pip install -e ".[embeddings]"
pytest tests/
```

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings for public APIs
- Keep functions focused and testable

---

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Anthropic Claude** for reasoning capabilities
- **FAISS** for efficient similarity search
- **Rich** for beautiful terminal rendering
- **SWE-bench** for real-world evaluation dataset

---

## 📬 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/Nare-Labs/NARE-CLI/issues)
- **Discussions**: [Join the community](https://github.com/Nare-Labs/NARE-CLI/discussions)

---

<div align="center">

**Built with ❤️ by the NARE Labs team**

[⭐ Star us on GitHub](https://github.com/Nare-Labs/NARE-CLI) • [📦 PyPI Package](https://pypi.org/project/narecli/)

</div>
