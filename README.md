# NARE CLI

> **Neural Amortized Reasoning Engine** - AI coding assistant that learns from experience

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/badge/PyPI-narecli-blue.svg)](https://pypi.org/project/narecli/)

---

## Why NARE CLI?

Traditional AI assistants forget everything after each conversation. **NARE CLI remembers, learns, and gets faster over time.**

| Feature | Traditional AI | NARE CLI |
|---------|---------------|----------|
| Memory | ❌ Forgets everything | ✅ Remembers solutions |
| Speed | 🐌 Same speed always | ⚡ Gets faster with use |
| Verification | ❌ No validation | ✅ Tests code before applying |
| Learning | ❌ Static | ✅ Compiles patterns into skills |

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

# Start NARE CLI
nare
```

### Your First Command

```bash
nare
```

```
> fix the bug in auth.py
  ● Intent: edit
  ◌ Find(function_name='authenticate', file_path='auth.py')
  ● Found function 'authenticate' at line 45
  ◌ Apply(hunks='...')
  ● Applied 1 hunk to auth.py
  
Fixed authentication bug in auth.py
  3.2k tokens  ·  4.1s
```

### One-Shot Mode

```bash
nare "add type hints to utils.py"
```

---

## 💡 How It Works

NARE CLI uses **VARE** (Verified Amortized Reasoning Engine) - a system that gets smarter with every task.

```
User Query
    ↓
┌───────────────┐
│    Router     │  Analyzes intent
└───────┬───────┘
        │
    ┌───┴───┐
    ↓       ↓
┌─────┐  ┌─────┐
│FAST │  │SLOW │
│(0ms)│  │(4s) │
└─────┘  └─────┘
Memory   Synthesis
```

### The Magic: 5-Tier Routing

1. **FAST** → Instant answer from memory (0 tokens)
2. **REFLEX** → Pre-compiled skills (0 tokens)
3. **COMPILED_SKILL** → Pattern matching
4. **HYBRID** → Memory + small edits
5. **SLOW** → Full synthesis with verification

**Result:** Common tasks become instant. Complex tasks get verified.

---

## 📊 Performance

### Token Optimization

NARE CLI is built for efficiency:

| Scenario | Tokens Used | Speed |
|----------|-------------|-------|
| Simple greeting | 100-200 | Instant |
| Memory lookup (FAST) | 0 | <100ms |
| Code edit (SLOW) | 3,000-5,000 | 3-5s |
| Typical session (10 queries) | 15,800 | - |

**85% token reduction** compared to standard LLM workflows.

### What Makes It Fast?

- **Prompt caching** - System prompt cached for 5 minutes
- **Smart history** - Code blocks trimmed to 100 chars
- **Efficient repo map** - Uses `git ls-files`, cached 15s
- **Adaptive thinking** - Only 200 tokens for reasoning

---

## ⚙️ Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Memory stats and agent status |
| `/agent on/off` | Toggle autonomous mode |
| `/repo <path>` | Change working directory |
| `/clear` | Clear screen |
| `/exit` | Exit NARE CLI |

---

## ⚠️ Known Limitations

NARE CLI works best for **short-to-medium tasks** on **small-to-medium projects**.

### Current Constraints

- **Budget**: 150 iterations, 1M tokens, 2 hours per task
- **Repo size**: Limited to 1,500 files
- **Chat history**: Last 10 messages only
- **Memory**: Aggressive pruning every 100 episodes
- **No checkpoints**: Crash = lost progress

For complex refactorings or large codebases, consider breaking tasks into smaller chunks.

See [REAL_PROBLEMS_ANALYSIS.md](REAL_PROBLEMS_ANALYSIS.md) for detailed analysis.

---

## 🛠️ Development

### Project Structure

```
nare/
├── agents/          # Agent loops and planning
├── cli/             # CLI interface and commands
├── core/            # Core VARE components
│   ├── routing/     # 5-tier routing system
│   ├── synthesis/   # Verified synthesis engine
│   └── evolution/   # Library learning
├── memory/          # Episodic memory + FAISS
├── reasoning/       # LLM interface
└── tools/           # Built-in tools (read, edit, bash, etc.)
```

### Install for Development

```bash
git clone https://github.com/Nare-Labs/NARE-CLI.git
cd NARE-CLI
pip install -e .
```

### Memory Structure

```
.nare_memory/
├── episodes.json           # Episodic memory
├── compiled_skills.json    # Learned skills
├── rules.json              # Semantic rules
├── chat_history.json       # Conversation history
├── episodic.faiss          # FAISS index
└── semantic.faiss          # Skills index
```

---

## 🤝 Contributing

We welcome contributions! Priority areas:

- **Checkpoint/resume** - Save progress for long tasks
- **Incremental repo map** - File watching for large projects
- **Batch operations** - Multi-file edits in one pass
- **Hierarchical memory** - Better scaling for 10k+ episodes
- **Session isolation** - Multi-project support

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📚 Research

NARE CLI builds on:

- **Reflexion** (Shinn et al.) - Self-refinement via verbal feedback
- **MemoryBank** (Zhong et al.) - Experience replay for LLMs
- **DreamCoder** (Ellis et al.) - Library learning
- **Anthropic Claude** - API and prompt caching

---

## 📄 Citation

```bibtex
@software{narecli2026,
  title={NARE CLI: Neural Amortized Reasoning Engine},
  author={Nare Labs},
  year={2026},
  url={https://github.com/Nare-Labs/NARE-CLI}
}
```

---

## 📜 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

## 🔗 Links

- **GitHub**: [Nare-Labs/NARE-CLI](https://github.com/Nare-Labs/NARE-CLI)
- **PyPI**: [narecli](https://pypi.org/project/narecli/)
- **Issues**: [Report bugs](https://github.com/Nare-Labs/NARE-CLI/issues)

---

<div align="center">

**Status:** Alpha - Optimized for short tasks

Made with ❤️ by [Nare Labs](https://github.com/Nare-Labs)

</div>
