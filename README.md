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

### Commands

- `/help` - Show available commands
- `/status` - Agent status and memory stats
- `/agent on/off` - Toggle autonomous mode
- `/repo <path>` - Set working directory
- `/clear` - Clear screen
- `/exit` - Exit NARE

## Architecture

NARE CLI implements the **VARE** (Verified Amortized Reasoning Engine) architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Router (Triage)     │
         │  QUESTION/EXPLORE/EDIT │
         └───────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌──────────────┐
│  FAST Route   │         │  SLOW Route  │
│  (Memory)     │         │  (Synthesis) │
└───────┬───────┘         └──────┬───────┘
        │                        │
        │  ┌─────────────────┐   │
        └─►│  Final Answer   │◄──┘
           └─────────────────┘
```

### Components

**$M_{cache}$ (Memory)**
- HNSW-indexed episodic memory (FAISS)
- Compiled skills with trigger/execute functions
- Semantic rules for pattern matching

**$V_{sandbox}$ (Verifier)**
- Python AST compilation check
- Subprocess isolation
- Binary feedback (R(y) = 1 or 0)

**$G_{\theta}$ (Generator)**
- LLM with fixed weights (Anthropic API)
- Self-refinement via error traces
- Thinking budget optimization

**Routing System**
1. **FAST** - Episodic memory lookup (instant)
2. **REFLEX** - Compiled skills execution (instant)
3. **COMPILED_SKILL** - Semantic pattern matching
4. **HYBRID** - Delta reasoning with memory
5. **SLOW** - Full verified synthesis loop

## Configuration

### Budget Limits

```python
# nare/agents/loops/autonomous.py
max_iterations: int = 150      # Max tool-calling iterations
max_tokens: int = 1_000_000    # Max tokens per task
max_wall_clock: float = 7200.0 # Max 2 hours per task
```

### Memory Settings

```python
# .nare_memory/ directory structure
├── episodes.json           # Episodic memory
├── compiled_skills.json    # Learned skills
├── rules.json              # Semantic rules
├── chat_history.json       # Conversation history
├── episodic.faiss          # FAISS index
└── semantic.faiss          # Skills index
```

## Token Optimization

NARE CLI includes aggressive token optimization:

- **System prompt**: Compressed to ~800 tokens (-68%)
- **Chat history**: Code blocks trimmed to 100 chars
- **Repo map**: Cached for 15 seconds, uses `git ls-files`
- **Thinking budget**: Adaptive 200 tokens
- **Prompt caching**: 5-minute TTL for system prompt + repo map

**Results:**
- Simple greeting: ~100-200 tokens (instant response)
- FAST route: 0 LLM tokens (memory lookup)
- Typical session: 106k → 15.8k tokens (-85%)

## Known Limitations

NARE CLI is optimized for **short-to-medium tasks** on **small-to-medium projects**. For production use on large projects, be aware of:

### Critical Limitations
1. **Budget limits** - 150 iterations may be insufficient for complex refactorings
2. **Repo map** - Limited to 1500 files via `git ls-files`
3. **Chat history** - Only last 10 messages retained
4. **Memory prune** - Aggressive pruning every 100 episodes
5. **No checkpoint/resume** - Crash = lost work

### Performance Issues
6. **FAISS scaling** - O(n) search, slow on 10k+ episodes
7. **Synchronous execution** - No parallelism
8. **No batch operations** - Each file = separate iteration

See [REAL_PROBLEMS_ANALYSIS.md](REAL_PROBLEMS_ANALYSIS.md) for complete list.

## Development

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

### Running Tests

```bash
# Memory flush test
python test_memory_flush.py

# Hunks system test
python test_hunks.py

# Full test suite
pytest tests/
```

## Contributing

Contributions welcome! Priority areas:

1. **Checkpoint/resume** for long-running tasks
2. **Incremental repo map** with file watching
3. **Batch operations** for multi-file edits
4. **Hierarchical memory** with summarization
5. **Session isolation** for multi-project work

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{narecli2026,
  title={NARE CLI: Neural Amortized Reasoning Engine},
  author={Nare Labs},
  year={2026},
  url={https://github.com/Nare-Labs/NARE-CLI}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built on research from:
- **Reflexion** (Shinn et al.) - Self-refinement via verbal feedback
- **MemoryBank** (Zhong et al.) - Experience replay for LLMs
- **DreamCoder** (Ellis et al.) - Library learning
- **Anthropic** - Claude API and prompt caching

---

**Status:** Alpha - Optimized for short tasks, known limitations for production use.

For questions: [Issues](https://github.com/Nare-Labs/NARE-CLI/issues)
