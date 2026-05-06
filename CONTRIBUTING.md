# Contributing to NARE

Thanks for your interest in contributing to NARE! This document provides guidelines and instructions for contributing.

---

## Quick Start

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/Nare-Labs/NARE-CLI
   cd nare
   ```
3. **Install in development mode**
   ```bash
   pip install -e .
   ```
4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make your changes**
6. **Run tests**
   ```bash
   pytest tests/
   ```
7. **Commit and push**
   ```bash
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```
8. **Open a Pull Request**

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- Anthropic API key (for testing)

### Installation

```bash
# Clone repo
git clone https://github.com/Nare-Labs/NARE-CLI
cd nare

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Set up API key
export ANTHROPIC_API_KEY="your-key"
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_agent_loop.py

# Run with coverage
pytest --cov=nare tests/

# Run with verbose output
pytest -v tests/
```

---

## Code Style

### Python Style Guide

- Follow **PEP 8**
- Use **type hints** for all function signatures
- Maximum line length: **100 characters**
- Use **docstrings** for all public functions/classes

### Example

```python
from typing import Dict, Any, Optional

def process_query(
    query: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a user query and return results.
    
    Args:
        query: User's input query
        context: Optional context dictionary
        
    Returns:
        Dictionary containing results and metadata
    """
    # Implementation here
    pass
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `MemorySystem`)
- **Functions/methods**: `snake_case` (e.g., `get_embedding`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`)
- **Private methods**: `_leading_underscore` (e.g., `_internal_method`)

---

## Project Structure

```
nare/
├── agents/          # Agent implementations
│   ├── loops/       # Agent execution loops
│   └── roles/       # Specialized agent roles
├── cli/             # Command-line interface
├── config/          # Configuration management
├── core/            # Core logic
│   ├── evolution/   # Learning and skill compilation
│   ├── routing/     # Smart routing engine
│   └── synthesis/   # Verified synthesis
├── memory/          # Memory system
│   └── analytics/   # Memory analytics
├── reasoning/       # LLM reasoning
│   ├── adapters/    # Provider adapters
│   └── generation/  # Generation logic
└── tools/           # Agent tools
    ├── builtin/     # Built-in tools
    └── parsing/     # Tool parsing
```

---

## What to Contribute

### High Priority

- 🐛 **Bug fixes** — Always welcome
- 📝 **Documentation** — Improve clarity, add examples
- ✅ **Tests** — Increase coverage
- ⚡ **Performance** — Optimize hot paths

### Feature Ideas

- Multi-language support (JavaScript, Go, Rust)
- VSCode extension
- Team memory sharing
- Self-hosted memory backend
- Additional LLM providers (OpenAI, Google)

### Not Accepting

- Breaking API changes without discussion
- Features that compromise security
- Large refactors without prior approval

---

## Pull Request Process

### Before Submitting

1. ✅ Tests pass locally
2. ✅ Code follows style guide
3. ✅ Docstrings added for new functions
4. ✅ CHANGELOG.md updated (if applicable)
5. ✅ No merge conflicts with main

### PR Title Format

```
<type>: <description>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation only
- refactor: Code refactoring
- test: Adding tests
- perf: Performance improvement
```

**Examples:**
- `feat: Add OpenAI provider support`
- `fix: Memory leak in FAISS indexing`
- `docs: Update installation instructions`

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

---

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Mirror source structure (e.g., `nare/core/agent.py` → `tests/test_agent.py`)
- Use descriptive test names: `test_memory_retrieval_with_empty_query`

### Test Structure

```python
import pytest
from nare.core.agent import NAREProductionAgent

def test_agent_initialization():
    """Test that agent initializes correctly."""
    agent = NAREProductionAgent()
    assert agent.memory is not None
    assert agent.router is not None

def test_agent_solve_simple_query():
    """Test agent can solve a simple query."""
    agent = NAREProductionAgent()
    result = await agent.solve("What is 2+2?")
    assert result["ok"] is True
```

### Running Specific Tests

```bash
# Run tests matching pattern
pytest -k "test_memory"

# Run tests in specific file
pytest tests/test_agent.py

# Run with markers
pytest -m "slow"  # Run only slow tests
```

---

## Documentation

### Docstring Format

Use **Google-style docstrings**:

```python
def retrieve_episodes(
    self,
    query_embedding: np.ndarray,
    k: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve similar episodes from memory.
    
    Args:
        query_embedding: Query vector (shape: [1, embedding_dim])
        k: Number of results to return
        
    Returns:
        List of episode dictionaries with similarity scores
        
    Raises:
        ValueError: If query_embedding has wrong shape
        
    Example:
        >>> memory = MemorySystem()
        >>> embedding = get_embedding("fix bug")
        >>> episodes = memory.retrieve_episodes(embedding, k=3)
    """
    pass
```

### Adding Examples

Add usage examples in `examples/` directory:

```python
# examples/basic_usage.py
from nare.core.agent import NAREProductionAgent

async def main():
    agent = NAREProductionAgent()
    result = await agent.solve("Fix the bug in auth.py")
    print(result["final_answer"])
```

---

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Example

```
feat(memory): Add FAISS index pruning

Implement automatic pruning of low-quality episodes
to keep memory size under control.

- Add pruning threshold configuration
- Implement similarity-based deduplication
- Add tests for pruning logic

Closes #123
```

---

## Code Review Process

1. **Automated checks** run on PR (tests, linting)
2. **Maintainer review** within 48 hours
3. **Address feedback** and push updates
4. **Approval** from at least one maintainer
5. **Merge** by maintainer

---

## Getting Help

- 💬 **Discussions**: [GitHub Discussions](https://github.com/Nare-Labs/NARE-CLI/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Nare-Labs/NARE-CLI/issues)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to NARE!** 🚀
