"""Domain detection for adaptive routing thresholds."""

import re
from typing import Literal

DomainType = Literal["code", "pattern", "reasoning"]

def detect_domain(query: str) -> DomainType:
    """Detect task domain from query text.

    Returns:
        "code" - Software engineering tasks (file paths, bugs, PRs)
        "pattern" - Visual/grid pattern tasks (ARC-AGI)
        "reasoning" - General reasoning tasks
    """
    query_lower = query.lower()

    code_indicators = [
        "file path", "github", "bug", "issue", "pull request",
        "django", "astropy", "pytest", "import", "class",
        "function", "method", "repository", "commit",
        ".py", ".js", ".java", ".cpp", "def ", "class "
    ]

    pattern_indicators = [
        "grid", "pattern", "transformation", "arc-agi",
        "input:", "output:", "training examples",
        "[[", "]]",
        "tile", "rotate", "flip", "mirror"
    ]

    code_score = sum(1 for ind in code_indicators if ind in query_lower)
    pattern_score = sum(1 for ind in pattern_indicators if ind in query_lower)

    if re.search(r'\[\[.*?\]\]', query):
        pattern_score += 3

    if re.search(r'[a-z_]+/[a-z_]+\.py', query_lower):
        code_score += 3

    if code_score > pattern_score and code_score >= 2:
        return "code"
    elif pattern_score > code_score and pattern_score >= 2:
        return "pattern"
    else:
        return "reasoning"

def get_adaptive_tau_fast(query: str, config) -> float:
    """Get adaptive tau_fast threshold based on query domain.

    Args:
        query: Task query text
        config: NareConfig instance

    Returns:
        Adaptive tau_fast threshold
    """
    domain = detect_domain(query)

    if domain == "code":
        return config.routing.tau_fast_code
    elif domain == "pattern":
        return config.routing.tau_fast_pattern
    else:
        return config.routing.tau_fast
