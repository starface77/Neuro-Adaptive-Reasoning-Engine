"""Backward compatibility alias for LLM module.

The actual implementation lives in nare.reasoning.generation.engine.
This shim re-exports the public API so that existing imports like
``from nare.llm import _validate_skill`` continue to work.
"""

from nare.reasoning.generation.engine import (
    _validate_skill,
    generate_samples,
    get_embedding,
    generate_stress_tests,
    extract_heuristic_rule,
    repair_skill,
    merge_heuristic_rules,
)

__all__ = [
    "_validate_skill",
    "generate_samples",
    "get_embedding",
    "generate_stress_tests",
    "extract_heuristic_rule",
    "repair_skill",
    "merge_heuristic_rules",
]
