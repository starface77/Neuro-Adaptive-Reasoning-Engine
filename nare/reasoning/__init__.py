"""Reasoning subsystem (LLM client, critic, oracle)."""

from nare.reasoning.critic import Critic  # noqa: F401
from nare.reasoning.oracle import (  # noqa: F401
    Oracle,
    build_oracle_from_spec,
    heuristic_overlap_oracle,
    numeric_set_oracle,
    python_assert_oracle,
    string_contains_oracle,
)

__all__ = [
    "Critic",
    "Oracle",
    "build_oracle_from_spec",
    "heuristic_overlap_oracle",
    "numeric_set_oracle",
    "python_assert_oracle",
    "string_contains_oracle",
]
