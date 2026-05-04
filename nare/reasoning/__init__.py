"""Reasoning subsystem (LLM client, critic, oracle)."""

from nare.reasoning.generation import engine as llm
from nare.reasoning.generation.ranker import Critic
from nare.reasoning.verification.oracle import (
    Oracle,
    build_oracle_from_spec,
    heuristic_overlap_oracle,
    numeric_set_oracle,
    python_assert_oracle,
    string_contains_oracle,
)

__all__ = [
    "llm",
    "Critic",
    "Oracle",
    "build_oracle_from_spec",
    "heuristic_overlap_oracle",
    "numeric_set_oracle",
    "python_assert_oracle",
    "string_contains_oracle",
]
