"""Backward compatibility alias for oracle module."""

from nare.reasoning.verification.oracle import (
    Oracle,
    numeric_set_oracle,
    string_contains_oracle,
    python_assert_oracle,
    build_oracle_from_spec,
)

__all__ = [
    "Oracle",
    "numeric_set_oracle",
    "string_contains_oracle",
    "python_assert_oracle",
    "build_oracle_from_spec",
]
