"""Compatibility shim: ``nare.oracle`` → :mod:`nare.reasoning.oracle`."""

from nare.reasoning.oracle import *  # noqa: F401,F403
from nare.reasoning.oracle import (  # noqa: F401
    Oracle,
    build_oracle_from_spec,
    heuristic_overlap_oracle,
    numeric_set_oracle,
    python_assert_oracle,
    string_contains_oracle,
)
