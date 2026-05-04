"""Compatibility shim: ``nare.sandbox`` → :mod:`nare.execution.sandbox`.

Older callers (tests, benchmarks, external scripts) import directly
from ``nare.sandbox``. This module re-exports the canonical symbols so
those imports keep working without forcing a global rename.
"""

from nare.execution.sandbox import *  # noqa: F401,F403
from nare.execution.sandbox import (  # noqa: F401
    SecurityError,
    safe_execute,
    safe_execute_freeform,
)
