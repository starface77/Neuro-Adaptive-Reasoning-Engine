"""Backward-compatibility re-exports for NARE.

All legacy import paths are consolidated here so that external scripts,
benchmarks, and tests that were written against older layouts keep working.

New code should import from the canonical locations:
    from nare.core.agent      import NAREProductionAgent
    from nare.core.synthesis  import verified_synthesis
    from nare.core.router     import ReasoningRouter
    from nare.reasoning       import llm
    from nare.reasoning.generation.ranker import Critic
    from nare.execution.sandboxes.base.base import safe_execute_freeform, SecurityError

These shims will NOT be removed before v1.0; they carry zero runtime cost
(pure re-export at import time).
"""

from nare.core.agent import NAREProductionAgent
from nare.core.routing.router import ReasoningRouter
from nare.core.synthesis.engine import verified_synthesis, SynthesisResult
from nare.core.evolution.engine import EvolutionEngine
from nare.core.evolution.learning import discover_rule

from nare.reasoning import llm
from nare.reasoning.generation.ranker import Critic
from nare.reasoning.generation.ranker import Critic as HybridCritic

from nare.execution.sandboxes.base import (
    SecurityError,
    safe_execute,
    safe_execute_freeform,
    extract_python_block,
    safe_call_trigger,
    safe_call_execute_in_namespace,
    safe_load_module,
)

__all__ = [

    "NAREProductionAgent",
    "ReasoningRouter",
    "verified_synthesis",
    "SynthesisResult",
    "EvolutionEngine",
    "discover_rule",

    "llm",
    "Critic",
    "HybridCritic",

    "SecurityError",
    "safe_execute",
    "safe_execute_freeform",
    "extract_python_block",
    "safe_call_trigger",
    "safe_call_execute_in_namespace",
    "safe_load_module",
]
