"""NARE — Neural Amortized Reasoning Engine.

Professional agentic framework for verified synthesis, episodic memory,
and library learning.
"""

__version__ = "0.2.1"

from .core.agent import NAREProductionAgent
from .memory.engine import MemorySystem
from .reasoning import llm
from .memory.analytics.metrics import MetricsTracker
from .config import (
    NareConfig,
    DEFAULT_CONFIG,
    RoutingConfig,
    SleepConfig,
    CriticConfig,
    SkillLifecycleConfig,
    RetrievalConfig,
    SkillValidationConfig,
)
from .reasoning.verification.oracle import (
    Oracle,
    numeric_set_oracle,
    string_contains_oracle,
    python_assert_oracle,
    heuristic_overlap_oracle,
    build_oracle_from_spec,
)
