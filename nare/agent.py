"""Compatibility shim.

Historical callers expect ``from nare.agent import NAREProductionAgent`` and
``from nare.agent import HybridCritic``. The implementation has since moved
into ``nare.core`` and ``nare.reasoning``. This module re-exports the
canonical objects and the ``llm`` submodule so the legacy import paths keep
working without forcing every benchmark, test, and external script to
update at once.

Prefer the explicit imports for new code:

    from nare.core.agent import NAREProductionAgent
    from nare.reasoning.critic import Critic
    from nare.reasoning import llm
"""

from nare.core.agent import NAREProductionAgent
from nare.reasoning.critic import Critic as HybridCritic
from nare.reasoning import llm

__all__ = ["NAREProductionAgent", "HybridCritic", "llm"]
