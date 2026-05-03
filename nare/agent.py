"""Backward compatibility shim — imports from nare.core.agent."""
from .core.agent import VareAgent, NAREProductionAgent  # noqa: F401

__all__ = ["VareAgent", "NAREProductionAgent"]
