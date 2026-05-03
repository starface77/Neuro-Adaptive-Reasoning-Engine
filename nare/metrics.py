"""Backward compatibility shim — imports from nare.memory.metrics."""
from .memory.metrics import MetricsTracker  # noqa: F401

__all__ = ["MetricsTracker"]
