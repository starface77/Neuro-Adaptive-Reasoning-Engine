"""Backward compatibility shim — imports from nare.memory.memory."""
from .memory.memory import MemorySystem  # noqa: F401

__all__ = ["MemorySystem"]
