"""
Tool registry for the NARE agent loop.

The loop calls tools by name with a JSON payload; the tool runs and
returns a `ToolResult`. Each tool has a typed schema (parameters and
their descriptions) which is rendered into the LLM system prompt so
the model can produce well-formed tool calls.

Public API:
    Tool, ToolResult, ToolRegistry
    DEFAULT_REGISTRY      — pre-populated with built-in tools
"""

from .base import Tool, ToolResult, ToolRegistry, ToolError
from .builtin import build_default_registry

DEFAULT_REGISTRY: ToolRegistry = build_default_registry()

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "ToolError",
    "DEFAULT_REGISTRY",
    "build_default_registry",
]
