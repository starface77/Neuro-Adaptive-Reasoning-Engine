"""Agent loops for autonomous and synthesis workflows."""

from .autonomous import AutonomousAgent
from .synthesis import SynthesisAgent
from nare.agents.state import ToolCall, ToolResult, AgentState
from nare.tools.registry import ToolRegistry

__all__ = [
    "AutonomousAgent",
    "SynthesisAgent",
    "ToolCall",
    "ToolResult",
    "AgentState",
    "ToolRegistry"
]