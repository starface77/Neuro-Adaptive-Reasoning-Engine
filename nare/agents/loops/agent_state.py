"""Data structures for autonomous agent state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """Represents a tool invocation request."""
    name: str
    args: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_call_id: Optional[str]
    content: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class AgentState:
    """Tracks the state of an autonomous agent session."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 50
    total_tokens: int = 0
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
    
    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Register a tool call."""
        self.tool_calls.append(tool_call)
    
    def add_tool_result(self, result: ToolResult) -> None:
        """Register a tool execution result."""
        self.tool_results.append(result)
    
    def increment_iteration(self) -> None:
        """Move to next iteration."""
        self.iteration += 1
    
    def is_complete(self) -> bool:
        """Check if agent should stop."""
        return self.iteration >= self.max_iterations