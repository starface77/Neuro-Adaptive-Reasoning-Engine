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
    is_error: bool = False


@dataclass
class AgentState:
    """State container for autonomous agent."""
    
    task: str
    context: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 50
    budget_used: int = 0
    budget_total: int = 200000
    completed: bool = False
    error: Optional[str] = None
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self.messages.append({"role": role, "content": content})
    
    def increment_iteration(self):
        """Increment iteration counter."""
        self.iteration += 1
    
    def consume_budget(self, amount: int):
        """Consume tokens from budget."""
        self.budget_used += amount
    
    @property
    def budget_remaining(self) -> int:
        """Get remaining budget."""
        return self.budget_total - self.budget_used
    
    def should_stop(self) -> bool:
        """Check if agent should stop."""
        return (
            self.completed
            or self.budget_remaining <= 0
            or self.iteration >= self.max_iterations
            or self.error is not None
        )