"""Agent state management for NARE."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class ToolCall:
    """Represents a tool invocation."""
    
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    
    call_id: str
    output: str
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """Maintains the state of an autonomous agent."""
    
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    pending_tool_calls: List[ToolCall] = field(default_factory=list)
    completed_tool_results: List[ToolResult] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Register a pending tool call."""
        self.pending_tool_calls.append(tool_call)
    
    def complete_tool_call(self, result: ToolResult) -> None:
        """Mark a tool call as completed."""
        self.completed_tool_results.append(result)
        self.pending_tool_calls = [
            tc for tc in self.pending_tool_calls 
            if tc.call_id != result.call_id
        ]
    
    def get_context_summary(self) -> str:
        """Generate a summary of current context."""
        return f"Messages: {len(self.conversation_history)}, " \
               f"Pending calls: {len(self.pending_tool_calls)}, " \
               f"Completed: {len(self.completed_tool_results)}"