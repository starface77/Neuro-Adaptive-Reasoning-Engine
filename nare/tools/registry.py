"""Tool registry for autonomous agent."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from nare.utils.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for agent tools with validation and execution."""
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        func: Callable,
        schema: Dict[str, Any]
    ) -> None:
        """Register a tool with its schema."""
        self._tools[name] = func
        self._schemas[name] = schema
        logger.debug(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get tool function by name."""
        return self._tools.get(name)
    
    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool schema by name."""
        return self._schemas.get(name)
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for LLM."""
        return list(self._schemas.values())
    
    def has_tool(self, name: str) -> bool:
        """Check if tool exists."""
        return name in self._tools
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())