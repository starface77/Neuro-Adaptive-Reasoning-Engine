"""Tool registry and execution."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from nare.utils.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
    
    def register(self, name: str, func: Callable):
        """Register a tool."""
        self._tools[name] = func
        logger.debug(f"Registered tool: {name}")
    
    def get(self, name: str) -> Optional[Callable]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def has_tool(self, name: str) -> bool:
        """Check if tool exists."""
        return name in self._tools


def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    registry: ToolRegistry
) -> Dict[str, Any]:
    """
    Execute a tool from the registry.
    
    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments to pass to the tool
        registry: Tool registry instance
    
    Returns:
        Dict with 'success', 'result', and optional 'error' keys
    """
    try:
        tool_func = registry.get(tool_name)
        
        if tool_func is None:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found in registry"
            }
        
        result = tool_func(**tool_args)
        
        return {
            "success": True,
            "result": result
        }
    
    except Exception as e:
        logger.error(f"Tool execution failed: {tool_name}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }