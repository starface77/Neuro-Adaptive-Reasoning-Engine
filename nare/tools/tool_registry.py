"""Tool registry for managing available tools."""

from typing import Dict, Callable, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ToolDefinition:
    """Defines a tool's metadata and implementation."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    category: Optional[str] = None


class ToolRegistry:
    """Registry for managing and executing tools."""
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
    
    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable,
        category: Optional[str] = None
    ) -> None:
        """Register a new tool."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            category=category
        )
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> List[ToolDefinition]:
        """List all registered tools, optionally filtered by category."""
        if category:
            return [t for t in self._tools.values() if t.category == category]
        return list(self._tools.values())
    
    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with given arguments."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found in registry")
        
        try:
            return tool.function(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Tool '{name}' execution failed: {str(e)}") from e
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for all registered tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self._tools.values()
        ]