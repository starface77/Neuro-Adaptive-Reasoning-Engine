"""
Task Executor
Handles execution of planned steps and tool calls.
"""

from typing import Dict, Any, List, Optional
import json
import re


class TaskExecutor:
    """Executes planned tasks and tool calls."""
    
    def __init__(self, tools_registry: Dict[str, Any], file_ops: Any, terminal: Any):
        """
        Initialize executor.
        
        Args:
            tools_registry: Available tools
            file_ops: File operations handler
            terminal: Terminal operations handler
        """
        self.tools = tools_registry
        self.file_ops = file_ops
        self.terminal = terminal
    
    def execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step from the plan.
        
        Args:
            step: Step definition with action and parameters
            context: Current execution context
            
        Returns:
            Execution result with status and output
        """
        action = step.get('action')
        params = step.get('parameters', {})
        
        try:
            if action == 'read_file':
                result = self.file_ops.read_file(params['path'])
            elif action == 'edit_file':
                result = self.file_ops.edit_file(
                    params['path'],
                    params.get('old', ''),
                    params.get('new', '')
                )
            elif action == 'create_file':
                result = self.file_ops.create_file(
                    params['path'],
                    params.get('content', '')
                )
            elif action == 'run_command':
                result = self.terminal.run_command(params['command'])
            else:
                result = {'error': f'Unknown action: {action}'}
            
            return {
                'status': 'success' if 'error' not in result else 'failed',
                'output': result
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool call.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            raise ValueError(f'Unknown tool: {tool_name}')
        
        tool = self.tools[tool_name]
        return tool(**arguments)
    
    def parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            List of parsed tool calls
        """
        tool_calls = []
        
        # Parse XML-style tool calls
        pattern = r'<(\w+)>(.*?)</\1>'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for tool_name, content in matches:
            if tool_name in self.tools:
                try:
                    # Try to parse as JSON
                    args = json.loads(content)
                except json.JSONDecodeError:
                    # Use as plain text
                    args = {'content': content.strip()}
                
                tool_calls.append({
                    'tool': tool_name,
                    'arguments': args
                })
        
        return tool_calls