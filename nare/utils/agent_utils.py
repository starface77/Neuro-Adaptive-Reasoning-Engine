"""Utility functions for autonomous agent."""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Tuple

from nare.utils.logger import get_logger

logger = get_logger(__name__)


def parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from XML-style tags in text."""
    tool_calls = []
    
    # Pattern: <tool_name>args</tool_name>
    pattern = r'<(\w+)>(.*?)</\1>'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        tool_name = match.group(1)
        args_text = match.group(2).strip()
        
        try:
            # Try to parse as JSON
            if args_text.startswith('{'):
                args = json.loads(args_text)
            else:
                # Simple key=value parsing
                args = {}
                for line in args_text.split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        args[key.strip()] = value.strip()
            
            tool_calls.append({
                "name": tool_name,
                "args": args
            })
        except Exception as e:
            logger.warning(f"Failed to parse tool call {tool_name}: {e}")
    
    return tool_calls


def format_tool_result(tool_name: str, result: str, success: bool = True) -> str:
    """Format tool execution result for LLM."""
    status = "SUCCESS" if success else "ERROR"
    return f"[{status}] {tool_name}:\n{result}"


def truncate_output(text: str, max_length: int = 2000) -> str:
    """Truncate long output for context management."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + f"\n... (truncated {len(text) - max_length} characters)"


def extract_code_blocks(text: str) -> List[Tuple[Optional[str], str]]:
    """Extract code blocks from markdown text."""
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    blocks = []
    for match in matches:
        language = match.group(1)
        code = match.group(2).strip()
        blocks.append((language, code))
    
    return blocks


def validate_tool_args(tool_name: str, args: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate tool arguments against schema."""
    try:
        required = schema.get("function", {}).get("parameters", {}).get("required", [])
        properties = schema.get("function", {}).get("parameters", {}).get("properties", {})
        
        # Check required args
        for req in required:
            if req not in args:
                return False, f"Missing required argument: {req}"
        
        # Check arg types
        for arg_name, arg_value in args.items():
            if arg_name not in properties:
                return False, f"Unknown argument: {arg_name}"
            
            expected_type = properties[arg_name].get("type")
            if expected_type == "string" and not isinstance(arg_value, str):
                return False, f"Argument {arg_name} must be string"
            elif expected_type == "integer" and not isinstance(arg_value, int):
                return False, f"Argument {arg_name} must be integer"
        
        return True, None
    
    except Exception as e:
        logger.error(f"Error validating args for {tool_name}: {e}")
        return False, str(e)


def build_context_summary(messages: List[Dict[str, Any]], max_messages: int = 5) -> str:
    """Build a summary of recent conversation context."""
    recent = messages[-max_messages:]
    
    summary_parts = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Truncate long messages
        if len(content) > 200:
            content = content[:200] + "..."
        
        summary_parts.append(f"{role}: {content}")
    
    return "\n".join(summary_parts)