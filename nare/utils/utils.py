"""Utility functions for NARE."""

import re
import json
from pathlib import Path
from typing import Dict, Any, Optional


def format_tool_output(result: Dict[str, Any]) -> str:
    """Format tool execution result for display."""
    if not result.get("success"):
        return f"❌ Error: {result.get('error', 'Unknown error')}"
    
    output_parts = ["✅ Success"]
    for key, value in result.items():
        if key not in ("success", "error"):
            if isinstance(value, (list, dict)):
                output_parts.append(f"{key}: {json.dumps(value, indent=2)}")
            else:
                output_parts.append(f"{key}: {value}")
    
    return "\n".join(output_parts)


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse a tool call from text format."""
    # Example format: tool_name(arg1="value1", arg2="value2")
    pattern = r'(\w+)\((.*?)\)'
    match = re.search(pattern, text)
    
    if not match:
        return None
    
    tool_name = match.group(1)
    args_str = match.group(2)
    
    # Simple argument parsing (can be enhanced)
    args = {}
    if args_str:
        for arg in args_str.split(','):
            if '=' in arg:
                key, value = arg.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                args[key] = value
    
    return {
        "name": tool_name,
        "arguments": args
    }


def validate_file_path(filepath: str, base_dir: str = ".") -> bool:
    """Validate that a file path is safe and within base directory."""
    try:
        base = Path(base_dir).resolve()
        target = Path(filepath).resolve()
        
        # Check if target is within base directory
        return target.is_relative_to(base)
    except (ValueError, OSError):
        return False


def sanitize_command(command: str) -> str:
    """Sanitize a shell command to prevent injection."""
    # Remove dangerous characters and patterns
    dangerous_patterns = [
        r'[;&|`$]',  # Command chaining
        r'\$\(',     # Command substitution
        r'>\s*/dev', # Device access
    ]
    
    sanitized = command
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized)
    
    return sanitized.strip()