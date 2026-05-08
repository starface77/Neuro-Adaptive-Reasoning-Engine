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
        
        # Try to parse as JSON
        try:
            args = json.loads(args_text)
        except:
            # Fallback: treat as single string argument
            args = {"content": args_text}
        
        tool_calls.append({
            "name": tool_name,
            "args": args
        })
    
    return tool_calls


def format_tool_result(tool_name: str, result: str, success: bool = True) -> str:
    """Format tool execution result for LLM."""
    status = "✓" if success else "✗"
    return f"{status} {tool_name}: {result}"


def truncate_output(text: str, max_length: int = 2000) -> str:
    """Truncate long output to prevent context overflow."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"\n... (truncated {len(text) - max_length} chars)"


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Extract code blocks with language tags from markdown."""
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang or "text", code.strip()) for lang, code in matches]


def count_tokens_estimate(text: str) -> int:
    """Rough token count estimation (1 token ≈ 4 chars)."""
    return len(text) // 4


def sanitize_file_path(path: str) -> str:
    """Sanitize file path to prevent directory traversal."""
    # Remove leading slashes and normalize
    path = path.lstrip('/')
    # Remove .. components
    parts = path.split('/')
    safe_parts = [p for p in parts if p != '..']
    return '/'.join(safe_parts)


def is_safe_command(command: str) -> bool:
    """Check if command is safe to execute."""
    dangerous_patterns = [
        r'\brm\s+-rf\s+/',  # rm -rf /
        r'\bformat\b',       # format command
        r'\bdel\s+/[sq]',   # Windows del /s /q
        r'>\s*/dev/sd',      # Writing to disk devices
        r'\bdd\s+if=',       # dd command
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            logger.warning(f"Blocked dangerous command: {command}")
            return False
    return True


def format_conversation_history(messages: List[Dict[str, Any]], max_messages: int = 10) -> str:
    """Format recent conversation history for display."""
    recent = messages[-max_messages:]
    formatted = []
    
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted.append(f"{role.upper()}: {content[:200]}...")
    
    return "\n".join(formatted)