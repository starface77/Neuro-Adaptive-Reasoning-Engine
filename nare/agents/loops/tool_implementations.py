"""Tool implementations for autonomous agent."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from nare.utils.logger import get_logger

logger = get_logger(__name__)


def read_file_tool(filepath: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Read file contents with optional line range."""
    try:
        path = Path(filepath)
        if not path.exists():
            return f"Error: File not found: {filepath}"
        
        with open(path, 'r', encoding='utf-8') as f:
            if start_line is not None or end_line is not None:
                lines = f.readlines()
                start = (start_line - 1) if start_line else 0
                end = end_line if end_line else len(lines)
                content = ''.join(lines[start:end])
                return f"Read {filepath} (lines {start+1}-{end}):\n```\n{content}\n```"
            else:
                content = f.read()
                line_count = content.count('\n') + 1
                if line_count > 100:
                    preview = '\n'.join(content.split('\n')[:20])
                    return f"[CHUNKING REQUIRED] File has {line_count} lines. Use start_line/end_line or grep to read specific sections.\n\nPreview (lines 1-20):\n```\n{preview}\n```"
                return f"Read {filepath}:\n```\n{content}\n```"
    except Exception as e:
        return f"Error reading {filepath}: {str(e)}"


def create_file_tool(path: str, content: str) -> str:
    """Create a new file with content."""
    try:
        file_path = Path(path)
        if file_path.exists():
            return f"Error: File already exists: {path}. Use edit_file to modify it."
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Created {path}"
    except Exception as e:
        return f"Error creating {path}: {str(e)}"


def edit_file_tool(path: str, old: str, new: str) -> str:
    """Edit file by replacing old text with new text."""
    try:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File not found: {path}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old not in content:
            return f"Error: Old text not found in {path}"
        
        new_content = content.replace(old, new, 1)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return f"Edited {path}"
    except Exception as e:
        return f"Error editing {path}: {str(e)}"


def list_files_tool(directory: str = ".") -> str:
    """List files in directory recursively."""
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return f"Error: Directory not found: {directory}"
        
        files = []
        for item in dir_path.rglob("*"):
            if item.is_file():
                files.append(str(item))
        
        if len(files) > 100:
            return f"Found {len(files)} files:\n" + "\n".join(files[:100]) + f"\n... and {len(files) - 100} more"
        return f"Found {len(files)} files:\n" + "\n".join(files)
    except Exception as e:
        return f"Error listing {directory}: {str(e)}"


def run_command_tool(command: str, cwd: Optional[str] = None) -> str:
    """Execute shell command and return output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout + result.stderr
        return f"Command: {command}\nExit code: {result.returncode}\nOutput:\n{output}"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out: {command}"
    except Exception as e:
        return f"Error running command: {str(e)}"


def search_files_tool(pattern: str, directory: str = ".", file_pattern: Optional[str] = None) -> str:
    """Search for text pattern in files."""
    try:
        import re
        dir_path = Path(directory)
        if not dir_path.exists():
            return f"Error: Directory not found: {directory}"
        
        results = []
        regex = re.compile(pattern)
        
        for file_path in dir_path.rglob(file_pattern or "*"):
            if not file_path.is_file():
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(f"{file_path}:{line_num}: {line.strip()}")
                            if len(results) >= 50:
                                return "Found 50+ matches:\n" + "\n".join(results[:50]) + "\n... (truncated)"
            except:
                continue
        
        if not results:
            return f"No matches found for pattern: {pattern}"
        return f"Found {len(results)} matches:\n" + "\n".join(results)
    except Exception as e:
        return f"Error searching: {str(e)}"


# Tool schemas for LLM
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents. For large files, use start_line/end_line to read specific sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to file"},
                    "start_line": {"type": "integer", "description": "Optional start line (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Optional end line (inclusive)"}
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file with content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit file by replacing old text with new text (single replacement).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old": {"type": "string", "description": "Text to replace"},
                    "new": {"type": "string", "description": "Replacement text"}
                },
                "required": ["path", "old", "new"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in directory recursively.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory path (default: current)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute shell command and return output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"},
                    "cwd": {"type": "string", "description": "Working directory (optional)"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for text pattern in files using regex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "directory": {"type": "string", "description": "Directory to search (default: current)"},
                    "file_pattern": {"type": "string", "description": "File glob pattern (e.g., '*.py')"}
                },
                "required": ["pattern"]
            }
        }
    }
]


def get_tool_function(name: str):
    """Get tool function by name."""
    tools = {
        "read_file": read_file_tool,
        "create_file": create_file_tool,
        "edit_file": edit_file_tool,
        "list_files": list_files_tool,
        "run_command": run_command_tool,
        "search_files": search_files_tool
    }
    return tools.get(name)