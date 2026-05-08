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
            lines = f.readlines()
        
        if start_line is not None or end_line is not None:
            start = (start_line - 1) if start_line else 0
            end = end_line if end_line else len(lines)
            lines = lines[start:end]
        
        content = ''.join(lines)
        total_lines = len(lines)
        
        if total_lines > 1000:
            return f"[CHUNKING REQUIRED] File has {total_lines} lines. Use start_line/end_line or grep to read specific sections.\n\nPreview (lines 1-20):\n{''.join(lines[:20])}"
        
        return f"Read {filepath}:\n```\n{content}\n```"
    
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return f"Error reading file: {str(e)}"


def create_file_tool(path: str, content: str) -> str:
    """Create a new file with content."""
    try:
        filepath = Path(path)
        
        if filepath.exists():
            return f"Error: File already exists: {path}. Use edit_file to modify it."
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Created {path}"
    
    except Exception as e:
        logger.error(f"Error creating file {path}: {e}")
        return f"Error creating file: {str(e)}"


def edit_file_tool(path: str, old: str, new: str) -> str:
    """Edit file by replacing old text with new text."""
    try:
        filepath = Path(path)
        
        if not filepath.exists():
            return f"Error: File not found: {path}"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old not in content:
            return f"Error: Failed to edit {path}: Target block not found in {path}.\nExpected:\n{old}\n\nHint: Check whitespace and indentation."
        
        new_content = content.replace(old, new, 1)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return f"Edited {path}"
    
    except Exception as e:
        logger.error(f"Error editing file {path}: {e}")
        return f"Error editing file: {str(e)}"


def list_files_tool(directory: str = ".") -> str:
    """List files in directory recursively."""
    try:
        path = Path(directory)
        
        if not path.exists():
            return f"Error: Directory not found: {directory}"
        
        files = []
        for item in path.rglob("*"):
            if item.is_file():
                files.append(str(item))
        
        if len(files) > 1000:
            return f"Found {len(files)} files:\n" + "\n".join(files[:100]) + f"\n... and {len(files) - 100} more files"
        
        return f"Found {len(files)} files:\n" + "\n".join(files)
    
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {e}")
        return f"Error listing files: {str(e)}"


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
        
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        
        if result.returncode != 0:
            output += f"\nCommand failed with exit code {result.returncode}"
        
        return output
    
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        logger.error(f"Error running command '{command}': {e}")
        return f"Error running command: {str(e)}"


def search_files_tool(pattern: str, directory: str = ".", file_pattern: Optional[str] = None) -> str:
    """Search for pattern in files using grep-like functionality."""
    try:
        path = Path(directory)
        
        if not path.exists():
            return f"Error: Directory not found: {directory}"
        
        results = []
        
        for item in path.rglob(file_pattern or "*"):
            if item.is_file():
                try:
                    with open(item, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern in line:
                                results.append(f"{item}:{line_num}: {line.rstrip()}")
                except (UnicodeDecodeError, PermissionError):
                    continue
        
        if not results:
            return f"No matches found for '{pattern}' in {directory}"
        
        if len(results) > 100:
            return f"Found {len(results)} matches:\n" + "\n".join(results[:100]) + f"\n... and {len(results) - 100} more matches"
        
        return f"Found {len(results)} matches:\n" + "\n".join(results)
    
    except Exception as e:
        logger.error(f"Error searching for '{pattern}': {e}")
        return f"Error searching: {str(e)}"


def get_tool_schemas() -> list[Dict[str, Any]]:
    """Get all tool schemas for LLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents with optional line range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "Path to file"},
                        "start_line": {"type": "integer", "description": "Start line (optional)"},
                        "end_line": {"type": "integer", "description": "End line (optional)"}
                    },
                    "required": ["filepath"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_file",
                "description": "Create a new file with content",
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
                "description": "Edit file by replacing old text with new text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "old": {"type": "string", "description": "Text to replace"},
                        "new": {"type": "string", "description": "New text"}
                    },
                    "required": ["path", "old", "new"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in directory recursively",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Directory path", "default": "."}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Execute shell command",
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
                "description": "Search for pattern in files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Search pattern"},
                        "directory": {"type": "string", "description": "Directory to search", "default": "."},
                        "file_pattern": {"type": "string", "description": "File pattern (e.g., '*.py')"}
                    },
                    "required": ["pattern"]
                }
            }
        }
    ]