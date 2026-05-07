"""List files tool - minimal prompt."""

import os

TOOL_NAME = "list_files"
TOOL_DESCRIPTION = "List files in directory"
TOOL_PROMPT = """List files in directory.
Args:
- path: directory path (default: current dir)
Example: list_files(path=".")"""

def execute(path: str = ".") -> str:
    """Execute list files operation."""
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing {path}: {e}"
