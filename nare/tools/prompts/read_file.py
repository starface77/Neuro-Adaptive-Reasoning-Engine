"""Read file tool - minimal prompt."""

TOOL_NAME = "read_file"
TOOL_DESCRIPTION = "Read contents of a file"
TOOL_PROMPT = """Read file and return contents.
Args:
- path: file path
Example: read_file(path="test.py")"""

def execute(path: str) -> str:
    """Execute read file operation."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"
