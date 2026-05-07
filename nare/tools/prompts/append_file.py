"""Append to file tool - minimal prompt."""

TOOL_NAME = "append_file"
TOOL_DESCRIPTION = "Append text to end of file"
TOOL_PROMPT = """Append content to file.
Args:
- path: file path
- content: text to append
Example: append_file(path="log.txt", content="new line")"""

def execute(path: str, content: str) -> str:
    """Execute append file operation."""
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
        return f"Appended to {path}"
    except Exception as e:
        return f"Error appending to {path}: {e}"
