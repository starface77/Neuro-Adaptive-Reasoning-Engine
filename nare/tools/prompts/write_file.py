"""Write file tool - minimal prompt."""

TOOL_NAME = "write_file"
TOOL_DESCRIPTION = "Create or overwrite a file"
TOOL_PROMPT = """Write content to file.
Args:
- path: file path
- content: text to write
Example: write_file(path="test.py", content="print('hi')")"""

def execute(path: str, content: str) -> str:
    """Execute write file operation."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Written to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"
