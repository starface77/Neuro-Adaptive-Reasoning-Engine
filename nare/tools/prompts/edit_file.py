"""Edit file tool - minimal prompt."""

TOOL_NAME = "edit_file"
TOOL_DESCRIPTION = "Edit existing file by replacing text"
TOOL_PROMPT = """Replace text in file.
Args:
- path: file path
- old_text: text to find
- new_text: replacement text
Example: edit_file(path="test.py", old_text="old", new_text="new")"""

def execute(path: str, old_text: str, new_text: str) -> str:
    """Execute edit file operation."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        if old_text not in content:
            return f"Text not found in {path}"

        new_content = content.replace(old_text, new_text)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return f"Edited {path}"
    except Exception as e:
        return f"Error editing {path}: {e}"
