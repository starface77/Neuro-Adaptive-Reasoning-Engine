"""Bash command tool - minimal prompt."""

import subprocess

TOOL_NAME = "bash_command"
TOOL_DESCRIPTION = "Execute bash command"
TOOL_PROMPT = """Run bash command and return output.
Args:
- command: bash command to run
Example: bash_command(command="ls -la")"""

def execute(command: str) -> str:
    """Execute bash command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error running command: {e}"
