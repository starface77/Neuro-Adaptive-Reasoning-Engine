"""
File system tools for autonomous LLM operations.

Provides safe, validated file operations that the LLM can call:
- read_file: Read file contents with optional line range
- create_file: Create new files with parent directories
- edit_file: Surgically replace code blocks in existing files
"""

import os
from pathlib import Path
from typing import Optional, Tuple

# Import safety layer for path validation
from nare.tools.safety import get_safety

def _validate_path(filepath: str, operation: str = "read") -> Path:
    """Validate file path for security.

    Args:
        filepath: Path to validate
        operation: Operation type ("read", "write", "delete")

    Returns:
        Resolved Path object

    Raises:
        ValueError: If path is unsafe
    """
    try:
        # Resolve to absolute path and check for traversal
        resolved = Path(filepath).resolve()

        # Get safety layer
        safety = get_safety()

        # Check if path is within working directory
        if operation in ("write", "delete"):
            ok, reason = safety.check_path_write(str(resolved))
            if not ok:
                raise ValueError(reason)

        return resolved
    except Exception as e:
        raise ValueError(f"Invalid path '{filepath}': {e}")

def read_file(filepath: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Read file contents safely.

    Args:
        filepath: Path to file (relative or absolute)
        start_line: Optional starting line (1-indexed)
        end_line: Optional ending line (1-indexed, inclusive)

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If line range is invalid or path is unsafe
    """
    # Validate path
    resolved = _validate_path(filepath, "read")

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        if len(lines) > 500 and start_line is None and end_line is None:
            return f"[WARNING] File has {len(lines)} lines. Please specify start_line and end_line to read a specific range.\n\nFirst 50 lines:\n" + ''.join(lines[:50])

        if start_line is not None or end_line is not None:
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else len(lines)

            if start_idx < 0 or end_idx > len(lines) or start_idx >= end_idx:
                raise ValueError(f"Invalid line range: {start_line}-{end_line} (file has {len(lines)} lines)")

            lines = lines[start_idx:end_idx]

        return ''.join(lines)

    except Exception as e:
        raise RuntimeError(f"Failed to read {filepath}: {e}")

def create_file(filepath: str, content: str, stream_callback=None) -> bool:
    """Create a new file with content.

    Args:
        filepath: Path to file (relative or absolute)
        content: File contents
        stream_callback: Optional callback(chunk) for streaming display

    Returns:
        True if successful

    Raises:
        FileExistsError: If file already exists
        ValueError: If path is unsafe
        RuntimeError: If creation fails
    """
    # Validate path
    resolved = _validate_path(filepath, "write")

    if resolved.exists():
        raise FileExistsError(f"File already exists: {filepath}. Use edit_file to modify it.")

    try:
        # Create parent directories
        parent = resolved.parent
        if parent:
            parent.mkdir(parents=True, exist_ok=True)

        # Take snapshot for rollback
        safety = get_safety()
        safety.snapshot(str(resolved))

        with open(resolved, 'w', encoding='utf-8') as f:
            if stream_callback:
                # Stream content in chunks
                chunk_size = 50
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i+chunk_size]
                    f.write(chunk)
                    stream_callback(chunk)

                    import time
                    time.sleep(0.01)
            else:
                f.write(content)

        return True

    except Exception as e:
        raise RuntimeError(f"Failed to create {filepath}: {e}")

def edit_file(filepath: str, target_block: str, replacement_block: str, stream_callback=None) -> bool:
    """Surgically replace a code block in an existing file.

    Args:
        filepath: Path to file (relative or absolute)
        target_block: Exact text to find and replace
        replacement_block: New text to insert
        stream_callback: Optional callback(chunk) for streaming display

    Returns:
        True if successful

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If target_block not found, appears multiple times, or path is unsafe
        RuntimeError: If edit fails
    """
    # Validate path
    resolved = _validate_path(filepath, "write")

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # Take snapshot for rollback
        safety = get_safety()
        safety.snapshot(str(resolved))

        with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        def _normalize_whitespace(text: str) -> str:
            """Normalize whitespace for fuzzy matching."""
            lines = text.splitlines()
            # Remove empty lines at start and end
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()

            # Normalize indentation (remove common prefix)
            if lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
                lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]

            return '\n'.join(lines)

        # Try exact match first
        count = content.count(target_block)
        if count == 0:
            # Try normalized match
            normalized_content = _normalize_whitespace(content)
            normalized_target = _normalize_whitespace(target_block)
            count = normalized_content.count(normalized_target)

            if count == 0:
                raise ValueError(
                    f"Target block not found in {filepath}.\n"
                    f"Expected:\n{target_block}\n\n"
                    f"Hint: Check whitespace and indentation."
                )
            elif count > 1:
                raise ValueError(f"Target block appears {count} times in {filepath}. Make it more specific.")

            # Use normalized versions for replacement
            content = content.replace(normalized_target, replacement_block, 1)
        elif count > 1:
            raise ValueError(f"Target block appears {count} times in {filepath}. Make it more specific.")
        else:
            # Exact match found
            content = content.replace(target_block, replacement_block, 1)

        import difflib

        diff = list(difflib.unified_diff(
            target_block.splitlines(keepends=True),
            replacement_block.splitlines(keepends=True),
            fromfile=f"a/{resolved.name}",
            tofile=f"b/{resolved.name}",
            n=3
        ))

        if diff:
            try:
                from nare.cli.display import ui
                from rich.text import Text

                diff_text = Text()
                for line in diff:
                    if line.startswith('+') and not line.startswith('+++'):
                        diff_text.append(line, style="green")
                    elif line.startswith('-') and not line.startswith('---'):
                        diff_text.append(line, style="red")
                    elif line.startswith('@@'):
                        diff_text.append(line, style="cyan")
                    else:
                        diff_text.append(line)

                ui.console.print()
                ui.console.print(f"[bold yellow]Proposed changes for {resolved.name}:[/]")
                ui.console.print(diff_text)

                from nare.cli.modes import get_mode_config
                ask = get_mode_config().ask_before_edit

                if ask:
                    from nare.cli.display.thinking import get_thinking_display
                    get_thinking_display()._stop_live_and_spinner()

                    from rich.prompt import Confirm
                    if not Confirm.ask(f"[bold yellow]Apply these changes to {resolved.name}?[/]"):
                        raise RuntimeError("User rejected the changes.")
            except ImportError:
                # CLI not available, proceed without confirmation
                pass

        new_content = content.replace(target_block, replacement_block)

        with open(resolved, 'w', encoding='utf-8') as f:
            if stream_callback:
                # Stream content in chunks
                chunk_size = 50
                for i in range(0, len(replacement_block), chunk_size):
                    chunk = replacement_block[i:i+chunk_size]
                    stream_callback(chunk)
                    import time
                    time.sleep(0.01)
                f.write(new_content)
            else:
                f.write(new_content)

        return True

    except Exception as e:
        raise RuntimeError(f"Failed to edit {filepath}: {e}")

def list_files(directory: str = ".", pattern: str = "*") -> list:
    """List files in a directory matching a pattern.

    Args:
        directory: Directory to search (default: current)
        pattern: Glob pattern (default: all files)

    Returns:
        List of file paths
    """
    import glob

    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")

    search_path = os.path.join(directory, "**", pattern)
    files = glob.glob(search_path, recursive=True)

    files = [f for f in files if os.path.isfile(f)]

    return sorted(files)

def run_command(command: str, cwd: str = ".") -> dict:
    """Run a shell command and return output.

    Args:
        command: Command to execute
        cwd: Working directory (default: current)

    Returns:
        Dict with stdout, stderr, returncode

    Raises:
        ValueError: If command is blocked by safety layer
        RuntimeError: If command fails
    """
    import subprocess

    # Validate command with safety layer
    safety = get_safety()
    ok, reason = safety.check_command(command)
    if not ok:
        raise ValueError(f"Command blocked: {reason}")

    # Log warning if command is potentially dangerous
    if reason:
        import logging
        logging.warning(f"Running potentially dangerous command: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after 300s: {command}")
    except Exception as e:
        raise RuntimeError(f"Failed to run command: {e}")
