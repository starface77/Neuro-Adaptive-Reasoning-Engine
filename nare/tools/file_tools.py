"""File system tools for autonomous LLM operations."""

import os
from pathlib import Path
from typing import Optional, Tuple

# Import safety layer for path validation
from nare.tools.safety import get_safety

MAX_UNCHUNKED_LINES = 50


def _validate_path(filepath: str, operation: str = "read") -> Path:
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
    """Read file contents with enforced chunking."""
    resolved = _validate_path(filepath, "read")

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        total = len(lines)

        if start_line is not None or end_line is not None:
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else total
            if start_idx < 0 or end_idx > total or start_idx >= end_idx:
                raise ValueError(f"Invalid line range: {start_line}-{end_line} (file has {total} lines)")
            lines = lines[start_idx:end_idx]

        if total > MAX_UNCHUNKED_LINES and start_line is None and end_line is None:
            preview = ''.join(lines[:20])
            return (
                f"[CHUNKING REQUIRED] File has {total} lines. "
                f"Use start_line/end_line or grep to read specific sections.\n\n"
                f"Preview (lines 1-20):\n{preview}"
            )

        return ''.join(lines)

    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to read {filepath}: {e}")

def create_file(filepath: str, content: str, stream_callback=None) -> bool:
    """Create a new file with content."""
    resolved = _validate_path(filepath, "write")

    if resolved.exists():
        raise FileExistsError(f"File already exists: {filepath}. Use edit_file to modify it.")

    try:
        parent = resolved.parent
        if parent:
            parent.mkdir(parents=True, exist_ok=True)

        safety = get_safety()
        safety.snapshot(str(resolved))

        with open(resolved, 'w', encoding='utf-8') as f:
            if stream_callback:
                chunk_size = 50
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
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
    """Surgically replace a code block in an existing file."""
    resolved = _validate_path(filepath, "write")

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        safety = get_safety()
        safety.snapshot(str(resolved))

        with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        def _normalize_whitespace(text: str) -> str:
            lines = text.splitlines()
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            if lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
                lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]
            return '\n'.join(lines)

        count = content.count(target_block)
        if count == 0:
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
            content = content.replace(normalized_target, replacement_block, 1)
        elif count > 1:
            raise ValueError(f"Target block appears {count} times in {filepath}. Make it more specific.")
        else:
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
    """List files in a directory matching a pattern."""
    import glob

    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")

    search_path = os.path.join(directory, "**", pattern)
    files = glob.glob(search_path, recursive=True)

    files = [f for f in files if os.path.isfile(f)]

    return sorted(files)

def run_command(command: str, cwd: str = ".") -> dict:
    """Run a shell command and return output."""
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
