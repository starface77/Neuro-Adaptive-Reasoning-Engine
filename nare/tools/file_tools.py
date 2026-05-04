"""
File system tools for autonomous LLM operations.

Provides safe, validated file operations that the LLM can call:
- read_file: Read file contents with optional line range
- create_file: Create new files with parent directories
- edit_file: Surgically replace code blocks in existing files
"""

import os
from typing import Optional, Tuple


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
        ValueError: If line range is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # If file is large and no range specified, warn
        if len(lines) > 500 and start_line is None and end_line is None:
            return f"[WARNING] File has {len(lines)} lines. Please specify start_line and end_line to read a specific range.\n\nFirst 50 lines:\n" + ''.join(lines[:50])

        # Apply line range if specified
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
        RuntimeError: If creation fails
    """
    if os.path.exists(filepath):
        raise FileExistsError(f"File already exists: {filepath}. Use edit_file to modify it.")

    try:
        # Create parent directories if needed
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Write file with streaming
        with open(filepath, 'w', encoding='utf-8') as f:
            if stream_callback:
                # Stream content in chunks for display
                chunk_size = 50  # characters per chunk
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i+chunk_size]
                    f.write(chunk)
                    stream_callback(chunk)
                    # Small delay for visual effect
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
        ValueError: If target_block not found or appears multiple times
        RuntimeError: If edit fails
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # Read current content
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Validate target block exists exactly once
        count = content.count(target_block)
        if count == 0:
            raise ValueError(f"Target block not found in {filepath}. Make sure the text matches exactly.")
        if count > 1:
            raise ValueError(f"Target block appears {count} times in {filepath}. Make it more specific.")

        # Show beautiful diff
        import difflib
        
        diff = list(difflib.unified_diff(
            target_block.splitlines(keepends=True),
            replacement_block.splitlines(keepends=True),
            fromfile=f"a/{os.path.basename(filepath)}",
            tofile=f"b/{os.path.basename(filepath)}",
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
                ui.console.print(f"[bold yellow]Proposed changes for {os.path.basename(filepath)}:[/]")
                ui.console.print(diff_text)
                
                # Check if we should ask for confirmation
                from nare.cli.modes import get_mode_config
                ask = get_mode_config().ask_before_edit
                
                if ask:
                    from nare.cli.display.thinking import get_thinking_display
                    get_thinking_display()._stop_live_and_spinner()
                    
                    from rich.prompt import Confirm
                    if not Confirm.ask(f"[bold yellow]Apply these changes to {os.path.basename(filepath)}?[/]"):
                        raise RuntimeError("User rejected the changes.")
            except ImportError:
                # Fallback for headless testing
                pass

        # Perform replacement
        new_content = content.replace(target_block, replacement_block)

        # Write back with streaming
        with open(filepath, 'w', encoding='utf-8') as f:
            if stream_callback:
                # Stream only the replacement block for display
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

    # Filter out directories
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
        RuntimeError: If command fails
    """
    import subprocess

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
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
