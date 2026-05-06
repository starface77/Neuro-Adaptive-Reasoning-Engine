#!/usr/bin/env python3
"""Remove ALL comments from Python files (including docstrings).

Usage:
    python strip_all_comments.py <file_or_directory>

This is aggressive - removes EVERYTHING:
- Single-line comments (#)
- Multi-line comments (triple quotes)
- ALL docstrings (module, class, function)
- Keeps only executable code

Creates .bak backups automatically.
"""

import os
import sys
import re
from pathlib import Path


def strip_all_comments(code: str) -> str:
    """Remove ALL comments and docstrings from Python code.

    Args:
        code: Python source code

    Returns:
        Code with only executable statements
    """
    lines = code.split('\n')
    result = []
    in_string = False
    string_char = None
    string_start_line = None

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        # Keep shebang
        if i == 0 and stripped.startswith('#!'):
            result.append(line)
            continue

        # Handle multi-line strings
        if in_string:
            # Look for closing quote
            if string_char in line:
                # Count quotes
                count = line.count(string_char)
                if count % 2 == 1:  # Odd = closing
                    in_string = False
                    string_char = None
                    # Skip this line (end of string)
                    continue
            # Skip lines inside string
            continue

        # Check for string start (''' or """)
        if '"""' in stripped or "'''" in stripped:
            string_char = '"""' if '"""' in stripped else "'''"
            count = line.count(string_char)
            if count == 2:
                # Opens and closes on same line - skip it
                continue
            elif count == 1:
                # Opens only - start skipping
                in_string = True
                string_start_line = i
                continue

        # Remove inline comments
        if '#' in line:
            # Check if # is in a string literal
            in_str = False
            str_char = None
            new_line = []

            for j, char in enumerate(line):
                # Track string boundaries
                if char in ('"', "'") and (j == 0 or line[j-1] != '\\'):
                    if not in_str:
                        in_str = True
                        str_char = char
                    elif char == str_char:
                        in_str = False
                        str_char = None

                # Stop at # outside string
                if char == '#' and not in_str:
                    break

                new_line.append(char)

            line = ''.join(new_line).rstrip()

        # Skip empty lines
        if not line.strip():
            continue

        result.append(line)

    # Remove trailing empty lines
    while result and not result[-1].strip():
        result.pop()

    return '\n'.join(result)


def process_file(filepath: Path):
    """Process a single file."""
    try:
        # Read original
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()

        # Strip comments
        cleaned = strip_all_comments(original)

        # Create backup
        backup = filepath.with_suffix(filepath.suffix + '.bak')
        with open(backup, 'w', encoding='utf-8') as f:
            f.write(original)

        # Write cleaned
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        # Stats
        orig_lines = len(original.split('\n'))
        new_lines = len(cleaned.split('\n'))
        saved = orig_lines - new_lines

        print(f"✓ {filepath}")
        print(f"  {orig_lines} → {new_lines} lines ({saved} removed)")

    except Exception as e:
        print(f"✗ {filepath}: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python strip_all_comments.py <file_or_directory>")
        print()
        print("Examples:")
        print("  python strip_all_comments.py nare/")
        print("  python strip_all_comments.py nare/cli/session.py")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)

    print(f"Stripping ALL comments from: {path}")
    print(f"Creating .bak backups")
    print()

    if path.is_file():
        if path.suffix == '.py':
            process_file(path)
        else:
            print(f"Error: Not a Python file")
            sys.exit(1)
    elif path.is_dir():
        count = 0
        for filepath in path.rglob('*.py'):
            if '__pycache__' in str(filepath) or '.git' in str(filepath):
                continue
            process_file(filepath)
            count += 1
        print()
        print(f"Processed {count} files")
    else:
        print(f"Error: Not a file or directory")
        sys.exit(1)

    print()
    print("Done! Backups saved as .bak files")
    print("To restore: mv file.py.bak file.py")


if __name__ == '__main__':
    main()
