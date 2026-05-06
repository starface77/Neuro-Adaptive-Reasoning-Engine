#!/usr/bin/env python3
"""Remove all comments from Python files.

Usage:
    python remove_comments.py <file_or_directory>
    python remove_comments.py nare/
    python remove_comments.py nare/cli/session.py

Features:
- Removes single-line comments (#)
- Removes multi-line comments (triple quotes)
- Preserves docstrings at module/class/function level
- Preserves shebang lines (#!/usr/bin/env python)
- Creates backup files (.bak)
"""

import os
import sys
import re
import ast
import argparse
from pathlib import Path


def remove_comments_from_code(code: str, keep_docstrings: bool = True) -> str:
    """Remove comments from Python code.

    Args:
        code: Python source code
        keep_docstrings: If True, keep module/class/function docstrings

    Returns:
        Code without comments
    """
    lines = code.split('\n')
    result = []
    in_multiline = False
    multiline_char = None
    skip_next_string = False

    # Parse AST to find docstring positions if we want to keep them
    docstring_lines = set()
    if keep_docstrings:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Find line numbers of docstring
                        if hasattr(node, 'body') and node.body:
                            first_stmt = node.body[0]
                            if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                                if hasattr(first_stmt, 'lineno'):
                                    docstring_lines.add(first_stmt.lineno)
        except:
            pass

    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()

        # Keep shebang
        if i == 1 and stripped.startswith('#!'):
            result.append(line)
            continue

        # Handle multiline strings
        if in_multiline:
            result.append(line)
            if multiline_char in line:
                # Check if it's the closing quote
                count = line.count(multiline_char)
                if count % 2 == 1:  # Odd number means closing
                    in_multiline = False
                    multiline_char = None
            continue

        # Check if this line starts a docstring we want to keep
        is_docstring = i in docstring_lines
        if is_docstring and keep_docstrings:
            result.append(line)
            # Check if multiline docstring
            if '"""' in stripped or "'''" in stripped:
                multiline_char = '"""' if '"""' in stripped else "'''"
                # Count quotes to see if it closes on same line
                count = line.count(multiline_char)
                if count == 1:  # Opening only
                    in_multiline = True
            continue

        # Remove single-line comments
        if '#' in line:
            # Check if # is inside a string
            in_string = False
            string_char = None
            new_line = []
            j = 0
            while j < len(line):
                char = line[j]

                # Handle string boundaries
                if char in ('"', "'") and (j == 0 or line[j-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None

                # If we hit # outside string, stop
                if char == '#' and not in_string:
                    break

                new_line.append(char)
                j += 1

            line = ''.join(new_line).rstrip()

        # Remove empty lines that were only comments
        if not line.strip():
            # Keep one empty line for readability
            if result and result[-1].strip():
                result.append('')
            continue

        # Remove multiline comments that are not docstrings
        if ('"""' in stripped or "'''" in stripped) and not is_docstring:
            multiline_char = '"""' if '"""' in stripped else "'''"
            # Check if it closes on same line
            count = line.count(multiline_char)
            if count == 2:  # Opens and closes on same line
                continue  # Skip this line
            elif count == 1:  # Only opens
                in_multiline = True
                continue

        result.append(line)

    # Remove trailing empty lines
    while result and not result[-1].strip():
        result.pop()

    return '\n'.join(result)


def process_file(filepath: Path, keep_docstrings: bool = True, create_backup: bool = True):
    """Process a single Python file.

    Args:
        filepath: Path to Python file
        keep_docstrings: Keep module/class/function docstrings
        create_backup: Create .bak backup file
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()

        cleaned = remove_comments_from_code(original, keep_docstrings)

        # Create backup
        if create_backup:
            backup_path = filepath.with_suffix(filepath.suffix + '.bak')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original)

        # Write cleaned code
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        print(f"✓ {filepath}")

    except Exception as e:
        print(f"✗ {filepath}: {e}")


def process_directory(dirpath: Path, keep_docstrings: bool = True, create_backup: bool = True):
    """Process all Python files in directory recursively.

    Args:
        dirpath: Path to directory
        keep_docstrings: Keep module/class/function docstrings
        create_backup: Create .bak backup files
    """
    for filepath in dirpath.rglob('*.py'):
        # Skip __pycache__ and .git
        if '__pycache__' in str(filepath) or '.git' in str(filepath):
            continue
        process_file(filepath, keep_docstrings, create_backup)


def main():
    parser = argparse.ArgumentParser(
        description='Remove comments from Python files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python remove_comments.py nare/                    # Process all files in nare/
  python remove_comments.py nare/cli/session.py      # Process single file
  python remove_comments.py nare/ --no-docstrings    # Remove docstrings too
  python remove_comments.py nare/ --no-backup        # Don't create backups
        """
    )
    parser.add_argument('path', help='File or directory to process')
    parser.add_argument('--no-docstrings', action='store_true',
                        help='Remove docstrings too (default: keep them)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Don\'t create .bak backup files')

    args = parser.parse_args()

    path = Path(args.path)
    keep_docstrings = not args.no_docstrings
    create_backup = not args.no_backup

    if not path.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)

    print(f"Removing comments from: {path}")
    print(f"Keep docstrings: {keep_docstrings}")
    print(f"Create backups: {create_backup}")
    print()

    if path.is_file():
        if path.suffix == '.py':
            process_file(path, keep_docstrings, create_backup)
        else:
            print(f"Error: {path} is not a Python file")
            sys.exit(1)
    elif path.is_dir():
        process_directory(path, keep_docstrings, create_backup)
    else:
        print(f"Error: {path} is neither file nor directory")
        sys.exit(1)

    print()
    print("Done!")
    if create_backup:
        print("Backup files created with .bak extension")
        print("To restore: mv file.py.bak file.py")


if __name__ == '__main__':
    main()
