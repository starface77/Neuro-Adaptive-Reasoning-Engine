#!/usr/bin/env python3
"""Find syntax errors in Python files."""

import sys
import ast
from pathlib import Path


def check_file(filepath):
    """Check if file has syntax errors."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()

        ast.parse(code)
        return None
    except SyntaxError as e:
        return {
            'file': str(filepath),
            'line': e.lineno,
            'msg': e.msg,
            'text': e.text
        }
    except Exception as e:
        return {
            'file': str(filepath),
            'line': 0,
            'msg': str(e),
            'text': ''
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_syntax.py <directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)

    errors = []

    if path.is_file():
        if path.suffix == '.py':
            err = check_file(path)
            if err:
                errors.append(err)
    elif path.is_dir():
        for filepath in path.rglob('*.py'):
            if '__pycache__' in str(filepath):
                continue
            err = check_file(filepath)
            if err:
                errors.append(err)

    if errors:
        print(f"Found {len(errors)} files with syntax errors:\n")
        for err in errors:
            print(f"ERROR: {err['file']}:{err['line']}")
            print(f"  {err['msg']}")
            if err['text']:
                print(f"  {err['text'].strip()}")
            print()
    else:
        print("OK: No syntax errors found")


if __name__ == '__main__':
    main()
