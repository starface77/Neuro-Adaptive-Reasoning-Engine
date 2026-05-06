#!/usr/bin/env python3
"""Auto-fix common syntax errors caused by comment removal."""

import re
from pathlib import Path


def fix_file(filepath):
    """Fix common syntax errors in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]

            # Fix: unterminated triple-quoted string - add opening """
            # Pattern: line starts with text that looks like it should be in a string
            if i > 0 and not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                prev = lines[i-1].strip()
                # Check if previous line ends with = or : and current line looks like string content
                if (prev.endswith('=') or prev.endswith(':')) and line.strip() and not line.strip().startswith('#'):
                    # Check if this looks like start of a docstring/prompt
                    if any(keyword in line.lower() for keyword in ['you are', 'task:', 'your job', 'important:', 'rules:', 'format:']):
                        # Add opening """
                        fixed_lines.append('"""')
                        fixed_lines.append(line)
                        i += 1
                        continue

            # Fix: invalid character (em dash) - replace with regular dash
            if '—' in line or '—' in line:
                line = line.replace('—', '-').replace('—', '-')

            # Fix: CREATE TABLE without opening """
            if 'CREATE TABLE' in line and i > 0:
                prev = lines[i-1].strip()
                if prev.endswith('=') or prev.endswith('"""'):
                    if not prev.endswith('"""'):
                        fixed_lines.append('"""')

            fixed_lines.append(line)
            i += 1

        # Write fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))

        print(f"Fixed: {filepath}")
        return True

    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def main():
    # List of broken files from check_syntax.py
    broken_files = [
        "nare/execution/local.py",
        "nare/interface/cli/autopilot.py",
        "nare/interface/daemon/task_queue.py",
        "nare/interface/cli/display/theme.py",
        "nare/interface/cli/utils/system_prompt.py",
        "nare/core/evolution/learning.py",
        "nare/core/routing/router.py",
        "nare/agents/loops/autonomous.py",
        "nare/agents/loops/synthesis.py",
        "nare/agents/roles/analyzer.py",
        "nare/agents/roles/coder.py",
        "nare/agents/roles/critic.py",
    ]

    print("Auto-fixing broken files...\n")

    fixed = 0
    for filepath in broken_files:
        path = Path(filepath)
        if path.exists():
            if fix_file(path):
                fixed += 1
        else:
            print(f"Not found: {filepath}")

    print(f"\nFixed {fixed}/{len(broken_files)} files")
    print("\nRun: python scripts/maintenance/check_syntax.py nare/")
    print("to verify all errors are fixed")


if __name__ == '__main__':
    main()
