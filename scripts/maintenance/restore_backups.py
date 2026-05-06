#!/usr/bin/env python3
"""Restore all files from .bak backups.

Usage:
    python restore_backups.py <directory>
"""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python restore_backups.py <directory>")
        print()
        print("Examples:")
        print("  python restore_backups.py nare/")
        print("  python restore_backups.py .")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)

    if not path.is_dir():
        print(f"Error: {path} is not a directory")
        sys.exit(1)

    # Find all .bak files
    bak_files = list(path.rglob('*.bak'))

    if not bak_files:
        print(f"No .bak files found in {path}")
        print("Cannot restore - backups already deleted!")
        return

    print(f"Found {len(bak_files)} .bak files in {path}")
    print("Restoring...")
    print()

    # Restore them
    restored = 0
    for bak_file in bak_files:
        try:
            # Get original filename (remove .bak)
            original = bak_file.with_suffix('')

            # Rename .bak back to original
            bak_file.rename(original)
            print(f"✓ Restored {original}")
            restored += 1
        except Exception as e:
            print(f"✗ Failed to restore {bak_file}: {e}")

    print()
    print(f"Done! Restored {restored} files")


if __name__ == '__main__':
    main()
