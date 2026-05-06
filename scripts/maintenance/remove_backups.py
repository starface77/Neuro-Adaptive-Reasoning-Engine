#!/usr/bin/env python3
"""Remove all .bak backup files.

Usage:
    python remove_backups.py <directory>
    python remove_backups.py nare/
    python remove_backups.py .
"""

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_backups.py <directory>")
        print()
        print("Examples:")
        print("  python remove_backups.py nare/")
        print("  python remove_backups.py .")
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
        return

    print(f"Found {len(bak_files)} .bak files in {path}")
    print()

    # Delete them
    for bak_file in bak_files:
        try:
            bak_file.unlink()
            print(f"✓ Deleted {bak_file}")
        except Exception as e:
            print(f"✗ Failed to delete {bak_file}: {e}")

    print()
    print(f"Done! Deleted {len(bak_files)} backup files")


if __name__ == '__main__':
    main()
