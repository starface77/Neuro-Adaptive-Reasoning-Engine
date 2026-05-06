"""
Unified Repository Map Generator

Single source of truth for generating repo maps across the entire system.
Used by PlanningAgent, AgentLoop, and CLI.

Design:
- Uses `git ls-files` for tracked repos (fast)
- Falls back to os.walk for non-git directories
- Configurable limits (max_files, max_chars)
- Smart truncation with priority for important directories
- 15-second cache to prevent disk hammering
"""

import os
import subprocess
from nare.utils.logger import get_logger
import time
from typing import Optional, Set
from collections import defaultdict

log = get_logger("nare.core.repo_map")

# Directories to skip
SKIP_DIRS = {
    '__pycache__', '.git', '.venv', 'venv', 'env', '.env',
    'node_modules', '.tox', '.mypy_cache', '.pytest_cache',
    'dist', 'build', '.eggs', '*.egg-info', '.nare_memory',
    'memory_store', '.idea', '.vscode', 'coverage',
}

# File extensions to skip
SKIP_EXTS = {
    '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib', '.exe',
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.svg',
    '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
    '.mp4', '.avi', '.mov', '.mp3', '.wav',
}

# Priority directories (shown first, not truncated)
PRIORITY_DIRS = {
    'src', 'lib', 'core', 'app', 'api', 'server', 'client',
    'components', 'services', 'models', 'views', 'controllers',
}

# Global cache
_cache: dict = {}
_cache_time: dict = {}
_cache_ttl: float = 15.0  # seconds


def generate_repo_map(
    repo_path: str,
    max_files: int = 1500,
    max_chars: int = 15000,
    use_cache: bool = True,
) -> str:
    """Generate a tree-like map of the repository.

    Parameters
    ----------
    repo_path : str
        Absolute path to the repository root.
    max_files : int
        Maximum number of files to include (default: 1500).
    max_chars : int
        Truncate output to this many characters (default: 15000).
    use_cache : bool
        Use cached result if available (default: True).

    Returns
    -------
    str
        Tree representation of the repository structure.

    Examples
    --------
    >>> map_str = generate_repo_map("/path/to/repo", max_files=500, max_chars=5000)
    >>> print(map_str)
    my-project/ (git tree)
    ├── src/
    │   ├── main.py
    │   └── utils.py
    ├── tests/
    │   └── test_main.py
    └── README.md
    """
    # Check cache
    cache_key = f"{repo_path}:{max_files}:{max_chars}"
    if use_cache and cache_key in _cache:
        if (time.time() - _cache_time.get(cache_key, 0)) < _cache_ttl:
            return _cache[cache_key]

    tree = []

    # Try fast path: git ls-files
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        files = result.stdout.strip().split('\n')
        if files and files[0]:
            tree.append(f"{os.path.basename(repo_path)}/ (git tree)")

            def build_tree_from_files(files: list, max_files: int) -> list:
                """Build tree structure from flat file list."""
                tree_lines = []
                dirs_seen = set()

                # Group files by directory
                dir_files = {}
                for f in files:
                    parts = f.split('/')
                    if len(parts) == 1:
                        # File in root
                        dir_files.setdefault('', []).append(f)
                    else:
                        # File in subdirectory
                        dir_name = parts[0]
                        dir_files.setdefault(dir_name, []).append(f)

                # Prioritize important directories
                priority_dirs = [d for d in PRIORITY_DIRS if d in dir_files]
                other_dirs = [d for d in sorted(dir_files.keys()) if d not in PRIORITY_DIRS and d != '']

                # Build tree
                files_per_dir = max(3, max_files // (len(dir_files) + 1))

                # Root files first
                if '' in dir_files:
                    for f in sorted(dir_files[''])[:files_per_dir]:
                        tree_lines.append(f"├── {f}")

                # Priority directories
                for dir_name in priority_dirs:
                    tree_lines.append(f"├── {dir_name}/")
                    for f in sorted(dir_files[dir_name])[:files_per_dir]:
                        filename = f.split('/', 1)[1] if '/' in f else f
                        tree_lines.append(f"│   ├── {filename}")

                # Other directories
                for dir_name in other_dirs[:10]:
                    tree_lines.append(f"├── {dir_name}/")
                    for f in sorted(dir_files[dir_name])[:files_per_dir]:
                        filename = f.split('/', 1)[1] if '/' in f else f
                        tree_lines.append(f"│   ├── {filename}")

                return tree_lines

            tree.extend(build_tree_from_files(files, max_files))

            total_files = len(files)
            shown_files = len([line for line in tree if '├──' in line and not line.endswith('/')])

            if total_files > shown_files:
                tree.append(f"... ({total_files - shown_files} files hidden)")

            output = "\n".join(tree)

            # Truncate if too long
            if len(output) > max_chars:
                output = output[:max_chars] + "\n... (truncated)"

            # Cache result
            _cache[cache_key] = output
            _cache_time[cache_key] = time.time()

            return output

    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.debug(f"[RepoMap] git ls-files failed: {e}, falling back to os.walk")

    # Fallback to os.walk
    def walk_dir(current_dir: str, prefix: str = "", depth: int = 0, max_depth: int = 5):
        if depth > max_depth:
            return

        try:
            entries = sorted(os.listdir(current_dir))
        except (PermissionError, OSError):
            return

        dirs = []
        files = []

        for e in entries:
            if e in SKIP_DIRS:
                continue

            path = os.path.join(current_dir, e)

            # Skip symlinks to prevent infinite loops
            if os.path.islink(path):
                continue

            if os.path.isdir(path):
                dirs.append(e)
            else:
                # Skip binary/media files
                if any(e.endswith(ext) for ext in SKIP_EXTS):
                    continue
                files.append(e)

        # Prioritize important directories
        priority_dirs = [d for d in dirs if d in PRIORITY_DIRS]
        other_dirs = [d for d in dirs if d not in PRIORITY_DIRS]
        sorted_dirs = priority_dirs + other_dirs

        for i, d in enumerate(sorted_dirs):
            if len(tree) >= max_files:
                tree.append(f"{prefix}... (max files reached)")
                return

            is_last_dir = (i == len(sorted_dirs) - 1) and not files
            marker = "└── " if is_last_dir else "├── "
            tree.append(f"{prefix}{marker}{d}/")

            extension = "    " if is_last_dir else "│   "
            walk_dir(os.path.join(current_dir, d), prefix + extension, depth + 1, max_depth)

        for i, f in enumerate(files):
            if len(tree) >= max_files:
                tree.append(f"{prefix}... (max files reached)")
                return

            is_last_file = (i == len(files) - 1)
            marker = "└── " if is_last_file else "├── "
            tree.append(f"{prefix}{marker}{f}")

    tree.append(f"{os.path.basename(repo_path)}/")
    walk_dir(repo_path)

    output = "\n".join(tree)

    # Truncate if too long
    if len(output) > max_chars:
        output = output[:max_chars] + "\n... (truncated)"

    # Cache result
    _cache[cache_key] = output
    _cache_time[cache_key] = time.time()

    return output


def clear_cache():
    """Clear the repo map cache."""
    global _cache, _cache_time
    _cache.clear()
    _cache_time.clear()
