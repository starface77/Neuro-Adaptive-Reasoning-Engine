"""Unified Repository Map Generator with Semantic Skeleton support."""

import os
import subprocess
from nare.utils.logger import get_logger
import time
from typing import Optional, Set
from collections import defaultdict

log = get_logger("nare.core.repo_map")

SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env", ".env",
    "node_modules", ".tox", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".eggs", "*.egg-info", ".nare_memory",
    "memory_store", ".idea", ".vscode", "coverage",
}

SKIP_EXTS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".svg",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".mp4", ".avi", ".mov", ".mp3", ".wav",
}

PRIORITY_DIRS = {
    "src", "lib", "core", "app", "api", "server",
    "client", "components", "services", "models",
    "views", "controllers",
}

_cache: dict = {}
_cache_time: dict = {}
_cache_ttl: float = 15.0


def generate_repo_map(
    repo_path: str,
    max_files: int = 200,
    max_chars: int = 3000,
    use_cache: bool = True,
    active_files: Optional[Set[str]] = None,
) -> str:
    """Generate a semantic skeleton of the repository.

    Only includes priority directories, active files, and a compact
    summary of the rest.  Keeps output under max_chars to minimize
    token usage.
    """
    cache_key = f"{repo_path}:{max_files}:{max_chars}"
    if use_cache and cache_key in _cache:
        if (time.time() - _cache_time.get(cache_key, 0)) < _cache_ttl:
            return _cache[cache_key]

    active_files = active_files or set()
    tree = []

    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        files = [f for f in result.stdout.strip().split("\n") if f]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        files = _walk_files(repo_path, max_files)

    if not files:
        return f"{os.path.basename(repo_path)}/ (empty)"

    tree.append(f"{os.path.basename(repo_path)}/ ({len(files)} files)")

    dir_files: dict[str, list[str]] = defaultdict(list)
    for f in files:
        parts = f.split("/")
        dir_name = parts[0] if len(parts) > 1 else ""
        dir_files[dir_name].append(f)

    priority = [d for d in PRIORITY_DIRS if d in dir_files]
    other = [d for d in sorted(dir_files.keys()) if d not in PRIORITY_DIRS and d != ""]
    files_per_dir = max(3, max_files // (len(dir_files) + 1))

    if "" in dir_files:
        for f in sorted(dir_files[""])[:files_per_dir]:
            tree.append(f"├── {f}")

    for dir_name in priority:
        tree.append(f"├── {dir_name}/")
        for f in sorted(dir_files[dir_name])[:files_per_dir]:
            filename = f.split("/", 1)[1] if "/" in f else f
            tree.append(f"│   ├── {filename}")

    shown = len(priority)
    max_other = max(5, 10 - shown)
    for dir_name in other[:max_other]:
        count = len(dir_files[dir_name])
        if count <= 3:
            tree.append(f"├── {dir_name}/")
            for f in sorted(dir_files[dir_name]):
                filename = f.split("/", 1)[1] if "/" in f else f
                tree.append(f"│   ├── {filename}")
        else:
            tree.append(f"├── {dir_name}/ ({count} files)")

    remaining = len(other) - max_other
    if remaining > 0:
        tree.append(f"... ({remaining} more directories)")

    for af in sorted(active_files):
        if not any(af in line for line in tree):
            tree.append(f"├── {af}")

    output = "\n".join(tree)
    if len(output) > max_chars:
        output = output[:max_chars] + "\n... (truncated)"

    _cache[cache_key] = output
    _cache_time[cache_key] = time.time()
    return output


def _walk_files(repo_path: str, max_files: int) -> list[str]:
    """Fallback file listing via os.walk."""
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not os.path.islink(os.path.join(root, d))]
        for fname in filenames:
            if any(fname.endswith(ext) for ext in SKIP_EXTS):
                continue
            rel = os.path.relpath(os.path.join(root, fname), repo_path)
            files.append(rel)
            if len(files) >= max_files:
                return files
    return files


def clear_cache():
    """Clear the repo map cache."""
    global _cache, _cache_time
    _cache.clear()
    _cache_time.clear()
