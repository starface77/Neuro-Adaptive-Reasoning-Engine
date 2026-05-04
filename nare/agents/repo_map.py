"""
Repository Map Generator

Generates a compact, LLM-friendly representation of a project's
file structure. Used by the PlanningAgent to understand what files
exist before proposing edits.

Design: Uses `git ls-files` for tracked repos, falls back to os.walk.
Output is a tree string capped at ~4000 chars to fit in prompts.
"""

import os
import subprocess
import logging
from typing import Optional
from collections import defaultdict

log = logging.getLogger("nare.agents.repo_map")

_SKIP_DIRS = {
    '__pycache__', '.git', '.venv', 'venv', 'node_modules',
    '.tox', '.mypy_cache', '.pytest_cache', 'dist', 'build',
    '.eggs', '*.egg-info', '.vare_memory', 'memory_store',
}

_CODE_EXTS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java',
    '.c', '.cpp', '.h', '.hpp', '.rb', '.php', '.swift', '.kt',
    '.yaml', '.yml', '.toml', '.json', '.md', '.txt', '.cfg',
    '.ini', '.sh', '.bat', '.ps1', '.sql', '.html', '.css',
}

def generate_repo_map(
    repo_path: str,
    max_depth: int = 3,
    max_files: int = 300,
    max_chars: int = 4000,
) -> str:
    """Generate a tree-like map of the repository.

    Parameters
    ----------
    repo_path : str
        Absolute path to the repository root.
    max_depth : int
        Maximum directory depth to traverse.
    max_files : int
        Maximum number of files to include.
    max_chars : int
        Truncate output to this many characters.

    Returns
    -------
    str
        A compact tree string suitable for LLM prompts.
    """
    files = _get_file_list(repo_path)

    if not files:
        return "(empty repository)"

    tree = defaultdict(list)
    for f in files[:max_files]:
        parts = f.replace('\\', '/').split('/')
        if len(parts) > max_depth + 1:

            key = '/'.join(parts[:max_depth])
            tree[key].append('.../' + parts[-1])
        elif len(parts) > 1:
            key = '/'.join(parts[:-1])
            tree[key].append(parts[-1])
        else:
            tree['.'].append(parts[0])

    lines = [f"Repository: {os.path.basename(repo_path)}/"]
    lines.append(f"({len(files)} tracked files)\n")

    for directory in sorted(tree.keys()):
        lines.append(f"  {directory}/")
        file_list = sorted(tree[directory])
        for fname in file_list[:15]:
            lines.append(f"    {fname}")
        if len(file_list) > 15:
            lines.append(f"    ... +{len(file_list) - 15} more")

    result = '\n'.join(lines)

    if len(result) > max_chars:
        result = result[:max_chars - 20] + "\n  ... (truncated)"

    return result

def _get_file_list(repo_path: str) -> list:
    """Get list of relevant files, preferring git ls-files."""

    try:
        result = subprocess.run(
            ['git', 'ls-files'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            files = result.stdout.strip().split('\n')

            return [
                f for f in files
                if _is_relevant(f) and '/test' not in f.lower()
            ]
    except Exception as e:
        log.debug(f"git ls-files failed: {e}")

    files = []
    for root, dirs, filenames in os.walk(repo_path):

        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith('.')]

        rel_root = os.path.relpath(root, repo_path)
        depth = rel_root.count(os.sep)
        if depth > 4:
            dirs.clear()
            continue

        for fname in filenames:
            rel = os.path.join(rel_root, fname).replace('\\', '/')
            if rel.startswith('./'):
                rel = rel[2:]
            if _is_relevant(rel):
                files.append(rel)
                if len(files) >= 500:
                    return files

    return files

def _is_relevant(path: str) -> bool:
    """Check if a file path is relevant for code analysis."""
    _, ext = os.path.splitext(path)
    if ext.lower() in _CODE_EXTS:
        return True

    basename = os.path.basename(path)
    if basename in ('Makefile', 'Dockerfile', 'Procfile', 'Gemfile',
                    'Rakefile', 'LICENSE', 'MANIFEST.in'):
        return True
    return False
