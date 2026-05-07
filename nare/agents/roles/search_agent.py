"""Sub-agent for file search delegation.

Offloads grep/find operations from the main context, returning only
the minimal result (file path + line number) to the primary agent.
"""

import os
import subprocess
from typing import List, Dict, Optional
from nare.utils.logger import get_logger

log = get_logger("nare.agents.roles.search_agent")

MAX_RESULTS = 10
MAX_LINE_LENGTH = 200


def search_files(
    pattern: str,
    repo_path: str,
    file_glob: Optional[str] = None,
    max_results: int = MAX_RESULTS,
) -> List[Dict[str, str]]:
    """Search for a pattern in the repository using grep.

    Returns a compact list of matches: [{file, line, text}].
    All search overhead stays outside the main agent context.
    """
    cmd = ["grep", "-rn", "--include=*.py", "-m", str(max_results)]
    if file_glob:
        cmd[3] = f"--include={file_glob}"
    cmd.extend([pattern, repo_path])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=repo_path,
        )
        matches = []
        for line in result.stdout.strip().split("\n"):
            if not line or ":" not in line:
                continue
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            filepath, lineno, text = parts[0], parts[1], parts[2]
            rel_path = os.path.relpath(filepath, repo_path) if os.path.isabs(filepath) else filepath
            matches.append({
                "file": rel_path,
                "line": lineno,
                "text": text.strip()[:MAX_LINE_LENGTH],
            })
            if len(matches) >= max_results:
                break
        return matches
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        log.warning(f"Search failed: {e}")
        return []


def find_function(
    name: str,
    repo_path: str,
    language: str = "python",
) -> List[Dict[str, str]]:
    """Find function/class definition by name.

    Returns compact location info without loading entire files.
    """
    if language == "python":
        pattern = rf"(def|class)\s+{name}"
    else:
        pattern = name

    return search_files(pattern, repo_path, file_glob="*.py")


def find_file(
    filename: str,
    repo_path: str,
) -> List[str]:
    """Find files by name pattern."""
    cmd = ["find", repo_path, "-name", filename, "-not", "-path", "*/__pycache__/*", "-not", "-path", "*/.git/*"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        paths = [
            os.path.relpath(p.strip(), repo_path)
            for p in result.stdout.strip().split("\n")
            if p.strip()
        ]
        return paths[:MAX_RESULTS]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        log.warning(f"Find failed: {e}")
        return []


def compact_search_result(matches: List[Dict[str, str]]) -> str:
    """Format search results for injection into the main agent context."""
    if not matches:
        return "No matches found."
    lines = []
    for m in matches:
        lines.append(f"{m['file']}:{m['line']}: {m['text']}")
    return "\n".join(lines)
