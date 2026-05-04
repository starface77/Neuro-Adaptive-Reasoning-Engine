"""Built-in tools for the NARE agent loop.

These are intentionally small wrappers around the local filesystem and
shell. They don't escape the working directory unless the caller passes
absolute paths explicitly. The tools are stateless — anything they need
to know (working_dir, tail-N output limits) is passed in by the loop.

Each tool returns a `ToolResult` whose `meta` dict carries any data the
UI renderer needs to draw the tool block (e.g. `lines`, `exit_code`).
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from .base import Tool, ToolParam, ToolRegistry, ToolResult, ToolError
from .web_search import web_search
from ..safety import get_safety

try:
    from ...execution.docker_sandbox import execute_code as docker_execute_code
    DOCKER_AVAILABLE = True
except Exception:
    DOCKER_AVAILABLE = False

def _resolve(path: str, working_dir: Optional[str] = None) -> str:
    """Resolve `path` relative to `working_dir` (or cwd)."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    base = working_dir or os.getcwd()
    return os.path.normpath(os.path.join(base, path))

def _safe_read_text(path: str, max_bytes: int = 1_000_000) -> str:
    with open(path, "rb") as fh:
        raw = fh.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
        truncated = True
    else:
        truncated = False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    if truncated:
        text += f"\n... (truncated, file > {max_bytes} bytes)"
    return text

def read_file(
    path: str,
    working_dir: Optional[str] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
) -> ToolResult:
    """Read file contents, optionally with offset/limit for large files."""
    full = _resolve(path, working_dir)
    if not os.path.exists(full):
        return ToolResult(ok=False, error=f"file not found: {path}")
    if os.path.isdir(full):
        return ToolResult(ok=False, error=f"is a directory, not a file: {path}")

    body = _safe_read_text(full)
    lines = body.splitlines(keepends=True)
    total_lines = len(lines)

    if offset is not None or limit is not None:
        start = offset or 0
        end = start + (limit or 200) if limit is not None else len(lines)
        lines = lines[start:end]
        body = "".join(lines)

        summary = f"{len(lines)} lines (lines {start}-{end-1} of {total_lines} total)"
    else:
        summary = f"{total_lines} lines"

    return ToolResult(
        ok=True,
        summary=summary,
        body=body,
        meta={"path": path, "lines": len(lines), "total_lines": total_lines, "abs_path": full},
    )

def write_file(
    path: str,
    content: str,
    *,
    working_dir: Optional[str] = None,
    create_dirs: bool = True,
) -> ToolResult:
    full = _resolve(path, working_dir)

    safety = get_safety(working_dir)
    ok, reason = safety.pre_write(full)
    if not ok:
        return ToolResult(ok=False, error=reason)
    if reason:
        import logging
        logging.getLogger("nare.tools.builtin").warning(reason)

    if create_dirs:
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    Path(full).write_text(content, encoding="utf-8")
    safety.clear_snapshots()
    line_count = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
    return ToolResult(
        ok=True,
        summary=f"Wrote {line_count} lines to {path}",
        body=content,
        meta={"path": path, "lines": line_count, "abs_path": full},
    )

def edit_file(
    path: str,
    old: str,
    new: str,
    *,
    working_dir: Optional[str] = None,
    replace_all: bool = False,
) -> ToolResult:
    full = _resolve(path, working_dir)
    if not os.path.exists(full):
        return ToolResult(ok=False, error=f"file not found: {path}")

    safety = get_safety(working_dir)
    ok, reason = safety.pre_write(full)
    if not ok:
        return ToolResult(ok=False, error=reason)

    original = _safe_read_text(full)
    if old not in original:
        return ToolResult(
            ok=False,
            error=f"`old` text not found in {path} (be exact, including whitespace)",
        )

    if not replace_all and original.count(old) > 1:
        return ToolResult(
            ok=False,
            error=(
                f"`old` text appears {original.count(old)} times in {path}; "
                "either include more surrounding context or pass replace_all=true"
            ),
        )

    updated = original.replace(old, new) if replace_all else original.replace(old, new, 1)
    Path(full).write_text(updated, encoding="utf-8")
    safety.clear_snapshots()

    additions = new.count("\n") + (1 if new and not new.endswith("\n") else 0)
    deletions = old.count("\n") + (1 if old and not old.endswith("\n") else 0)
    diff_body = (
        f"@@ {path} @@\n"
        + "\n".join(f"-{line}" for line in old.splitlines())
        + "\n"
        + "\n".join(f"+{line}" for line in new.splitlines())
    )
    return ToolResult(
        ok=True,
        summary=f"+{additions}  -{deletions}",
        body=diff_body,
        body_lang="diff",
        meta={
            "path": path,
            "additions": additions,
            "deletions": deletions,
            "abs_path": full,
        },
    )

def bash(
    command: str,
    *,
    working_dir: Optional[str] = None,
    timeout: int = 60,
) -> ToolResult:

    safety = get_safety(working_dir)
    allowed, reason = safety.pre_bash(command)
    if not allowed:
        return ToolResult(
            ok=False,
            error=reason,
            meta={"command": command, "blocked_by_safety": True},
        )

    if reason:
        import logging
        logging.getLogger("nare.tools.builtin").warning(
            "[Safety] %s", reason
        )

    cwd = working_dir or os.getcwd()
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            ok=False,
            error=f"command timed out after {timeout}s",
            meta={"command": command, "timeout": True},
        )
    except FileNotFoundError as e:
        return ToolResult(ok=False, error=f"shell not found: {e}")

    output = (proc.stdout or "").rstrip()
    if proc.stderr:
        output = (output + "\n" + proc.stderr.rstrip()).strip()

    duration = time.time() - started
    return ToolResult(
        ok=proc.returncode == 0,
        summary=None if (output and proc.returncode == 0) else (
            "Done" if proc.returncode == 0 else f"exit {proc.returncode}"
        ),
        body=output or None,
        meta={
            "command": command,
            "exit_code": proc.returncode,
            "duration": duration,
        },
    )

def grep(
    pattern: str,
    *,
    path: str = ".",
    working_dir: Optional[str] = None,
    glob: Optional[str] = None,
    case_insensitive: bool = False,
    max_matches: int = 100,
) -> ToolResult:
    root = _resolve(path, working_dir)
    if not os.path.exists(root):
        return ToolResult(ok=False, error=f"path not found: {path}")

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return ToolResult(ok=False, error=f"invalid regex: {e}")

    matches: List[str] = []
    files_scanned = 0
    targets: List[str] = []

    if os.path.isfile(root):
        targets = [root]
    else:
        for dirpath, dirnames, filenames in os.walk(root):

            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
            ]
            for fn in filenames:
                if glob and not fnmatch.fnmatch(fn, glob):
                    continue
                targets.append(os.path.join(dirpath, fn))

    for fp in targets:
        files_scanned += 1
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if regex.search(line):
                        rel = os.path.relpath(fp, root if os.path.isdir(root) else os.path.dirname(root))
                        matches.append(f"{rel}:{i}: {line.rstrip()}")
                        if len(matches) >= max_matches:
                            break
        except (PermissionError, OSError):
            continue
        if len(matches) >= max_matches:
            break

    n = len(matches)
    summary = f"{n} match" if n == 1 else f"{n} matches"
    if files_scanned and n == 0:
        summary += f" (scanned {files_scanned} files)"
    return ToolResult(
        ok=True,
        summary=summary,
        body="\n".join(matches) if matches else None,
        meta={"matches": n, "files_scanned": files_scanned, "pattern": pattern},
    )

def list_dir(
    path: str = ".",
    *,
    working_dir: Optional[str] = None,
    show_hidden: bool = False,
) -> ToolResult:
    root = _resolve(path, working_dir)
    if not os.path.exists(root):
        return ToolResult(ok=False, error=f"path not found: {path}")
    if not os.path.isdir(root):
        return ToolResult(ok=False, error=f"not a directory: {path}")

    entries: List[str] = []
    for name in sorted(os.listdir(root)):
        if not show_hidden and name.startswith("."):
            continue
        full = os.path.join(root, name)
        if os.path.isdir(full):
            entries.append(name + "/")
        else:
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            entries.append(f"{name}  ({_human_bytes(size)})")

    return ToolResult(
        ok=True,
        summary=f"{len(entries)} entries",
        body="\n".join(entries) if entries else "(empty)",
        meta={"path": path, "entries": len(entries)},
    )

def find_files(
    glob: str,
    *,
    path: str = ".",
    working_dir: Optional[str] = None,
    max_results: int = 200,
) -> ToolResult:
    root = _resolve(path, working_dir)
    if not os.path.exists(root):
        return ToolResult(ok=False, error=f"path not found: {path}")

    results: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in {".git", "node_modules", "__pycache__", ".venv", "venv"}
        ]
        for fn in filenames:
            if fnmatch.fnmatch(fn, glob):
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                results.append(rel)
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break

    return ToolResult(
        ok=True,
        summary=f"{len(results)} {('match' if len(results) == 1 else 'matches')}",
        body="\n".join(results) if results else None,
        meta={"glob": glob, "matches": len(results)},
    )

def _human_bytes(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n / 1024:.1f}{unit}"
        n //= 1024
    return f"{n}T"

def git_status(working_dir: Optional[str] = None) -> ToolResult:
    return bash("git status --short --branch", working_dir=working_dir, timeout=10)

def update_todos(items: list, *, working_dir: Optional[str] = None) -> ToolResult:
    """Update the agent's task list.

    `items` is a list of ``{"text": str, "state": "pending"|"in_progress"|"done"}``
    dicts. The result carries the normalized list in ``meta['_todo_items']``,
    which the loop turns into a ``TodoUpdated`` event for the renderer.
    """
    if not isinstance(items, list):
        return ToolResult(ok=False, error="`items` must be a list of {text, state} dicts")

    valid_states = {"pending", "in_progress", "done"}
    normalized: list[dict] = []
    for raw in items:
        if not isinstance(raw, dict):
            return ToolResult(ok=False, error=f"todo item must be a dict, got {type(raw).__name__}")
        text = str(raw.get("text", "")).strip()
        if not text:
            return ToolResult(ok=False, error="todo item missing 'text'")
        state = str(raw.get("state", "pending")).strip().lower()
        if state not in valid_states:
            state = "pending"
        normalized.append({"text": text, "state": state})

    counts = {s: sum(1 for x in normalized if x["state"] == s) for s in valid_states}
    summary = (
        f"{len(normalized)} todos · "
        f"{counts['done']} done, {counts['in_progress']} in progress, {counts['pending']} pending"
    )
    return ToolResult(
        ok=True,
        summary=summary,
        meta={"_todo_items": normalized, "counts": counts},
    )

def build_default_registry(working_dir: Optional[str] = None) -> ToolRegistry:
    """Build a registry with all built-in tools.

    `working_dir` is captured by closure so the loop can fix the agent's
    workspace once at start-up and tools resolve paths relative to it.
    """
    reg = ToolRegistry()

    get_safety(working_dir=working_dir or ".")

    def _wd(fn):

        def wrapper(**kwargs):
            kwargs.setdefault("working_dir", working_dir)
            return fn(**kwargs)
        return wrapper

    reg.register(Tool(
        name="read_file",
        description=(
            "Read file contents. For large files (>200 lines), use offset/limit to read in chunks. "
            "Reading 500+ lines at once wastes tokens - read strategically."
        ),
        parameters=[
            ToolParam("path", "string", "Path relative to the working directory"),
            ToolParam("offset", "integer", "Line number to start from (0-indexed, optional)", required=False),
            ToolParam("limit", "integer", "Max lines to read (optional, default 200)", required=False),
        ],
        run=_wd(read_file),
        display_verb="Read",
    ))
    reg.register(Tool(
        name="write_file",
        description="Write content to a file (creates parent dirs as needed; overwrites if exists).",
        parameters=[
            ToolParam("path", "string", "Path relative to the working directory"),
            ToolParam("content", "string", "Full file contents to write"),
        ],
        run=_wd(write_file),
        display_verb="Write",
        requires_confirmation=True,
    ))
    reg.register(Tool(
        name="edit_file",
        description="Replace exact text `old` with `new` inside an existing file.",
        parameters=[
            ToolParam("path", "string", "Path relative to the working directory"),
            ToolParam("old", "string", "Exact text to find — include surrounding context for uniqueness"),
            ToolParam("new", "string", "Replacement text"),
            ToolParam("replace_all", "boolean",
                      "Replace every occurrence; otherwise the call fails if `old` appears multiple times",
                      required=False, default=False),
        ],
        run=_wd(edit_file),
        display_verb="Edit",
        requires_confirmation=True,
    ))
    reg.register(Tool(
        name="bash",
        description="Run a shell command and return its combined stdout+stderr and exit code.",
        parameters=[
            ToolParam("command", "string", "Shell command to execute"),
            ToolParam("timeout", "integer",
                      "Max seconds to wait before killing the process",
                      required=False, default=60),
        ],
        run=_wd(bash),
        display_verb="Bash",
        requires_confirmation=True,
    ))
    reg.register(Tool(
        name="grep",
        description="Search files for a Python regex pattern.",
        parameters=[
            ToolParam("pattern", "string", "Python regular expression"),
            ToolParam("path", "string",
                      "Directory or file to search (relative to working dir)",
                      required=False, default="."),
            ToolParam("glob", "string",
                      "Optional fnmatch filter, e.g. '*.py'",
                      required=False),
            ToolParam("case_insensitive", "boolean",
                      "Case-insensitive match",
                      required=False, default=False),
        ],
        run=_wd(grep),
        display_verb="Grep",
    ))
    reg.register(Tool(
        name="list_dir",
        description="List the entries in a directory.",
        parameters=[
            ToolParam("path", "string",
                      "Directory to list (relative to working dir)",
                      required=False, default="."),
            ToolParam("show_hidden", "boolean",
                      "Include dotfiles in the listing",
                      required=False, default=False),
        ],
        run=_wd(list_dir),
        display_verb="List",
    ))
    reg.register(Tool(
        name="find_files",
        description="Find files whose name matches a glob.",
        parameters=[
            ToolParam("glob", "string", "fnmatch pattern, e.g. '*.py'"),
            ToolParam("path", "string", "Root directory", required=False, default="."),
        ],
        run=_wd(find_files),
        display_verb="Find",
    ))
    reg.register(Tool(
        name="git_status",
        description="Show `git status --short --branch` for the current repo.",
        parameters=[],
        run=_wd(git_status),
        display_verb="Git",
    ))
    reg.register(Tool(
        name="update_todos",
        description=(
            "Update the task list. Call ONCE at the start to create tasks, "
            "then call again ONLY to mark tasks as 'in_progress' or 'done'. "
            "DO NOT recreate the entire list every time — keep existing tasks "
            "and only change their 'state' field. The UI renders a checklist panel."
        ),
        parameters=[
            ToolParam(
                "items",
                "array",
                'List of {"text": "...", "state": "pending|in_progress|done"} dicts. '
                'Include ALL existing tasks with updated states, not just changed ones.',
            ),
        ],
        run=update_todos,
        display_verb="Update todos",
    ))
    reg.register(Tool(
        name="web_search",
        description=(
            "Search the web for information, solutions to errors, package installation "
            "instructions, or general programming questions. Use this when you encounter "
            "errors you don't know how to fix or need up-to-date information."
        ),
        parameters=[
            ToolParam("query", "string", "Search query (e.g., 'python ModuleNotFoundError numpy solution')"),
            ToolParam("max_results", "integer", "Maximum number of results (default 5)", required=False, default=5),
        ],
        run=lambda query, max_results=5, **kwargs: ToolResult(
            ok=True,
            summary=f"Found results for: {query}",
            body=web_search(query, max_results),
            meta={"query": query, "max_results": max_results},
        ),
        display_verb="Search",
    ))

    if DOCKER_AVAILABLE:
        def run_code_sandbox(code: str, language: str = "python", stdin: str = "", packages: str = "", **kwargs) -> ToolResult:
            """Execute code in Docker sandbox."""
            packages_list = [p.strip() for p in packages.split(",")] if packages else None

            result = docker_execute_code(
                code=code,
                language=language,
                stdin=stdin or None,
                packages=packages_list,
                timeout=30,
            )

            summary = f"Exit code: {result['exit_code']}"
            if result['error']:
                summary = f"Error: {result['error']}"

            body = ""
            if result['stdout']:
                body += f"STDOUT:\n{result['stdout']}\n"
            if result['stderr']:
                body += f"STDERR:\n{result['stderr']}\n"

            return ToolResult(
                ok=result['exit_code'] == 0,
                summary=summary,
                body=body.strip() or None,
                meta={
                    'exit_code': result['exit_code'],
                    'elapsed': result['elapsed'],
                    'language': language,
                },
                error=result.get('error'),
            )

        reg.register(Tool(
            name="run_code",
            description=(
                "Execute code in a secure Docker sandbox. Supports Python, Node.js, Rust, Go, Java, C++. "
                "Use this to test code, verify solutions, or run experiments safely. "
                "The sandbox is isolated with resource limits (512MB RAM, 1 CPU, 30s timeout)."
            ),
            parameters=[
                ToolParam("code", "string", "Source code to execute"),
                ToolParam("language", "string", "Language: python, node, rust, go, java, cpp (default: python)", required=False, default="python"),
                ToolParam("stdin", "string", "Optional stdin input", required=False, default=""),
                ToolParam("packages", "string", "Comma-separated packages to install (e.g., 'numpy,pandas')", required=False, default=""),
            ],
            run=run_code_sandbox,
            display_verb="Run",
            requires_confirmation=True,
        ))

    return reg
