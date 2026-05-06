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
import sys
import time
import asyncio
from pathlib import Path
from typing import List, Optional

from .base import Tool, ToolParam, ToolRegistry, ToolResult, ToolError
from .web_search import web_search
from ..safety import get_safety
from ..hunks import HunkParser, HunkApplier

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
        
    if b'\0' in raw[:1024]:
        return f"<binary file omitted: {path}>"
        
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

async def read_file(
    path: str,
    working_dir: Optional[str] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
) -> ToolResult:
    """Read file contents, optionally with offset/limit for large files.

    For large files (>100 lines), prefer reading specific line ranges:
    - offset: starting line number (0-indexed)
    - limit: number of lines to read

    Example: read_file("big.py", offset=50, limit=20) reads lines 50-69
    """
    full = _resolve(path, working_dir)
    if not os.path.exists(full):
        return ToolResult(ok=False, error=f"file not found: {path}")
    if os.path.isdir(full):
        return ToolResult(ok=False, error=f"is a directory, not a file: {path}")

    body = _safe_read_text(full)
    lines = body.splitlines(keepends=True)
    total_lines = len(lines)

    # Auto-limit large files to first 100 lines if no offset/limit specified
    if offset is None and limit is None and total_lines > 100:
        lines = lines[:100]
        body = "".join(lines)
        summary = f"{len(lines)} lines (showing first 100 of {total_lines} total - use offset/limit for more)"
    elif offset is not None or limit is not None:
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

async def write_file(
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

async def edit_file(
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
    # Validate arguments
    if not old:
        return ToolResult(ok=False, error="`old` text cannot be empty")

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

async def batch_edit(
    edits: list,
    *,
    working_dir: Optional[str] = None,
) -> ToolResult:
    """Execute multiple file edits in a single operation.
    
    `edits` is a list of dicts: {"path": str, "old": str, "new": str}
    """
    if not isinstance(edits, list):
        return ToolResult(ok=False, error="`edits` must be a list of dicts")
        
    results = []
    errors = []
    total_additions = 0
    total_deletions = 0
    
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            errors.append(f"Edit #{i}: Not a dictionary")
            continue
            
        path = edit.get("path")
        old = edit.get("old")
        new = edit.get("new")
        
        if not path or old is None or new is None:
            errors.append(f"Edit #{i}: Missing path, old, or new")
            continue
            
        res = await edit_file(path=path, old=old, new=new, working_dir=working_dir, replace_all=True)
        if res.ok:
            results.append(f"[{path}] {res.summary}")
            if res.meta:
                total_additions += res.meta.get("additions", 0)
                total_deletions += res.meta.get("deletions", 0)
        else:
            errors.append(f"[{path}] Failed: {res.error}")
            
    if errors and not results:
        return ToolResult(ok=False, error="\n".join(errors))
        
    summary = f"Edited {len(results)} files (+{total_additions} -{total_deletions})"
    body = "\n".join(results)
    if errors:
        summary += f" (with {len(errors)} errors)"
        body += "\n\nERRORS:\n" + "\n".join(errors)
        
    return ToolResult(
        ok=len(errors) == 0,
        summary=summary,
        body=body,
        meta={"files_edited": len(results), "errors": len(errors)}
    )

async def edit_lines(
    path: str,
    start_line: int,
    end_line: int,
    new_content: str,
    *,
    working_dir: Optional[str] = None,
) -> ToolResult:
    """Edit specific lines in a file (hunk-based editing).

    Args:
        path: File path
        start_line: First line to replace (1-indexed)
        end_line: Last line to replace (1-indexed, inclusive)
        new_content: New content for those lines
        working_dir: Working directory

    Example:
        edit_lines("Hero.vue", 56, 56, "    <div class='pt-56'>")
    """
    full = _resolve(path, working_dir)
    if not os.path.exists(full):
        return ToolResult(ok=False, error=f"file not found: {path}")

    safety = get_safety(working_dir)
    ok, reason = safety.pre_write(full)
    if not ok:
        return ToolResult(ok=False, error=reason)

    lines = _safe_read_text(full).splitlines(keepends=True)
    total_lines = len(lines)

    if start_line < 1 or start_line > total_lines:
        return ToolResult(ok=False, error=f"start_line {start_line} out of range (1-{total_lines})")
    if end_line < start_line or end_line > total_lines:
        return ToolResult(ok=False, error=f"end_line {end_line} invalid (must be {start_line}-{total_lines})")

    # Convert to 0-indexed
    start_idx = start_line - 1
    end_idx = end_line

    # Build old content for diff
    old_lines = lines[start_idx:end_idx]
    old_content = "".join(old_lines)

    # Ensure new_content ends with newline if original did
    if old_lines and old_lines[-1].endswith('\n') and not new_content.endswith('\n'):
        new_content += '\n'

    # Replace lines
    new_lines = lines[:start_idx] + [new_content] + lines[end_idx:]
    updated = "".join(new_lines)

    Path(full).write_text(updated, encoding="utf-8")
    safety.clear_snapshots()

    additions = new_content.count("\n") + (1 if new_content and not new_content.endswith("\n") else 0)
    deletions = len(old_lines)

    diff_body = (
        f"@@ {path}:{start_line}-{end_line} @@\n"
        + "".join(f"-{line.rstrip()}\n" for line in old_lines)
        + "".join(f"+{line.rstrip()}\n" for line in new_content.splitlines())
    )

    return ToolResult(
        ok=True,
        summary=f"Edited lines {start_line}-{end_line}: +{additions} -{deletions}",
        body=diff_body,
        body_lang="diff",
        meta={
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "additions": additions,
            "deletions": deletions,
            "abs_path": full,
        },
    )

async def bash(
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

    # Windows platform compatibility: translate Unix commands
    original_command = command
    if sys.platform == "win32":
        command = _translate_unix_to_windows(command)

    cwd = working_dir or os.getcwd()
    started = time.time()
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ToolResult(
                ok=False,
                error=f"command timed out after {timeout}s",
                meta={"command": original_command, "timeout": True},
            )
        
        output = (stdout.decode('utf-8', errors='replace') or "").rstrip()
        err_out = stderr.decode('utf-8', errors='replace').rstrip()
        if err_out:
            output = (output + "\n" + err_out).strip()
            
        return_code = proc.returncode
    except FileNotFoundError as e:
        return ToolResult(ok=False, error=f"shell not found: {e}")

    duration = time.time() - started
    return ToolResult(
        ok=return_code == 0,
        summary=None if (output and return_code == 0) else (
            "Done" if return_code == 0 else f"exit {return_code}"
        ),
        body=output or None,
        meta={
            "command": original_command,
            "translated_command": command if command != original_command else None,
            "exit_code": return_code,
            "duration": duration,
        },
    )


def _translate_unix_to_windows(command: str) -> str:
    """
    Robust Unix-to-Windows command translation with proper shell parsing.

    Handles pipes, redirections, and complex command structures.
    """
    import shlex

    # Handle pipes and redirections
    if '|' in command:
        # Split by pipe and translate each part
        parts = command.split('|')
        translated_parts = [_translate_single_command(part.strip()) for part in parts]
        return ' | '.join(translated_parts)

    # Handle redirections
    if '>' in command or '<' in command:
        # Parse redirection
        import re
        match = re.match(r'(.+?)\s*(>>?|<)\s*(.+)', command)
        if match:
            cmd_part = match.group(1).strip()
            redir_op = match.group(2)
            redir_target = match.group(3).strip()
            translated_cmd = _translate_single_command(cmd_part)
            return f"{translated_cmd} {redir_op} {redir_target}"

    return _translate_single_command(command)

def _translate_single_command(cmd: str) -> str:
    """Translate a single Unix command to Windows."""
    import re

    cmd = cmd.strip()

    # find . -name 'pattern' or find . -name "pattern"
    match = re.match(r"find\s+\.\s+-name\s+['\"]([^'\"]+)['\"]", cmd)
    if match:
        pattern = match.group(1)
        pattern = pattern.strip('*')
        return f'dir /s/b *{pattern}*'

    # find . -type f (files only)
    if re.match(r"find\s+\.\s+-type\s+f", cmd):
        return 'dir /s/b /a-d'

    # find . -type d (directories only)
    if re.match(r"find\s+\.\s+-type\s+d", cmd):
        return 'dir /s/b /ad'

    # Simple find . (list all)
    if cmd == "find .":
        return 'dir /s/b'

    # ls with arguments
    if cmd.startswith("ls ") or cmd == "ls":
        args = cmd[2:].strip() if len(cmd) > 2 else ""
        return _translate_ls_args(args)

    # grep pattern file → findstr pattern file
    match = re.match(r"grep\s+(.+)", cmd)
    if match:
        rest = match.group(1)
        return _translate_grep_args(rest)

    # cat file → type file
    if cmd.startswith("cat "):
        return cmd.replace("cat", "type", 1)

    # rm file → del file
    if cmd.startswith("rm "):
        args = cmd[3:].strip()
        return _translate_rm_args(args)

    # cp source dest → copy source dest
    if cmd.startswith("cp "):
        return cmd.replace("cp", "copy", 1)

    # mv source dest → move source dest
    if cmd.startswith("mv "):
        return cmd.replace("mv", "move", 1)

    # pwd → cd
    if cmd == "pwd":
        return "cd"

    # No translation needed
    return cmd

def _translate_ls_args(args: str) -> str:
    """Translate ls arguments to dir equivalents."""
    if not args or args == ".":
        return "dir"
    elif "-la" in args or "-al" in args:
        return "dir /a"
    elif "-l" in args:
        return "dir"
    else:
        return f"dir {args}"

def _translate_rm_args(args: str) -> str:
    """Translate rm arguments to del equivalents."""
    # Remove common flags that don't translate
    args = args.replace("-f", "").replace("-r", "/s").strip()
    return f"del {args}"

def _translate_grep_args(args: str) -> str:
    """Translate grep arguments to findstr equivalents."""
    # Basic translation - findstr has different syntax
    return f"findstr {args}"

def _grep_sync(
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

async def grep(
    pattern: str,
    *,
    path: str = ".",
    working_dir: Optional[str] = None,
    glob: Optional[str] = None,
    case_insensitive: bool = False,
    max_matches: int = 100,
) -> ToolResult:
    return await asyncio.to_thread(
        _grep_sync, pattern, path=path, working_dir=working_dir, 
        glob=glob, case_insensitive=case_insensitive, max_matches=max_matches
    )

def _list_dir_sync(
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

async def list_dir(
    path: str = ".",
    *,
    working_dir: Optional[str] = None,
    show_hidden: bool = False,
) -> ToolResult:
    return await asyncio.to_thread(_list_dir_sync, path, working_dir=working_dir, show_hidden=show_hidden)

def _find_files_sync(
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

async def find_files(
    glob: str,
    *,
    path: str = ".",
    working_dir: Optional[str] = None,
    max_results: int = 200,
) -> ToolResult:
    return await asyncio.to_thread(_find_files_sync, glob, path=path, working_dir=working_dir, max_results=max_results)

def _human_bytes(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n / 1024:.1f}{unit}"
        n //= 1024
    return f"{n}T"

async def git_status(working_dir: Optional[str] = None) -> ToolResult:
    return await bash("git status --short --branch", working_dir=working_dir, timeout=10)

def _compile_skills_from_memory_sync(
    min_uses: int = 2,
    max_skills: int = 5,
    working_dir: Optional[str] = None,
) -> ToolResult:
    """Analyze episodic memory and compile important patterns into skills.

    This tool allows the AI to autonomously create skills from successful patterns.
    """
    try:
        # Import here to avoid circular dependencies
        from ...core.agent import NAREProductionAgent
        from ...config import NareConfig
        import os

        # Load memory from working directory
        persist_dir = os.path.join(working_dir or ".", ".nare_memory")
        if not os.path.exists(persist_dir):
            return ToolResult(
                ok=False,
                error="No memory found. Execute some tasks first to build memory."
            )

        # Create temporary agent to access memory
        agent = NAREProductionAgent(
            config=NareConfig(),
            persist_dir=persist_dir,
            embedding_dim=1024,
        )

        # Load memory from disk
        agent.memory.load()

        # Analyze episodes for patterns
        episodes = agent.memory.episodes
        if len(episodes) < min_uses:
            return ToolResult(
                ok=False,
                error=f"Not enough episodes ({len(episodes)}) to compile skills. Need at least {min_uses}."
            )

        # Find patterns (simple heuristic: group by similar queries)
        from collections import Counter

        # Extract keywords from queries
        patterns = []
        for ep in episodes:
            query = ep.get('query', '').lower()
            # Simple keyword extraction
            words = [w for w in query.split() if len(w) > 3]
            if words:
                patterns.append(' '.join(words[:3]))  # First 3 words as pattern

        # Count pattern frequency
        pattern_counts = Counter(patterns)
        common_patterns = [p for p, count in pattern_counts.most_common(max_skills) if count >= min_uses]

        if not common_patterns:
            return ToolResult(
                ok=False,
                error=f"No patterns found with {min_uses}+ occurrences."
            )

        # Compile skills for common patterns
        compiled_count = 0
        for pattern in common_patterns:
            # Find episodes matching this pattern
            matching_eps = [ep for ep in episodes if pattern in ep.get('query', '').lower()]

            if matching_eps:
                # Generate meaningful skill name using LLM
                queries_sample = '\n'.join([f"- {ep.get('query', '')}" for ep in matching_eps[:5]])
                naming_prompt = f"""Analyze these similar user requests and create a short, descriptive skill name (2-4 words, snake_case).

{queries_sample}

The skill name should capture the SPECIFIC action, not just generic terms.
Examples:
- "создай компонент Button" → "create_vue_component" (NOT "add_new_component")
- "увеличь padding в style.css" → "increase_css_padding" (NOT "modify_css")
- "добавь API endpoint /users" → "add_api_endpoint" (NOT "add_endpoint")

Output ONLY the skill name in snake_case. No explanations, just the name."""

                try:
                    from ...reasoning import llm
                    samples, _ = llm.generate_samples(naming_prompt, n=1, temperature=0.3, mode="DIRECT")
                    if samples and len(samples) > 0 and isinstance(samples[0], dict):
                        skill_name = samples[0].get('solution', pattern.replace(' ', '_')).strip()
                        # Clean up the name
                        skill_name = skill_name.lower().replace(' ', '_').replace('-', '_')
                        # Remove any non-alphanumeric characters except underscore
                        import re
                        skill_name = re.sub(r'[^a-z0-9_]', '', skill_name)
                    else:
                        skill_name = pattern.replace(' ', '_')
                except Exception:
                    skill_name = pattern.replace(' ', '_')

                # Use first matching episode as template
                ep = matching_eps[0]

                # Extract actions from episode
                actions = ep.get('actions', [])

                # Generate executable code based on actions
                action_code = []
                imports = set()

                for action in actions:
                    tool = action.get('tool', '')
                    args = action.get('args', {})

                    if tool == 'Write':
                        imports.add('from pathlib import Path')
                        file_path = args.get('file_path', '')
                        content = args.get('content', '')

                        # Check if content is a placeholder - if so, generate full template
                        if len(content) < 100 or '<template>...</template>' in content:
                            # Generate full Vue component template
                            action_code.append(f'''
    # Extract component name from query
    import re
    words = query.split()
    component_name = words[-1] if words else "Component"
    if component_name.endswith('.vue'):
        component_name = component_name[:-4]

    # Generate full Vue component
    file_path = f"web/src/components/{{component_name}}.vue"

    vue_template = f"""<template>
  <div class="{{component_name.lower()}}">
    <h2>{{{{title}}}}</h2>
    <p>{{{{description}}}}</p>
  </div>
</template>

<script>
import {{{{ ref }}}} from 'vue'

export default {{{{
  name: '{{component_name}}',
  setup() {{{{
    const title = ref('{{component_name}}')
    const description = ref('Component description')

    return {{{{
      title,
      description
    }}}}
  }}}}
}}}}
</script>

<style scoped>
.{{component_name.lower()}} {{{{
  padding: 2rem;
}}}}

.{{component_name.lower()}} h2 {{{{
  font-size: 1.5rem;
  margin-bottom: 1rem;
}}}}
</style>
"""

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(vue_template)
    result += f"Created {{file_path}}\\n"''')
                        else:
                            # Use actual content from episode
                            action_code.append(f'''
    # Write file
    file_path = query.split()[-1] if len(query.split()) > 0 else "{file_path}"
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("""{content}""")
    result += f"Created {{file_path}}\\n"''')

                    elif tool == 'Edit':
                        imports.add('from pathlib import Path')
                        file_path = args.get('file_path', '')
                        old_string = args.get('old_string', '')
                        new_string = args.get('new_string', '')
                        action_code.append(f'''
    # Edit file
    file_path = "{file_path}"
    if Path(file_path).exists():
        content = Path(file_path).read_text(encoding='utf-8')
        content = content.replace("""{old_string}""", """{new_string}""")
        Path(file_path).write_text(content, encoding='utf-8')
        result += f"Updated {{file_path}}\\n"''')

                imports_str = '\n'.join(sorted(imports)) if imports else ''
                actions_str = '\n'.join(action_code) if action_code else '    result = f"Executed skill: {skill_name}"'

                # Create executable skill
                skill_code = f'''"""Auto-compiled skill from pattern analysis.

Pattern: {pattern}
Occurrences: {len(matching_eps)}
Skill: {skill_name}
"""

{imports_str}

def trigger(query: str) -> bool:
    """Check if this skill should handle the query."""
    keywords = {repr(pattern.split())}
    query_lower = query.lower()
    return all(kw in query_lower for kw in keywords)

def execute(query: str, context: dict = None) -> str:
    """Execute the compiled skill."""
    result = ""
    try:
{actions_str}
        return result if result else f"Executed skill: {skill_name}"
    except Exception as e:
        return f"Error: {{e}}"
'''

                # Save skill
                import numpy as np

                # Check if skill with this name already exists
                existing_patterns = [s.get('pattern', '') for s in agent.memory.compiled_skills]
                if skill_name in existing_patterns:
                    # Skip duplicate
                    continue

                query_emb = llm.get_embedding(pattern)
                trigger_emb = np.array(query_emb, dtype=np.float32)

                agent.memory.add_compiled_skill(
                    pattern=skill_name,
                    code=skill_code,
                    trigger_emb=trigger_emb
                )

                # Add metadata
                if agent.memory.compiled_skills:
                    agent.memory.compiled_skills[-1]['confidence'] = 0.70
                    agent.memory.compiled_skills[-1]['source'] = 'ai_compiled'
                    agent.memory.compiled_skills[-1]['occurrences'] = len(matching_eps)

                compiled_count += 1

        # Save to disk
        agent.memory.force_save()

        summary = f"Compiled {compiled_count} skills from {len(episodes)} episodes"
        body = "Skills compiled:\n"
        for i, pattern in enumerate(common_patterns[:compiled_count], 1):
            count = pattern_counts[pattern]
            body += f"{i}. {pattern} ({count} uses)\n"

        return ToolResult(
            ok=True,
            summary=summary,
            body=body,
            meta={"compiled": compiled_count, "patterns": common_patterns[:compiled_count]}
        )

    except Exception as e:
        return ToolResult(
            ok=False,
            error=f"Failed to compile skills: {e}"
        )

async def compile_skills_from_memory(
    min_uses: int = 2,
    max_skills: int = 5,
    working_dir: Optional[str] = None,
) -> ToolResult:
    return await asyncio.to_thread(_compile_skills_from_memory_sync, min_uses, max_skills, working_dir)

async def update_todos(items: list, *, working_dir: Optional[str] = None) -> ToolResult:
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
        name="batch_edit",
        description="Replace exact text in multiple files simultaneously. Saves massive amounts of tokens for bulk updates (e.g. imports).",
        parameters=[
            ToolParam("edits", "array", "List of objects: {'path': '...', 'old': '...', 'new': '...'}"),
        ],
        run=_wd(batch_edit),
        display_verb="Batch Edit",
        requires_confirmation=True,
    ))
    reg.register(Tool(
        name="edit_lines",
        description=(
            "Edit specific lines in a file (hunk-based editing). "
            "Use this instead of edit_file when you know the exact line numbers. "
            "Saves 90%+ tokens by avoiding full file reads/writes."
        ),
        parameters=[
            ToolParam("path", "string", "Path relative to the working directory"),
            ToolParam("start_line", "integer", "First line to replace (1-indexed)"),
            ToolParam("end_line", "integer", "Last line to replace (1-indexed, inclusive)"),
            ToolParam("new_content", "string", "New content for those lines"),
        ],
        run=_wd(edit_lines),
        display_verb="Edit",
        requires_confirmation=True,
    ))

    # Import hunk tools
    from .hunk_tool import apply_hunks
    from .hunk_search_tool import search_for_hunk, find_function_for_hunk, find_class_for_hunk

    reg.register(Tool(
        name="search_for_hunk",
        description=(
            "Search for a pattern in a file and return context for hunk application. "
            "Use this BEFORE apply_hunks to find line numbers without reading entire file. "
            "Saves 90%+ tokens. Returns hunk template ready to modify."
        ),
        parameters=[
            ToolParam("pattern", "string", "Text or regex pattern to search for"),
            ToolParam("file_path", "string", "Path to file (relative to working directory)"),
            ToolParam("context_lines", "integer", "Number of context lines before/after (default: 3)", required=False, default=3),
            ToolParam("is_regex", "boolean", "Whether pattern is a regex (default: False)", required=False, default=False),
        ],
        run=_wd(search_for_hunk),
        display_verb="Search",
    ))

    reg.register(Tool(
        name="find_function",
        description=(
            "Find a function definition and return context for hunk application. "
            "Faster than search_for_hunk for known function names. "
            "Returns hunk template ready to modify."
        ),
        parameters=[
            ToolParam("function_name", "string", "Name of the function to find"),
            ToolParam("file_path", "string", "Path to file (relative to working directory)"),
            ToolParam("context_lines", "integer", "Number of context lines (default: 5)", required=False, default=5),
        ],
        run=_wd(find_function_for_hunk),
        display_verb="Find",
    ))

    reg.register(Tool(
        name="find_class",
        description=(
            "Find a class definition and return context for hunk application. "
            "Faster than search_for_hunk for known class names. "
            "Returns hunk template ready to modify."
        ),
        parameters=[
            ToolParam("class_name", "string", "Name of the class to find"),
            ToolParam("file_path", "string", "Path to file (relative to working directory)"),
            ToolParam("context_lines", "integer", "Number of context lines (default: 5)", required=False, default=5),
        ],
        run=_wd(find_class_for_hunk),
        display_verb="Find",
    ))

    reg.register(Tool(
        name="apply_hunks",
        description=(
            "Apply unified diff hunks to files. Most efficient for multiple changes. "
            "Format: <<<<<<< path\\n@@ -line,count +line,count @@\\n context\\n-removed\\n+added\\n>>>>>>>. "
            "Validates context before applying. Use for complex multi-line edits."
        ),
        parameters=[
            ToolParam("hunks", "string", "Unified diff format hunks (see tool description for format)"),
        ],
        run=_wd(apply_hunks),
        display_verb="Patch",
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
        def _run_code_sandbox_sync(code: str, language: str = "python", stdin: str = "", packages: str = "", **kwargs) -> ToolResult:
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
            
        async def run_code_sandbox(*args, **kwargs) -> ToolResult:
            return await asyncio.to_thread(_run_code_sandbox_sync, *args, **kwargs)

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

    reg.register(Tool(
        name="compile_skills",
        description=(
            "Analyze episodic memory and compile the most important, frequently used patterns into reusable skills. "
            "Use this when asked to 'compile skills', 'save important patterns', or 'create skills from history'. "
            "The system will analyze recent successful tasks and create deterministic skills for repeated patterns."
        ),
        parameters=[
            ToolParam("min_uses", "integer", "Minimum times a pattern must appear to be compiled (default: 2)", required=False, default=2),
            ToolParam("max_skills", "integer", "Maximum number of skills to compile (default: 5)", required=False, default=5),
        ],
        run=compile_skills_from_memory,
        display_verb="Compile skills",
    ))

    return reg
