"""Safety layer for NARE agent tool execution.

Guards against destructive operations:
  - Protected paths (.git, .env, secrets) cannot be written or deleted.
  - Dangerous shell commands (rm -rf, dd, mkfs, etc.) are blocked outright
    or require explicit confirmation before running.
  - Rollback hooks: file snapshots before write/edit so operations can be
    undone.

Usage
-----
    from nare.tools.safety import SafetyLayer

    safety = SafetyLayer(working_dir="/path/to/repo")

    # Before a bash call:
    ok, reason = safety.check_command("rm -rf /tmp/test")
    if not ok:
        return ToolResult(ok=False, error=reason)

    # Before write_file / edit_file:
    ok, reason = safety.check_path_write("/path/to/.env")
    if not ok:
        return ToolResult(ok=False, error=reason)

    # Snapshot before editing (enables rollback):
    safety.snapshot("/path/to/file.py")
    # ... edit ...
    # On failure:
    safety.rollback("/path/to/file.py")
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("nare.tools.safety")

PROTECTED_PATH_PATTERNS: List[str] = [
    ".git",
    ".env",
    ".env.local",
    ".env.production",
    ".env.secret",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_ed25519",
    "authorized_keys",
    "known_hosts",
]

BLOCKED_COMMAND_PATTERNS: List[str] = [
    r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f",
    r"rm\s+-[a-zA-Z]*f[a-zA-Z]*r",
    r">\s*/dev/sd[a-z]",
    r"mkfs\b",
    r"dd\s+.*of=/dev/",
    r":.*\{.*:.*\|.*&.*\}.*;",
    r"wget\s+.*\|\s*sh",
    r"curl\s+.*\|\s*sh",
    r"curl\s+.*\|\s*bash",
    r"chmod\s+777\s+/",
    r"chown\s+.*:\s*root\s+/",
    r"shutdown\b",
    r"reboot\b",
    r"halt\b",
    r"poweroff\b",
]

WARNING_COMMAND_PATTERNS: List[str] = [
    r"\brm\b",
    r"\bmv\b.*\.\.\.",
    r"\btruncate\b",
    r"\bdrop\s+table\b",
    r"\bdrop\s+database\b",
    r"\bdelete\s+from\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-fd\b",
    r"\bgit\s+push\s+.*--force\b",
    r"\bnpm\s+publish\b",
    r"\bpip\s+uninstall\b",
]

_BLOCKED_RE = [re.compile(p, re.IGNORECASE) for p in BLOCKED_COMMAND_PATTERNS]
_WARNING_RE = [re.compile(p, re.IGNORECASE) for p in WARNING_COMMAND_PATTERNS]

class SafetyLayer:
    """Central safety gate for all agent write/exec operations.

    Parameters
    ----------
    working_dir:
        The agent's workspace root. Paths outside this root require
        explicit opt-in (``allow_outside_workdir``).
    allow_outside_workdir:
        If True, allow writing/reading paths outside ``working_dir``.
        Default False (safer).
    extra_protected_patterns:
        Additional glob-style patterns to protect (appended to the
        built-in list).
    """

    def __init__(
        self,
        working_dir: str = ".",
        allow_outside_workdir: bool = False,
        extra_protected_patterns: Optional[List[str]] = None,
    ) -> None:
        self.working_dir = Path(working_dir).resolve()
        self.allow_outside_workdir = allow_outside_workdir
        self._extra_patterns: List[str] = extra_protected_patterns or []

        self._snapshots: Dict[str, str] = {}

    def check_path_write(self, path: str) -> Tuple[bool, str]:
        """Return (allowed, reason).

        Blocks writes to:
        1. Protected path patterns (.git, .env, private keys, …)
        2. Paths outside the working directory (unless opted-in)
        """
        resolved = Path(path).resolve()

        if not self.allow_outside_workdir:
            try:
                resolved.relative_to(self.working_dir)
            except ValueError:
                return (
                    False,
                    f"Safety: path '{path}' is outside the working directory "
                    f"'{self.working_dir}'. Refusing write.",
                )

        all_patterns = PROTECTED_PATH_PATTERNS + self._extra_patterns
        for pattern in all_patterns:
            if _path_matches_pattern(resolved, pattern):
                return (
                    False,
                    f"Safety: '{path}' matches protected pattern '{pattern}'. "
                    "Refusing write to sensitive file.",
                )

        return True, ""

    def check_path_delete(self, path: str) -> Tuple[bool, str]:
        """Same as check_path_write but for deletions."""
        ok, reason = self.check_path_write(path)
        if not ok:
            return ok, reason.replace("Refusing write", "Refusing delete")
        return True, ""

    def check_command(self, command: str) -> Tuple[bool, str]:
        """Return (allowed, reason).

        Blocks unconditionally dangerous commands.
        Returns a warning message in *reason* for moderate-risk commands
        even when ``allowed=True``.
        """
        stripped = command.strip()

        for rx in _BLOCKED_RE:
            if rx.search(stripped):
                msg = (
                    f"Safety: command blocked — matches dangerous pattern "
                    f"'{rx.pattern}'. Command was: {stripped!r}"
                )
                log.warning(msg)
                return False, msg

        for rx in _WARNING_RE:
            if rx.search(stripped):
                msg = (
                    f"Safety warning: command matches potentially destructive "
                    f"pattern '{rx.pattern}'. Proceeding with caution."
                )
                log.warning(msg)

                return True, msg

        return True, ""

    def snapshot(self, path: str) -> bool:
        """Copy ``path`` to a temp file before modification.

        Returns True if a snapshot was created, False if the file did
        not exist (nothing to snapshot).
        """
        resolved = str(Path(path).resolve())
        if not os.path.isfile(resolved):
            return False

        try:
            tmp_fd, tmp_path = tempfile.mkstemp(prefix="nare_snap_")
            os.close(tmp_fd)
            shutil.copy2(resolved, tmp_path)
            self._snapshots[resolved] = tmp_path
            log.debug("Snapshot: %s → %s", resolved, tmp_path)
            return True
        except OSError as exc:
            log.warning("Snapshot failed for %s: %s", resolved, exc)
            return False

    def rollback(self, path: str) -> bool:
        """Restore ``path`` from its snapshot.

        Returns True if rollback succeeded, False if no snapshot exists.
        """
        resolved = str(Path(path).resolve())
        tmp_path = self._snapshots.get(resolved)
        if tmp_path is None:
            log.warning("Rollback: no snapshot for %s", resolved)
            return False

        try:
            shutil.copy2(tmp_path, resolved)
            os.unlink(tmp_path)
            del self._snapshots[resolved]
            log.info("Rollback: restored %s", resolved)
            return True
        except OSError as exc:
            log.error("Rollback failed for %s: %s", resolved, exc)
            return False

    def clear_snapshots(self) -> None:
        """Delete all temp snapshot files (call after successful operation)."""
        for tmp_path in self._snapshots.values():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        self._snapshots.clear()

    def pre_write(self, path: str) -> Tuple[bool, str]:
        """Check path safety and take a snapshot. Returns (allowed, reason)."""
        ok, reason = self.check_path_write(path)
        if not ok:
            return False, reason
        self.snapshot(path)
        return True, reason

    def pre_bash(self, command: str) -> Tuple[bool, str]:
        """Check command safety. Returns (allowed, reason/warning)."""
        return self.check_command(command)

def _path_matches_pattern(resolved: Path, pattern: str) -> bool:
    """Return True if any component of ``resolved`` matches the pattern."""
    import fnmatch

    for part in resolved.parts:
        if fnmatch.fnmatch(part, pattern):
            return True

    if fnmatch.fnmatch(resolved.name, pattern):
        return True

    return False

_default_safety: Optional[SafetyLayer] = None

def get_safety(working_dir: Optional[str] = None) -> SafetyLayer:
    """Return (or lazily create) the module-level SafetyLayer.

    If ``working_dir`` is provided the layer is (re)initialised with
    that directory. Subsequent calls without ``working_dir`` return the
    cached instance.
    """
    global _default_safety
    if working_dir is not None or _default_safety is None:
        _default_safety = SafetyLayer(working_dir=working_dir or ".")
    return _default_safety
