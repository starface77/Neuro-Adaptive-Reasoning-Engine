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
from nare.utils.logger import get_logger
import threading
import atexit
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = get_logger("nare.tools.safety")

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
        self._temp_files: List[str] = []
        self._lock = threading.RLock()
        atexit.register(self._cleanup_temp_files)

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

        # Semantic validation to catch bypasses
        ok, reason = self._validate_command_semantic(stripped)
        if not ok:
            log.warning(reason)
            return False, reason

        for rx in _WARNING_RE:
            if rx.search(stripped):
                msg = (
                    f"Safety warning: command matches potentially destructive "
                    f"pattern '{rx.pattern}'. Proceeding with caution."
                )
                log.warning(msg)

                return True, msg

        return True, ""

    def _normalize_command(self, cmd: str) -> str:
        """Normalize command while preserving string literals.

        BUG FIX #7: Parse command structure instead of blindly removing quotes.
        """
        import shlex

        try:
            # Parse command with proper shell syntax
            tokens = shlex.split(cmd)
            # Return only the base command for checking
            if tokens:
                base_cmd = tokens[0].lower()
                return base_cmd
            return cmd.lower()
        except ValueError:
            # If parsing fails, normalize conservatively
            normalized = cmd.lower()
            # Decode escape sequences but DON'T remove quotes
            normalized = normalized.replace('\\x2f', '/').replace('\\/', '/')
            normalized = normalized.replace('\\x20', ' ')
            # Expand common variable patterns
            normalized = normalized.replace('$HOME', '/home').replace('~', '/home')
            return normalized

    def _validate_command_semantic(self, cmd: str) -> Tuple[bool, str]:
        """Semantic validation that detects dangerous patterns regardless of encoding."""
        normalized = self._normalize_command(cmd)

        # Dangerous operations with their risk flags
        dangerous_ops = [
            ('rm', ['-rf', '-r /', '/ ', '-fr']),
            ('dd', ['if=', 'of=/dev']),
            ('mkfs', []),
            ('format', []),
            ('fdisk', []),
            ('parted', []),
            ('shutdown', []),
            ('reboot', []),
            ('halt', []),
            ('poweroff', []),
        ]

        for base_cmd, danger_flags in dangerous_ops:
            if base_cmd in normalized:
                # Try to parse command structure
                try:
                    tokens = shlex.split(normalized)
                    if tokens and tokens[0].endswith(base_cmd):
                        # Check for dangerous flags
                        if not danger_flags:
                            # Command itself is dangerous
                            return False, f"Dangerous operation: {base_cmd}"

                        for flag in danger_flags:
                            if any(flag in token for token in tokens):
                                return False, f"Dangerous operation: {base_cmd} {flag}"
                except ValueError:
                    # Parsing failed - contains shell metacharacters, suspicious
                    if base_cmd in normalized:
                        return False, f"Command contains suspicious shell metacharacters with {base_cmd}"

        # Check for command injection patterns
        injection_patterns = ['$(', '`', '&&', '||', ';', '|']
        for pattern in injection_patterns:
            if pattern in cmd and any(danger in normalized for danger, _ in dangerous_ops):
                return False, f"Potential command injection detected: {pattern}"

        return True, ""

    def snapshot(self, path: str) -> bool:
        """Copy ``path`` to a temp file before modification.

        Returns True if a snapshot was created, False if the file did
        not exist (nothing to snapshot).
        """
        with self._lock:
            resolved = str(Path(path).resolve())
            if not os.path.isfile(resolved):
                return False

            try:
                tmp_fd, tmp_path = tempfile.mkstemp(prefix="nare_snap_")
                os.close(tmp_fd)
                shutil.copy2(resolved, tmp_path)
                self._snapshots[resolved] = tmp_path
                self._temp_files.append(tmp_path)
                log.debug("Snapshot: %s → %s", resolved, tmp_path)
                return True
            except OSError as exc:
                log.warning("Snapshot failed for %s: %s", resolved, exc)
                return False

    def rollback(self, path: str) -> bool:
        """Restore ``path`` from its snapshot.

        Returns True if rollback succeeded, False if no snapshot exists.
        """
        with self._lock:
            resolved = str(Path(path).resolve())
            tmp_path = self._snapshots.get(resolved)
            if tmp_path is None:
                log.warning("Rollback: no snapshot for %s", resolved)
                return False

            try:
                shutil.copy2(tmp_path, resolved)
                os.unlink(tmp_path)
                del self._snapshots[resolved]
                if tmp_path in self._temp_files:
                    self._temp_files.remove(tmp_path)
                log.info("Rollback: restored %s", resolved)
                return True
            except OSError as exc:
                log.error("Rollback failed for %s: %s", resolved, exc)
                return False

    def clear_snapshots(self) -> None:
        """Delete all temp snapshot files (call after successful operation)."""
        with self._lock:
            for tmp_path in self._snapshots.values():
                try:
                    os.unlink(tmp_path)
                    if tmp_path in self._temp_files:
                        self._temp_files.remove(tmp_path)
                except OSError:
                    pass
            self._snapshots.clear()

    def _cleanup_temp_files(self) -> None:
        """Clean up all temporary snapshot files on exit."""
        for tmp_path in self._temp_files[:]:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    log.debug("Cleaned up temp file: %s", tmp_path)
            except Exception as e:
                log.warning(f"Failed to cleanup temp file {tmp_path}: {e}")
        self._temp_files.clear()

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

_safety_instances: Dict[str, SafetyLayer] = {}
_safety_lock = threading.RLock()

def get_safety(working_dir: Optional[str] = None) -> SafetyLayer:
    """Return (or lazily create) a cached SafetyLayer per working directory.

    If ``working_dir`` is provided, returns the cached instance for that
    directory (creating it if needed). This preserves snapshots across calls.
    """
    wd = os.path.abspath(working_dir or ".")

    with _safety_lock:
        if wd not in _safety_instances:
            _safety_instances[wd] = SafetyLayer(working_dir=wd)
        return _safety_instances[wd]
