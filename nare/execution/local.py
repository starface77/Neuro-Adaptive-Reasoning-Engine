"""Subprocess-based sandbox for skill execution (Tier A4).

The in-process AST sandbox in :mod:`nare.sandbox` blocks every known
escape path inside CPython, but CPython is fundamentally porous: any
new dunder, any C-extension trick, any reference cycle through
``gc.get_referrers`` is a potential breakout. As soon as NARE is given
network or filesystem access, in-process is no longer enough.

This module spawns the validated skill in a **separate Python process**
with:

* AST validation still applied first (defense in depth — we never
  spawn unvalidated code);
* an empty environment dict so secrets do not leak in via env vars;
* a temporary working directory we own and can rm-rf afterwards;
* POSIX resource limits (CPU time, address space, open files,
  filesystem writes) via ``preexec_fn`` (POSIX only — on Windows the
  limits become best-effort timeouts);
* a hard wall-clock timeout enforced by ``subprocess.run``;
* IPC via JSON-on-stdin / JSON-on-stdout so we never have to ``eval``
  or ``import`` user code in the parent process.

The interface intentionally mirrors :func:`nare.sandbox.safe_execute`
so callers can swap implementations behind a single flag.
"""

from __future__ import annotations

import json
from nare.utils.logger import get_logger
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Optional

from .sandboxes.base import SecurityError, validate_code

logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS: float = 5.0
DEFAULT_MEM_LIMIT_BYTES: int = 256 * 1024 * 1024
DEFAULT_CPU_LIMIT_SECONDS: int = 5
DEFAULT_FSIZE_LIMIT_BYTES: int = 0
DEFAULT_NOFILE_LIMIT: int = 32

_CHILD_RUNNER = """
import json
import sys

# Import AST validator for defense in depth
try:
    from nare.execution.sandboxes.base import validate_code, SecurityError
    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False

req = json.loads(sys.stdin.read())
code = req["code"]
query = req["query"]
mode = req.get("mode", "execute")  # 'trigger', 'execute', or 'execute_if_trigger'

# Defense in depth: validate code in child process too
if HAS_VALIDATOR:
    try:
        validate_code(code)
    except SecurityError as e:
        print(json.dumps({"ok": False, "error": "AST validation failed: " + str(e)}))
        sys.exit(0)

# Use restricted builtins
restricted_builtins = {
    'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
    'enumerate': enumerate, 'filter': filter, 'float': float, 'int': int,
    'len': len, 'list': list, 'map': map, 'max': max, 'min': min,
    'range': range, 'reversed': reversed, 'set': set, 'sorted': sorted,
    'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
}

ns = {"__builtins__": restricted_builtins}
try:
    exec(code, ns)
except Exception as e:
    print(json.dumps({"ok": False, "error": "load: " + repr(e)}))
    sys.exit(0)

trigger_fn = ns.get("trigger")
execute_fn = ns.get("execute")
if trigger_fn is None or execute_fn is None:
    print(json.dumps({"ok": False, "error": "missing trigger or execute"}))
    sys.exit(0)

try:
    if mode == "trigger":
        result = bool(trigger_fn(query))
        print(json.dumps({"ok": True, "result": result, "kind": "bool"}))
        sys.exit(0)

    if mode == "execute_if_trigger":
        try:
            triggered = bool(trigger_fn(query))
        except Exception as e:
            print(json.dumps({"ok": False, "error": "trigger: " + repr(e)}))
            sys.exit(0)
        if not triggered:
            print(json.dumps({"ok": True, "result": "Error: trigger returned False.", "kind": "str"}))
            sys.exit(0)

    out = execute_fn(query)
    print(json.dumps({"ok": True, "result": str(out), "kind": "str"}))
except Exception as e:
    print(json.dumps({"ok": False, "error": "execute: " + repr(e)}))
"""

def _make_preexec(
    mem_limit_bytes: int,
    cpu_limit_seconds: int,
    fsize_limit_bytes: int,
    nofile_limit: int,
):
    """Return a POSIX ``preexec_fn`` that applies RLIMIT_* limits.

    On non-POSIX platforms returns ``None`` so callers fall back to the
    timeout-only enforcement.
    """
    if os.name != "posix":
        return None

    import resource

    def _apply():

        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (mem_limit_bytes, mem_limit_bytes),
            )
        except (ValueError, OSError):
            pass

        try:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (cpu_limit_seconds, cpu_limit_seconds),
            )
        except (ValueError, OSError):
            pass

        try:
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (fsize_limit_bytes, fsize_limit_bytes),
            )
        except (ValueError, OSError):
            pass

        try:
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (nofile_limit, nofile_limit),
            )
        except (ValueError, OSError):
            pass

    return _apply

class SubprocessSandboxError(RuntimeError):
    """Raised when the child process violated a sandbox invariant
    (timeout, non-zero exit, malformed JSON output, etc.)."""

def safe_execute_subprocess(
    python_code: str,
    query: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    mem_limit_bytes: int = DEFAULT_MEM_LIMIT_BYTES,
    cpu_limit_seconds: int = DEFAULT_CPU_LIMIT_SECONDS,
    fsize_limit_bytes: int = DEFAULT_FSIZE_LIMIT_BYTES,
    nofile_limit: int = DEFAULT_NOFILE_LIMIT,
    mode: str = "execute_if_trigger",
) -> str:
    """Validate, then run ``trigger``/``execute`` in a clean subprocess.

    AST validation happens first in the parent process — we never spawn
    code that has not passed :func:`nare.sandbox.validate_code`.

    Parameters
    ----------
    python_code:
        Skill source. Must define ``trigger(query)`` and
        ``execute(query)``.
    query:
        Single user query passed to the skill.
    timeout:
        Wall-clock timeout in seconds for the child process. The child
        is killed with SIGKILL on overrun. Defaults to 5s.
    mem_limit_bytes, cpu_limit_seconds, fsize_limit_bytes, nofile_limit:
        POSIX RLIMIT_* values applied via ``preexec_fn``. On non-POSIX
        platforms only the wall-clock timeout is enforced.
    mode:
        ``"execute_if_trigger"`` (default): mirror :func:`safe_execute`
        — return ``"Error: trigger returned False."`` if trigger is
        false, otherwise return ``str(execute(query))``.
        ``"trigger"``: return only the boolean result of ``trigger()``
        as ``"True"`` / ``"False"`` (string, for symmetry with the
        in-process API).

    Returns
    -------
    str
        Skill output (already stringified) on success.

    Raises
    ------
    nare.sandbox.SecurityError
        If the AST validator rejects the skill.
    SubprocessSandboxError
        If the child times out, exits non-zero, returns malformed
        JSON, or reports a runtime error inside ``trigger``/``execute``.
    """

    validate_code(python_code)

    if mode not in {"trigger", "execute", "execute_if_trigger"}:
        raise ValueError(f"unsupported mode: {mode!r}")

    request = json.dumps({"code": python_code, "query": query, "mode": mode})

    env = {"PYTHONIOENCODING": "utf-8"}

    cwd = tempfile.mkdtemp(prefix="nare_sbx_")
    try:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _CHILD_RUNNER],
                input=request,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                timeout=timeout,
                preexec_fn=_make_preexec(
                    mem_limit_bytes,
                    cpu_limit_seconds,
                    fsize_limit_bytes,
                    nofile_limit,
                ),
                text=True,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise SubprocessSandboxError(
                f"skill exceeded {timeout}s wall-clock budget"
            ) from e

        if completed.returncode != 0:
            raise SubprocessSandboxError(
                f"child exited with code {completed.returncode}: "
                f"{completed.stderr.strip()[:200]}"
            )

        out = completed.stdout.strip()
        if not out:
            raise SubprocessSandboxError(
                "child produced no output (was it killed by RLIMIT?)"
            )

        try:
            payload = json.loads(out.splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as e:
            raise SubprocessSandboxError(
                f"child produced malformed JSON: {out[:200]!r}"
            ) from e

        if not payload.get("ok"):
            raise SubprocessSandboxError(
                f"skill error: {payload.get('error', '<unknown>')}"
            )

        result = payload.get("result")
        if payload.get("kind") == "bool":
            return "True" if bool(result) else "False"
        return str(result)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)

def safe_call_trigger_subprocess(
    python_code: str,
    query: str,
    **kwargs: Any,
) -> bool:
    """Subprocess analogue of :func:`nare.sandbox.safe_call_trigger`.

    Returns the boolean ``trigger(query)`` result. Raises
    :class:`SecurityError` on AST violation or
    :class:`SubprocessSandboxError` on runtime/timeout failure.
    """
    out = safe_execute_subprocess(python_code, query, mode="trigger", **kwargs)
    return out == "True"

__all__ = [
    "SubprocessSandboxError",
    "safe_execute_subprocess",
    "safe_call_trigger_subprocess",
    "SecurityError",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MEM_LIMIT_BYTES",
    "DEFAULT_CPU_LIMIT_SECONDS",
]
