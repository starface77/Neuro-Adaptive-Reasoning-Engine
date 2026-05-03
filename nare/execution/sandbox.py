"""AST-validated sandbox for LLM-generated skill code.

SECURITY DISCLAIMER
-------------------
This sandbox uses AST whitelisting + restricted ``exec``. In-process
Python sandboxes are *fundamentally* leaky — any unforeseen path through
``__class__``, ``__subclasses__``, ``__globals__``, ``__builtins__``,
``__getattribute__``, etc. can escape. We block all known escape routes
below, but **for production use you should isolate execution in a
separate process** (subprocess + seccomp / firejail / gVisor / pyodide).

This module is intentionally the *single* entry-point for executing any
generated code. ``agent.py`` MUST route every ``trigger`` and
``execute`` call through ``safe_execute`` / ``safe_load_module`` so that
``ASTValidator`` always runs first.
"""

import ast
import math
import re
from typing import Any, Callable, Dict, Tuple


class SecurityError(Exception):
    """Raised when generated code violates sandbox safety rules."""


# ---------------------------------------------------------------------------
# AST validator
# ---------------------------------------------------------------------------

# Names that, if accessed as attributes, allow trivial sandbox escape.
_FORBIDDEN_ATTRS = frozenset({
    "__class__", "__bases__", "__base__", "__mro__", "__subclasses__",
    "__globals__", "__builtins__", "__import__", "__loader__",
    "__getattribute__", "__getattr__", "__setattr__", "__delattr__",
    "__dict__", "__code__", "__closure__", "__func__", "__module__",
    "__init_subclass__", "__class_getitem__", "__reduce__",
    "__reduce_ex__", "f_globals", "f_locals", "f_back",
    "gi_frame", "cr_frame", "ag_frame",
})

_FORBIDDEN_BARE_CALLS = frozenset({
    "eval", "exec", "open", "compile", "__import__",
    "globals", "locals", "vars", "dir", "delattr",
    "setattr", "getattr",  # blocked — used in many escapes
    "input", "breakpoint", "memoryview",
    "exit", "quit", "help", "copyright", "credits", "license",
})


class ASTValidator(ast.NodeVisitor):
    """Validates Python AST against a strict whitelist."""

    ALLOWED_IMPORTS = frozenset({"re", "math", "copy", "collections", "numpy"})

    # Builtins the skill is allowed to reference.
    #
    # ``__import__`` is included because Python's ``import`` statement
    # looks it up in builtins at runtime. Removing it would break even
    # ``import re``. Defense in depth instead:
    #   * ``visit_Import`` / ``visit_ImportFrom`` restrict imports to
    #     ``{re, math}`` regardless of ``__import__`` being callable;
    #   * ``visit_Call`` blocks bare-name ``__import__(...)``;
    #   * ``visit_Attribute`` blocks attribute access to ``__import__``.
    # Together this means ``__import__`` is only reachable via the
    # constrained ``import`` statement.
    ALLOWED_BUILTINS = frozenset({
        "__import__",
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
        "callable", "chr", "complex", "dict", "divmod", "enumerate",
        "filter", "float", "format", "frozenset", "hash", "hex",
        "int", "isinstance", "issubclass", "iter", "len", "list",
        "map", "max", "min", "next", "object", "oct", "ord", "pow",
        "print", "property", "range", "repr", "reversed", "round",
        "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
        "Exception", "ValueError", "TypeError", "IndexError", "KeyError",
        "ZeroDivisionError", "ArithmeticError", "AttributeError",
        "RuntimeError", "StopIteration",
    })

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in self.ALLOWED_IMPORTS:
                raise SecurityError(
                    f"import '{alias.name}' is forbidden (allowed: {sorted(self.ALLOWED_IMPORTS)})"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module not in self.ALLOWED_IMPORTS:
            raise SecurityError(
                f"from-import '{node.module}' is forbidden"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block any access to dunder/internal attributes.
        if node.attr in _FORBIDDEN_ATTRS or (
            node.attr.startswith("__") and node.attr.endswith("__")
            and node.attr not in {"__name__", "__doc__"}
        ):
            raise SecurityError(
                f"access to attribute '{node.attr}' is forbidden"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Block bare-name calls to dangerous builtins.
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_BARE_CALLS:
            raise SecurityError(
                f"calling '{node.func.id}()' is forbidden"
            )
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        raise SecurityError("'global' statements are forbidden")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise SecurityError("'nonlocal' statements are forbidden")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_safe_globals() -> Dict[str, Any]:
    """Construct the restricted globals dict used for skill execution."""
    builtins_dict = (
        __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    )
    safe_builtins = {
        name: builtins_dict[name]
        for name in ASTValidator.ALLOWED_BUILTINS
        if name in builtins_dict
    }

    try:
        import numpy as np
        numpy_module = np
    except ImportError:
        numpy_module = None

    safe_globals = {
        "__builtins__": safe_builtins,
        "re": re,
        "math": math,
    }

    if numpy_module is not None:
        safe_globals["numpy"] = numpy_module
        safe_globals["np"] = numpy_module

    return safe_globals


def validate_code(python_code: str) -> ast.AST:
    """Parse + AST-validate. Returns parsed tree on success.

    Raises SecurityError on any policy violation; propagates SyntaxError
    as SecurityError so callers have a single exception class to handle.
    """
    if not isinstance(python_code, str) or not python_code.strip():
        raise SecurityError("empty or non-string code")

    try:
        tree = ast.parse(python_code)
    except SyntaxError as exc:
        raise SecurityError(f"SyntaxError: {exc}") from exc

    ASTValidator().visit(tree)
    return tree


def safe_load_module(python_code: str) -> Dict[str, Any]:
    """Validate then exec the skill code, returning its namespace.

    This is the ONLY function modules outside this file should use to
    materialize ``trigger`` / ``execute`` from generated code. It runs
    the AST validator, then executes inside fresh restricted globals.

    Returns the populated namespace (containing the defined functions).
    """
    validate_code(python_code)
    safe_globals = _build_safe_globals()
    exec(python_code, safe_globals)  # noqa: S102 - intentional, after validation
    return safe_globals


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)\n```", re.DOTALL)


def extract_python_block(text: str) -> str:
    """Pull the first ```python ... ``` block out of an LLM response.

    Returns "" if no fenced block is present.  Used by HYBRID to detect
    when the LLM has emitted code instead of a direct answer.
    """
    if not text:
        return ""
    m = _CODE_BLOCK_RE.search(text)
    return m.group(1) if m else ""


def safe_execute_freeform(python_code: str, max_output_chars: int = 4096) -> str:
    """Run free-form Python code in the sandbox, capturing stdout.

    Unlike :func:`safe_execute`, this does *not* require ``trigger`` /
    ``execute`` functions. It is meant for HYBRID-path code blocks emitted
    by the LLM — short scripts that compute a value and ``print()`` it.

    Returns:
        * If the script printed anything, the captured stdout (trimmed).
        * Else, the value of a ``result`` variable if the script defined one.
        * Else, "" (caller should fall back to the original LLM text).

    Returns "Error: ..." on controlled failures (validation or runtime).
    Raises SecurityError on policy violations (caller must handle).
    """
    import io
    import contextlib

    validate_code(python_code)
    safe_globals = _build_safe_globals()

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(python_code, safe_globals)  # noqa: S102 - intentional, after validation
    except Exception as exc:  # noqa: BLE001 - intentional broad catch in sandbox
        return f"Error: {type(exc).__name__}: {exc}"

    out = buf.getvalue().strip()
    if out:
        # Take the last non-empty line as the answer, trimmed.
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if lines:
            answer = lines[-1].strip()
            return answer[:max_output_chars]
        return out[:max_output_chars]

    # Fallback: explicit `result` variable.
    if "result" in safe_globals:
        try:
            return str(safe_globals["result"])[:max_output_chars]
        except Exception:  # noqa: BLE001
            pass

    return ""


def safe_execute(python_code: str, query: str) -> str:
    """Validate, load, and run a skill against a single query string.

    Skill must define ``trigger(query) -> bool`` and
    ``execute(query) -> str``. Returns the execute() result on success
    (stringified), or "Error: ..." on a controlled failure.

    Raises SecurityError if validation fails.
    Raises ValueError if the skill is missing required functions.
    """
    namespace = safe_load_module(python_code)

    trigger_fn: Callable[[str], bool] = namespace.get("trigger")  # type: ignore[assignment]
    execute_fn: Callable[[str], str] = namespace.get("execute")   # type: ignore[assignment]

    if trigger_fn is None or execute_fn is None:
        raise ValueError(
            "skill must define both trigger(query) and execute(query)"
        )

    try:
        if not trigger_fn(query):
            return "Error: trigger returned False."
        result = execute_fn(query)
    except Exception as exc:  # noqa: BLE001 - intentional broad catch in sandbox
        # Controlled error path: caller treats "Error:" prefix as failure.
        return f"Error: {type(exc).__name__}: {exc}"

    return str(result)


def safe_call_trigger(python_code: str, query: str) -> Tuple[bool, Dict[str, Any]]:
    """Validate, load, and call only ``trigger(query)``.

    Returns (triggered, namespace) so callers can re-use the same
    validated namespace to call ``execute`` afterwards without re-loading.
    """
    namespace = safe_load_module(python_code)
    trigger_fn = namespace.get("trigger")
    if trigger_fn is None:
        raise ValueError("skill must define trigger(query)")

    try:
        triggered = bool(trigger_fn(query))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"trigger() crashed: {type(exc).__name__}: {exc}") from exc

    return triggered, namespace


def safe_call_execute_in_namespace(
    namespace: Dict[str, Any], query: str
) -> str:
    """Call ``execute(query)`` inside an already-validated namespace.

    Use together with ``safe_call_trigger`` to avoid double-loading.
    """
    execute_fn = namespace.get("execute")
    if execute_fn is None:
        raise ValueError("skill must define execute(query)")
    try:
        return str(execute_fn(query))
    except Exception as exc:  # noqa: BLE001
        return f"Error: {type(exc).__name__}: {exc}"
