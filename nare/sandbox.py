"""Backward compatibility alias for sandbox module.

The actual sandbox implementation lives in nare.execution.sandboxes.base.
This shim re-exports the public API so that existing imports like
``from nare.sandbox import safe_execute`` continue to work.
"""

from nare.execution.sandboxes.base import (
    SecurityError,
    ASTValidator,
    validate_code,
    safe_load_module,
    safe_execute,
    safe_execute_freeform,
    safe_call_trigger,
    safe_call_execute_in_namespace,
    extract_python_block,
)

__all__ = [
    "SecurityError",
    "ASTValidator",
    "validate_code",
    "safe_load_module",
    "safe_execute",
    "safe_execute_freeform",
    "safe_call_trigger",
    "safe_call_execute_in_namespace",
    "extract_python_block",
]
