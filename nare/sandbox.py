import ast
import re
import math
from typing import Dict, Any

class SecurityError(Exception):
    """Raised when generated code violates sandbox safety rules."""
    pass

class ASTValidator(ast.NodeVisitor):
    """
    Validates Python AST to ensure it contains no malicious or unsafe operations.
    Allowed: basic arithmetic, loops, conditionals, regex, math.
    Blocked: imports (except re, math), file I/O, eval, exec, globals manipulation.
    """
    
    ALLOWED_IMPORTS = {'re', 'math'}
    ALLOWED_BUILTINS = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'callable', 'chr', 'complex', 'dict', 'divmod', 'enumerate', 'filter',
        'float', 'format', 'frozenset', 'getattr', 'hasattr', 'hash', 'hex',
        'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'map', 'max',
        'min', 'next', 'object', 'oct', 'ord', 'pow', 'print', 'property', 'range',
        'repr', 'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum',
        'tuple', 'type', 'zip', 'Exception', 'ValueError', 'TypeError', 'IndexError',
        '__import__'
    }

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name not in self.ALLOWED_IMPORTS:
                raise SecurityError(f"Importing '{alias.name}' is strictly forbidden.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module not in self.ALLOWED_IMPORTS:
            raise SecurityError(f"Importing from '{node.module}' is strictly forbidden.")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in ['eval', 'exec', 'open', 'globals', 'locals', '__import__', 'compile']:
                raise SecurityError(f"Calling built-in '{func_name}()' is strictly forbidden.")
        self.generic_visit(node)

def safe_execute(python_code: str, query: str) -> str:
    """
    Safely executes an LLM-generated script by first validating its AST.
    The script MUST contain `trigger(query)` and `execute(query)`.
    Returns the execution result, or raises an exception.
    """
    if not python_code or not isinstance(python_code, str):
        raise ValueError("Invalid python code provided to sandbox.")

    # 1. Parse and validate AST
    try:
        tree = ast.parse(python_code)
    except SyntaxError as e:
        raise SecurityError(f"SyntaxError in generated code: {e}")

    validator = ASTValidator()
    validator.visit(tree)

    # 2. Setup restricted environment
    builtins_dict = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    safe_globals = {
        "__builtins__": {k: builtins_dict[k] for k in ASTValidator.ALLOWED_BUILTINS if k in builtins_dict},
        "re": re,
        "math": math
    }
    # 3. Execute definition safely
    exec(python_code, safe_globals)

    if 'trigger' not in safe_globals or 'execute' not in safe_globals:
        raise ValueError("Generated code missing required 'trigger' or 'execute' functions.")

    trigger_fn = safe_globals['trigger']
    execute_fn = safe_globals['execute']

    # 4. Run execution
    try:
        should_trigger = trigger_fn(query)
        if not should_trigger:
            return "Error: Trigger evaluated to False."
            
        result = execute_fn(query)
        return str(result)
    except Exception as e:
        raise RuntimeError(f"Execution crashed: {e}")
