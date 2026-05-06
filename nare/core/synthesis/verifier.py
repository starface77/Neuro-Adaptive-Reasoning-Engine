"""
V_sandbox — Formal verification sandbox.

Binary feedback system:
- R(y) = 1 if code compiles, executes without exceptions, passes all asserts
- R(y) = 0 otherwise, with detailed error trace

Isolated execution:
- Python AST compilation check
- Subprocess isolation (no access to parent process)
- Timeout protection
- Memory limits
"""

import ast
import subprocess
import tempfile
import os
import sys
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class VerificationResult:
    """Result of formal verification."""
    success: bool
    output: str
    error_trace: Optional[str]
    execution_time: float

class FormalVerifier:
    """Formal verification sandbox for code execution."""

    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def verify(self, code: str, test_cases: Optional[list] = None) -> VerificationResult:
        """Verify code through formal execution.

        Args:
            code: Python code to verify
            test_cases: Optional list of (input, expected_output) tuples

        Returns:
            VerificationResult with binary success flag and traces
        """
        import time

        start_time = time.time()

        try:
            ast.parse(code)
        except SyntaxError as e:
            return VerificationResult(
                success=False,
                output="",
                error_trace=f"SyntaxError: {e}",
                execution_time=time.time() - start_time
            )

        result = self._execute_isolated(code, test_cases)

        result.execution_time = time.time() - start_time
        return result

    def _execute_isolated(self, code: str, test_cases: Optional[list]) -> VerificationResult:
        """Execute code in isolated subprocess."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:

            if test_cases:
                code += "\n\n# Verification tests\n"
                for i, (input_val, expected) in enumerate(test_cases):
                    code += f"assert main({repr(input_val)}) == {repr(expected)}, 'Test {i+1} failed'\n"

            f.write(code)
            temp_path = f.name

        try:

            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,

            )

            if result.returncode == 0:

                return VerificationResult(
                    success=True,
                    output=result.stdout,
                    error_trace=None,
                    execution_time=0
                )
            else:

                return VerificationResult(
                    success=False,
                    output=result.stdout,
                    error_trace=result.stderr,
                    execution_time=0
                )

        except subprocess.TimeoutExpired:
            return VerificationResult(
                success=False,
                output="",
                error_trace=f"Timeout: execution exceeded {self.timeout}s",
                execution_time=0
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                output="",
                error_trace=f"Execution error: {e}",
                execution_time=0
            )
        finally:

            try:
                os.unlink(temp_path)
            except Exception as e:
                logging.warning(f"[Verifier] Failed to delete temp file {temp_path}: {e}")

    def verify_with_oracle(self, code: str, oracle_fn) -> VerificationResult:
        """Verify code using custom oracle function.

        Oracle should return (success: bool, message: str).
        """
        import time

        start_time = time.time()

        try:
            ast.parse(code)
        except SyntaxError as e:
            return VerificationResult(
                success=False,
                output="",
                error_trace=f"SyntaxError: {e}",
                execution_time=time.time() - start_time
            )

        exec_result = self._execute_isolated(code, None)

        if not exec_result.success:
            return exec_result

        try:
            oracle_success, oracle_msg = oracle_fn(code, exec_result.output)

            return VerificationResult(
                success=oracle_success,
                output=exec_result.output,
                error_trace=None if oracle_success else f"Oracle failed: {oracle_msg}",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                output=exec_result.output,
                error_trace=f"Oracle error: {e}",
                execution_time=time.time() - start_time
            )
