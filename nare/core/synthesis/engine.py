"""Verified Synthesis — the actual lever NARE has over vanilla CoT.

Vanilla LLM does one shot: prompt → answer.  It cannot run its own
code, so it commits to whatever string-reverse / arithmetic /
roman-numeral conversion happens to come out of its first attempt.

NARE has a sandbox.  This module turns that capability into a closed
synthesis loop:

    propose code → execute → observe trace → if oracle fails, give the
    LLM the trace and ask it to fix → repeat up to N times.

Two correctness guarantees we rely on:

* Vanilla CoT cannot become *worse* by gaining execution feedback;
  worst case the loop accepts attempt 1 and behaves identically to
  vanilla.
* Whenever the oracle is available (LEARN phase, REM replay, sleep
  hold-out, A/B harness with ``oracle_spec``), the loop converges to
  the first oracle-passing attempt.  The Δ vs vanilla is therefore
  bounded below by 0 and is positive on every task where vanilla's
  first attempt was wrong but a fix is reachable in N steps.

This is intentionally minimal — it does NOT try to be cute about
prompt format, multi-temperature search, or speculative parallelism.
Those are further levers; the first one is just *running the code*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import re as _re
from typing import Any, Callable, List, Optional, Tuple

from ...execution.sandboxes.base import extract_python_block, safe_execute_freeform
from ..solve_context import SolveContext

FileProviderFn = Callable[[str], Optional[str]]

OracleFn = Callable[[str, str], Tuple[bool, dict]]

def _extract_needed_file(raw: str) -> Optional[str]:
    """Parse 'CANNOT_FIX: need file <path>' from LLM response."""
    m = _re.search(r'CANNOT_FIX:\s*need\s+file\s+([a-zA-Z0-9_./\-]+)', raw)
    if m:
        return m.group(1).strip()
    return None

@dataclass
class SynthesisAttempt:
    attempt: int
    raw_response: str
    extracted_code: str
    executed_output: str
    error: Optional[str]
    oracle_passed: Optional[bool]

    oracle_info: Any = None

@dataclass
class SynthesisResult:
    final_answer: str
    attempts: List[SynthesisAttempt]
    converged: bool
    oracle_used: bool

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)

def _normalise_oracle_info(info: Any) -> dict:
    """Coerce anything an oracle hands back into a plain mapping.

    The synthesis loop logs ``oracle_info`` verbatim to the LLM via the
    feedback prompt, so it must be representable. ``dict(info)`` worked
    for the original oracle contract (``info`` is a dict) but the
    in-tree oracles return ``str`` and an external oracle could return
    ``None`` / ``int`` / a tuple. ``dict(str)`` raises ``ValueError``
    and ``dict(None)`` raises ``TypeError`` — both surfaced as opaque
    "ERROR:..." routes in the user's last A/B run. Normalise here so
    the loop never crashes on a weird oracle.
    """
    if isinstance(info, dict):
        return dict(info)
    if info is None:
        return {}
    if isinstance(info, (str, bytes)):
        return {"diagnostic": info if isinstance(info, str) else info.decode("utf-8", "replace")}

    return {"diagnostic": repr(info)}

def _try_execute(raw: str, use_subprocess: bool = True) -> Tuple[str, str, Optional[str]]:
    """Extract a fenced block (if any) and execute it.

    Added subprocess isolation by default for security.

    Args:
        raw: Raw LLM response (may contain fenced code block)
        use_subprocess: If True, execute in isolated subprocess (default: True)

    Returns:
        (code, executed_output, error) tuple.
        - code: Extracted Python code block
        - executed_output: stdout from execution
        - error: Error message if execution failed, None on success

    Plain non-code answers (no fenced block) are passed through as
    ``executed_output == raw, code == ""``.
    """
    code = extract_python_block(raw)
    if not code.strip():

        return "", raw, None

    if use_subprocess:
        try:
            from ...execution.local import safe_execute_subprocess
            out = safe_execute_subprocess(code)
        except Exception as exc:
            return code, "", f"{type(exc).__name__}: {exc}"
    else:

        try:
            out = safe_execute_freeform(code)
        except Exception as exc:
            return code, "", f"{type(exc).__name__}: {exc}"

    if isinstance(out, str) and out.startswith("Error: "):

        return code, "", out[len("Error: "):]
    return code, out, None

def _format_feedback(
    query: str,
    attempt: SynthesisAttempt,
    expected_hint: Optional[str],
    context: Optional[SolveContext] = None,
) -> str:
    """Turn an attempt's trace into a feedback prompt for the next attempt.

    ``expected_hint`` is included only when we have ground truth (LEARN
    phase, hold-out replay).  In TEST/VERIFY phase we tell the LLM the
    output was rejected without leaking the answer.
    """
    parts: List[str] = [
        "Your previous attempt did NOT pass the oracle.",
        f"Query: {query}",
        "",
        "Previous code:",
        "```python",
        attempt.extracted_code or "(no fenced python block was provided)",
        "```",
        "",
        f"Executed output: {attempt.executed_output[:512]}",
    ]
    if attempt.error:
        parts += ["", f"Sandbox error: {attempt.error}"]

    if (
        attempt.extracted_code
        and attempt.extracted_code.strip()
        and not (attempt.executed_output or "").strip()
        and not attempt.error
    ):
        parts += [
            "",
            " your code did not call `print()`, so the sandbox",
            "captured no output. End your code with `print(answer)` (or",
            "whatever variable holds the result) so the answer is visible.",
        ]
    if attempt.oracle_info:

        parts += ["", f"Oracle diagnostic: {attempt.oracle_info}"]

    if context is not None:
        context_feedback = context.get_feedback_for_next_attempt()
        if context_feedback:
            parts += ["", context_feedback]

    if expected_hint:
        parts += ["", f"Expected (training only): {expected_hint}"]

    is_swe_bench = "File:" in query and "CRITICAL FORMAT REQUIREMENT" in query

    if is_swe_bench:

        parts += [
            "",
            "Fix the issue and output in the EXACT format from the original query:",
            "File: <path>",
            "```python",
            "<complete file content>",
            "```",
            "",
            "NOTHING ELSE. No explanations, no test code, no analysis.",
            "START YOUR RESPONSE with 'File:' on the first line.",
        ]
    else:

        parts += [
            "",
            "Rewrite the code so it passes the oracle.  Output ONLY a",
            "single fenced ```python``` block that prints the final answer.",
            "Do NOT hardcode the expected value; compute it from the query.",
            "If your previous approach computed a different quantity than",
            "what the query asks for, re-read the query and try a fundamentally",
            "different decomposition rather than tweaking the same code.",
        ]
    return "\n".join(parts)

def verified_synthesis(
    query: str,
    propose_fn: Callable[[str, List[SynthesisAttempt]], str],
    oracle: Optional[OracleFn] = None,
    max_attempts: int = 5,
    expected_hint: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    context: Optional[SolveContext] = None,
    file_provider: Optional[FileProviderFn] = None,
) -> SynthesisResult:
    """Run the verified-synthesis loop.

    Parameters
    ----------
    query
        The user query.  Passed through to ``propose_fn`` and the
        oracle.
    propose_fn
        ``(prompt, prior_attempts) -> raw_response``.  The first call
        receives the bare query as ``prompt``; subsequent calls
        receive a feedback prompt built from the previous attempt's
        trace + oracle diagnostic.  The function is expected to
        return either plain text or a fenced ```python``` block.
    oracle
        Optional ``(query, answer) -> (passed, info)``.  If ``None``
        we cannot self-correct and the loop degrades to a single
        attempt — exactly equivalent to vanilla.
    max_attempts
        Hard cap on LLM calls.  Default 5.
    expected_hint
        Ground-truth answer string, included verbatim in feedback
        prompts.  Only set this when the caller actually has the
        answer (LEARN phase, hold-out validation, REM replay).
    logger
        Optional logger; falls back to module logger.
    context
        Optional SolveContext for component coordination. Tracks IoU
        progress and enables adaptive max_attempts extension.

    Returns
    -------
    ``SynthesisResult`` with the final answer (best attempt by oracle
    pass first, then by "did the code execute without error"), the
    full list of attempts, and a ``converged`` flag.
    """
    log = logger or logging.getLogger(__name__)
    attempts: List[SynthesisAttempt] = []

    prompt = query

    _fetched_files: set = set()

    for i in range(max_attempts):
        raw = propose_fn(prompt, attempts)

        needed = _extract_needed_file(raw)
        if needed and file_provider:
            if needed not in _fetched_files:
                _fetched_files.add(needed)
                file_content = file_provider(needed)
                if file_content is not None:
                    log.info(
                        "[VS] Agent requested file '%s' — injecting into context",
                        needed,
                    )

                    injection = (
                        f"\n\nHere is the file you requested ({needed}):\n"
                        f"File: {needed}\n```python\n{file_content[:15000]}\n```\n"
                        f"\nNow fix the bug using this file. Output in the required format."
                    )
                    prompt = prompt + injection
                else:
                    log.warning(
                        "[VS] Agent requested file '%s' but it was not found",
                        needed,
                    )
                    injection = (
                        f"\n\nYou requested file '{needed}', but it DOES NOT EXIST in the repository. "
                        f"Please check the path or search for another file. Output ONLY your fix or another CANNOT_FIX request."
                    )
                    prompt = prompt + injection
                continue
            else:
                log.warning("[VS] Agent requested file '%s' AGAIN", needed)
                injection = (
                    f"\n\nYou already requested file '{needed}'. "
                    f"It was either already provided or it does not exist. "
                    f"DO NOT request it again. You MUST provide a fix now. Output in the required format."
                )
                prompt = prompt + injection
                continue
        code, out, err = _try_execute(raw)

        attempt = SynthesisAttempt(
            attempt=i + 1,
            raw_response=raw,
            extracted_code=code,
            executed_output=out,
            error=err,
            oracle_passed=None,
            oracle_info={},
        )

        if oracle is not None:
            try:

                oracle_input = raw
                verdict = oracle(query, oracle_input)

                if verdict is None:

                    passed, info = None, None
                elif isinstance(verdict, tuple) and len(verdict) >= 2:
                    passed, info = verdict[0], verdict[1]
                elif isinstance(verdict, tuple) and len(verdict) == 1:
                    passed, info = verdict[0], None
                else:
                    passed, info = bool(verdict), None
            except Exception as oracle_exc:

                passed, info = False, {"oracle_error": repr(oracle_exc)}

            if passed is not None:
                attempt.oracle_passed = bool(passed)
                attempt.oracle_info = _normalise_oracle_info(info)
            else:

                attempt.oracle_passed = None
                attempt.oracle_info = _normalise_oracle_info(info) if info else {}

            if context is not None and passed is not None:
                iou = info.get('iou', 0.0) if isinstance(info, dict) else 0.0
                context.add_attempt(
                    solution=out,
                    iou=iou,
                    converged=bool(passed),
                    error=err
                )

        attempts.append(attempt)

        if attempt.oracle_passed is True:
            log.info(
                "[VS] '%s' converged on attempt %d (oracle passed)",
                query[:60],
                attempt.attempt,
            )

            is_swe_bench = "File:" in raw and "```" in raw
            final_answer = raw if is_swe_bench else (out if out.strip() else raw)
            return SynthesisResult(
                final_answer=final_answer,
                attempts=attempts,
                converged=True,
                oracle_used=True,
            )

        if attempt.oracle_passed is None:
            log.info(
                "[VS] '%s' oracle unavailable, returning attempt %d without validation",
                query[:60],
                attempt.attempt,
            )

            is_swe_bench = "File:" in raw and "```" in raw
            final_answer = raw if is_swe_bench else (out if out.strip() else raw)
            return SynthesisResult(
                final_answer=final_answer,
                attempts=attempts,
                converged=False,
                oracle_used=False,
            )

        if context is not None and i + 1 >= max_attempts:
            if context.should_extend_attempts(max_attempts):
                max_attempts += 2
                log.info(
                    "[VS] Extending max_attempts to %d (best IoU: %.2f)",
                    max_attempts,
                    context.best_iou,
                )

        if oracle is None:

            log.debug(
                "[VS] '%s' no oracle, returning attempt 1", query[:60]
            )
            return SynthesisResult(
                final_answer=out,
                attempts=attempts,
                converged=False,
                oracle_used=False,
            )

        prompt = _format_feedback(query, attempt, expected_hint, context)

    log.warning(
        "[VS] '%s' did not converge in %d attempts",
        query[:60],
        max_attempts,
    )

    def _executed(a: SynthesisAttempt) -> str:
        """Return the attempt's executed stdout if non-empty, else ''."""
        if a.executed_output and a.executed_output.strip():
            return a.executed_output
        return ""

    def _payload(a: SynthesisAttempt) -> str:
        """The externally-visible answer for an attempt.

        For SWE-bench (file-based tasks), always return raw response.
        For code execution tasks, prefer executed stdout over raw LLM text.
        """
        raw = a.raw_response or ""
        is_swe_bench = "File:" in raw and "```" in raw

        if is_swe_bench:
            return raw if raw.strip() else ""

        ex = _executed(a)
        if ex:
            return ex
        if raw.strip():
            return raw
        return ""

    first = attempts[0] if attempts else None
    first_executed = _executed(first) if first else ""
    final = _payload(first) if first else ""

    if not first_executed:
        producing = [
            a for a in attempts[1:]
            if _executed(a)
        ]
        if producing:

            best_attempt = producing[-1]
            final = _payload(best_attempt)

    return SynthesisResult(
        final_answer=final,
        attempts=attempts,
        converged=False,
        oracle_used=oracle is not None,
    )
