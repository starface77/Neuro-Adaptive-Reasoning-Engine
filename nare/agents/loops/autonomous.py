"""Tool-calling agent loop for NARE.

This is a thin orchestration layer that sits *next to* the existing
`ReasoningRouter`. It does:

    triage → (plan when EDIT) → tool-call loop → final answer

Each iteration the LLM is given the conversation, the available tool
schemas, and the previous tool observations; it responds with either a
`<tool_call>` (executed locally; observation fed back) or a
`<final_answer>` (loop terminates).

The loop emits events on an `EventBus` so the CLI can render Phase-2
tool blocks live, without the loop knowing anything about Rich.

Honest scope:
- This loop deliberately does **not** replace `ReasoningRouter`. The
  existing router still owns 5-tier routing + verified synthesis +
  library learning. Phase 3 introduces a *parallel* path that the
  caller opts in to via `loop.run(...)`.
- Budgets are enforced (iterations / tokens / wall-clock). When a
  budget is exhausted the loop stops with a partial answer.
- The loop is single-threaded and synchronous. Phase 4 will layer
  resume / persistent task list on top.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from ...core.events import (
    EventBus,
    IterationCompleted,
    PlanProposed,
    TaskFinished,
    TaskStarted,
    Thought,
    TokensConsumed,
    TokenStreamed,
    TodoUpdated,
    ToolEnd,
    ToolStart,
)
from ...tools.builtin import DEFAULT_REGISTRY, ToolRegistry, build_default_registry
from ...tools.builtin.base import ToolResult

log = logging.getLogger("nare.agents.loop")

@dataclass
class Budget:
    """Hard caps on a single agent run."""

    max_iterations: int = 50
    max_tokens: int = 200_000
    max_wall_clock: float = 3600.0

    iterations: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    started: float = 0.0

    def start(self) -> None:
        self.started = time.time()
        self.iterations = 0
        self.tokens_in = 0
        self.tokens_out = 0

    @property
    def elapsed(self) -> float:
        return time.time() - self.started if self.started else 0.0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def exhausted(self) -> Optional[str]:
        """Return a reason string if the budget is exhausted, else None."""
        if self.iterations >= self.max_iterations:
            return f"max_iterations ({self.max_iterations})"
        if self.total_tokens >= self.max_tokens:
            return f"max_tokens ({self.max_tokens})"
        if self.elapsed >= self.max_wall_clock:
            return f"max_wall_clock ({self.max_wall_clock:.0f}s)"
        return None

@dataclass
class RunResult:
    ok: bool
    final_answer: Optional[str] = None
    iterations: int = 0
    tokens: int = 0
    elapsed: float = 0.0
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""

SYSTEM_PROMPT_HEAD = """
You are NARE, a senior software engineer operating in a CLI agent loop.
You have real filesystem access. You execute real commands.

IDENTITY:
- Your name is NARE (Neural Amortized Reasoning Engine).
- You are NOT Claude, GPT, Kiro, or any other assistant.
- Do NOT greet the user unless they greet you first.

OUTPUT FORMAT (strict):
Each turn MUST end with exactly ONE of these:

<tool_call>
{"name": "tool_name", "args": {"param": "value"}}
</tool_call>

OR

<final_answer>
Your response here.
</final_answer>

You may think/reason before the block, but MUST end with one of the two.

HONESTY RULES (CRITICAL):
- NEVER pretend to execute a tool. You MUST use <tool_call> to execute anything.
- NEVER describe what you "would do" — DO IT via tool_call.
- NEVER fabricate file contents, directory listings, or command output.
- If you haven't read a file, you don't know its contents. Read it first.
- If you haven't run a command, you don't know its output. Run it first.
- NEVER say "I created file X" without a preceding write_file tool_call.
- NEVER say "I ran command X" without a preceding bash tool_call.

TOOL CALL RULES:
- NEVER call tools with empty args: {}
- Every tool requires specific parameters — check the schema below.
- read_file REQUIRES: path (string). NEVER call read_file({}).
- write_file REQUIRES: path (string), content (string).
- edit_file REQUIRES: path (string), old (string), new (string).
- bash REQUIRES: command (string).
- For update_todos: items must be [{text: "...", state: "todo|done|in_progress"}]
- Read large files in chunks (use offset/limit) — avoid reading 500+ lines at once.
- If you already read a file, the cache has it — don't re-read.
- After a tool_call, you get an OBSERVATION — use it for your next step.
- Do NOT emit both tool_call and final_answer in the same turn.
- If edit_file returns "+0 -0" (no changes), do NOT retry the same edit.
- If reading a file doesn't help, try a different approach.
- When you have enough information, emit <final_answer>.
- Be concise in <final_answer>; the user reads it on a terminal.

RESPONSE STYLE:
- Short, direct answers (1-3 sentences when possible).
- No emojis, no decorative formatting, no marketing tone.
- Russian or English — match the user's language.
"""

def render_system_prompt(registry: ToolRegistry, working_dir: str) -> str:
    """Render system prompt with cacheable tool schemas.

    Returns a string that will be converted to cache_control format
    by _default_llm_call.
    """
    return (
        SYSTEM_PROMPT_HEAD
        + f"\nWorking directory: {working_dir}\n\n"
        + registry.schema_block()
    )

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)

@dataclass
class ParsedTurn:
    """Outcome of parsing one LLM turn."""

    kind: str
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[str] = None
    raw: str = ""
    error: Optional[str] = None

def parse_turn(text: str) -> ParsedTurn:
    """Parse the LLM's raw output for a tool call or final answer.

    Note: Ignores thinking blocks - they're handled separately in streaming.
    """

    final = _FINAL_ANSWER_RE.search(text)
    if final:
        return ParsedTurn(
            kind="final_answer",
            answer=final.group(1).strip(),
            raw=text,
        )

    call = _TOOL_CALL_RE.search(text)
    if call:
        body = call.group(1).strip()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            return ParsedTurn(
                kind="malformed",
                error=f"tool_call JSON parse error: {e}",
                raw=text,
            )
        name = payload.get("name")
        args = payload.get("args", {}) or {}
        if not name or not isinstance(name, str):
            return ParsedTurn(
                kind="malformed",
                error="tool_call missing 'name' string",
                raw=text,
            )
        if not isinstance(args, dict):
            return ParsedTurn(
                kind="malformed",
                error="tool_call 'args' must be an object",
                raw=text,
            )
        return ParsedTurn(
            kind="tool_call",
            tool_name=name,
            tool_args=args,
            raw=text,
        )

    if len(text.strip()) < 10:
        return ParsedTurn(
            kind="malformed",
            error="Response too short - waiting for more content",
            raw=text,
        )

    return ParsedTurn(
        kind="malformed",
        error=(
            "no <tool_call> or <final_answer> block found. "
            "Respond with exactly one of those two formats."
        ),
        raw=text,
    )

@dataclass
class AgentLoop:
    """Tool-calling loop driven by an LLM and an EventBus."""

    registry: ToolRegistry = field(default_factory=lambda: DEFAULT_REGISTRY)
    bus: EventBus = field(default_factory=EventBus)
    working_dir: str = "."
    budget: Budget = field(default_factory=Budget)

    llm_call: Optional[Callable[[str], str]] = None

    triage: Optional[Any] = None
    planner: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.llm_call is None:
            self.llm_call = _default_llm_call
        if self.triage is None:
            try:
                from ..roles.triage import TriageAgent
                self.triage = TriageAgent()
            except Exception as e:
                log.warning(f"[Loop] TriageAgent unavailable: {e}")
        if self.planner is None:
            try:
                from ..planning import PlanningAgent
                self.planner = PlanningAgent()
            except Exception as e:
                log.warning(f"[Loop] PlanningAgent unavailable: {e}")

    def run(
        self,
        query: str,
        *,
        chat_history: Optional[str] = None,
        repo_map: Optional[str] = None,
        on_final_token: Optional[Callable[[str], None]] = None,
    ) -> RunResult:
        """Execute the agent loop for `query`. Returns a `RunResult`.

        ``on_final_token`` (optional) is invoked with each text fragment
        the LLM streams while it is inside a ``<final_answer>`` block.
        The CLI uses this to typewriter the answer to the terminal in
        real time. Tokens emitted *outside* a ``<final_answer>`` block
        (i.e. inside ``<tool_call>`` JSON or model preambles) are not
        forwarded — those are still parsed and acted on, but never
        printed raw.
        """

        self.budget.start()

        intent = "EDIT"
        if self.triage is not None:
            try:
                intent = self.triage.classify(query)
            except Exception as e:
                log.warning(f"[Loop] triage failed: {e}")

        self.bus.emit(TaskStarted(query=query, intent=intent))

        transcript: List[Dict[str, Any]] = [
            {"role": "user", "content": query},
        ]

        plan_steps: List[str] = []
        if intent == "EDIT" and self.planner is not None:
            try:
                plan = self.planner.generate_plan(query, repo_map=repo_map)
                plan_steps = plan.get("plan_steps") or []
                self.bus.emit(PlanProposed(
                    steps=plan_steps,
                    target_files=plan.get("target_files", []),
                    complexity=plan.get("complexity", "moderate"),
                ))
            except Exception as e:
                log.warning(f"[Loop] planning failed: {e}")

        system_prompt = render_system_prompt(self.registry, self.working_dir)
        observations: List[str] = []
        if chat_history:
            observations.append(f"# CHAT HISTORY\n{chat_history}\n")
        if plan_steps:
            steps_block = "\n".join(f"  {i}. {s}" for i, s in enumerate(plan_steps, 1))
            observations.append(f"# PLAN\n{steps_block}\n")
        observations.append(f"# USER REQUEST\n{query}\n")

        final_answer: Optional[str] = None
        stop_reason = "ok"

        failed_tools: List[tuple] = []
        no_op_tools: List[tuple] = []
        all_reads: List[str] = []
        max_same_failure = 2
        max_same_no_op = 2
        max_same_read = 3

        read_cache: Dict[str, str] = {}

        while True:
            reason = self.budget.exhausted()
            if reason:
                stop_reason = f"budget_exhausted: {reason}"
                final_answer = (
                    final_answer
                    or "I ran out of budget before finishing. "
                    + f"Reason: {reason}. Use /resume in Phase 4 to continue."
                )
                break

            self.budget.iterations += 1

            user_body = "\n".join(observations) + "\n\nNext turn:"

            stream_cb: Optional[Callable[[str], None]] = None

            def emit_token(text: str) -> None:
                self.bus.emit(TokenStreamed(text=text))

            def emit_thinking(text: str) -> None:

                self.bus.emit(Thought(text=text))

            if on_final_token is not None:

                final_streamer = _make_final_answer_streamer(on_final_token)
                def combined_callback(text: str) -> None:

                    if text.startswith("[thinking] "):
                        thinking_text = text[len("[thinking] "):]
                        emit_thinking(thinking_text)
                    else:
                        final_streamer(text)
                stream_cb = combined_callback
            else:

                final_streamer = _make_final_answer_streamer(emit_token)
                def combined_callback(text: str) -> None:
                    if text.startswith("[thinking] "):
                        thinking_text = text[len("[thinking] "):]
                        emit_thinking(thinking_text)
                    else:
                        final_streamer(text)
                stream_cb = combined_callback

            try:

                import inspect as _inspect
                try:
                    sig = _inspect.signature(self.llm_call)
                    params = [
                        p for p in sig.parameters.values()
                        if p.kind in (
                            _inspect.Parameter.POSITIONAL_ONLY,
                            _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                    ]
                    n_params = len(params)
                    accepts_stream = (
                        n_params >= 3
                        or any(
                            p.name == "stream_callback" for p in sig.parameters.values()
                        )
                    )
                except (TypeError, ValueError):
                    n_params = 1
                    accepts_stream = False

                if accepts_stream:
                    output = self.llm_call(system_prompt, user_body, stream_cb)
                elif n_params >= 2:
                    output = self.llm_call(system_prompt, user_body)
                else:
                    output = self.llm_call(_assemble_prompt(system_prompt, observations))
            except Exception as e:
                stop_reason = f"llm_error: {e}"
                final_answer = f"LLM call failed: {e}"
                break

            tok_in = max(1, (len(system_prompt) + len(user_body)) // 4)
            tok_out = max(1, len(output) // 4)
            self.budget.tokens_in += tok_in
            self.budget.tokens_out += tok_out
            self.bus.emit(TokensConsumed(delta_in=tok_in, delta_out=tok_out))

            parsed = parse_turn(output)
            transcript.append({"role": "assistant", "content": output})

            if parsed.kind == "final_answer":
                final_answer = parsed.answer or ""
                stop_reason = "final_answer"
                break

            if parsed.kind == "malformed":
                self.bus.emit(Thought(text=f"(parse retry: {parsed.error})"))
                observations.append(
                    "# OBSERVATION\n"
                    "Your last turn was malformed. "
                    f"{parsed.error}\n"
                    "Reply with EXACTLY ONE <tool_call>...</tool_call> or "
                    "<final_answer>...</final_answer> block.\n"
                )
                self.bus.emit(IterationCompleted(
                    n=self.budget.iterations,
                    budget_iterations=self.budget.max_iterations,
                    budget_tokens=self.budget.max_tokens,
                    used_tokens=self.budget.total_tokens,
                ))
                continue

            assert parsed.tool_name is not None

            if not parsed.tool_args:
                self.bus.emit(Thought(text=f"  Invalid: {parsed.tool_name} called with no arguments"))

                example = ""
                if parsed.tool_name == "read_file":
                    example = '\n\nCorrect example:\n<tool_call>\n{"name": "read_file", "args": {"path": "nare/core/__init__.py"}}\n</tool_call>'
                elif parsed.tool_name == "write_file":
                    example = '\n\nCorrect example:\n<tool_call>\n{"name": "write_file", "args": {"path": "file.py", "content": "# code here"}}\n</tool_call>'
                elif parsed.tool_name == "bash":
                    example = '\n\nCorrect example:\n<tool_call>\n{"name": "bash", "args": {"command": "ls -la"}}\n</tool_call>'

                observations.append(
                    f"# OBSERVATION\n"
                    f"ERROR: Tool '{parsed.tool_name}' was called with empty args: {{}}\n"
                    f"This is INVALID. Every tool requires arguments.\n"
                    f"Check the TOOLS schema in the system prompt for required parameters.{example}\n"
                )
                self.bus.emit(IterationCompleted(
                    n=self.budget.iterations,
                    budget_iterations=self.budget.max_iterations,
                    budget_tokens=self.budget.max_tokens,
                    used_tokens=self.budget.total_tokens,
                ))
                continue

            required_params = {
                'read_file': ['path'],
                'write_file': ['path', 'content'],
                'edit_file': ['path', 'old', 'new'],
                'bash': ['command'],
                'grep': ['pattern'],
                'list_dir': [],
            }

            if parsed.tool_name in required_params:
                missing = [p for p in required_params[parsed.tool_name] if not parsed.tool_args.get(p)]
                if missing:
                    self.bus.emit(Thought(text=f"  Invalid: {parsed.tool_name} missing {', '.join(missing)}"))

                    example = ""
                    if parsed.tool_name == "read_file":
                        example = '\n\nCorrect:\n<tool_call>\n{"name": "read_file", "args": {"path": "path/to/file.py"}}\n</tool_call>'
                    elif parsed.tool_name == "write_file":
                        example = '\n\nCorrect:\n<tool_call>\n{"name": "write_file", "args": {"path": "file.py", "content": "code here"}}\n</tool_call>'
                    elif parsed.tool_name == "edit_file":
                        example = '\n\nCorrect:\n<tool_call>\n{"name": "edit_file", "args": {"path": "file.py", "old": "old text", "new": "new text"}}\n</tool_call>'
                    elif parsed.tool_name == "bash":
                        example = '\n\nCorrect:\n<tool_call>\n{"name": "bash", "args": {"command": "ls -la"}}\n</tool_call>'
                    elif parsed.tool_name == "grep":
                        example = '\n\nCorrect:\n<tool_call>\n{"name": "grep", "args": {"pattern": "search_term"}}\n</tool_call>'

                    observations.append(
                        f"# OBSERVATION\n"
                        f"ERROR: Tool '{parsed.tool_name}' missing required parameters: {', '.join(missing)}\n"
                        f"You provided: {parsed.tool_args}\n"
                        f"Required parameters: {', '.join(required_params[parsed.tool_name])}{example}\n"
                    )
                    self.bus.emit(IterationCompleted(
                        n=self.budget.iterations,
                        budget_iterations=self.budget.max_iterations,
                        budget_tokens=self.budget.max_tokens,
                        used_tokens=self.budget.total_tokens,
                    ))
                    continue

            verb = self._verb_for(parsed.tool_name)
            self.bus.emit(ToolStart(
                name=parsed.tool_name,
                display_verb=verb,
                args=parsed.tool_args,
            ))

            if parsed.tool_name == 'read_file':
                file_path = parsed.tool_args.get('path', '')
                if file_path in read_cache:

                    from nare.tools.builtin.base import ToolResult
                    result = ToolResult(
                        ok=True,
                        summary=f"Read {file_path} (cached)",
                        body=read_cache[file_path],
                        meta={'lines': len(read_cache[file_path].splitlines()), 'cached': True}
                    )
                    self.bus.emit(Thought(text=f"  Using cached: {file_path}"))
                else:

                    result = self.registry.call(parsed.tool_name, parsed.tool_args)
                    if result.ok and result.body:
                        read_cache[file_path] = result.body
            else:
                result = self.registry.call(parsed.tool_name, parsed.tool_args)

            if not result.ok:
                import hashlib
                args_hash = hashlib.md5(str(parsed.tool_args).encode()).hexdigest()[:8]
                failure_sig = (parsed.tool_name, args_hash, result.error or "error")

                same_failures = sum(1 for f in failed_tools if f[:2] == failure_sig[:2])

                if same_failures >= max_same_failure:

                    self.bus.emit(Thought(
                        text=f"  Stopping: {parsed.tool_name} failed {same_failures + 1} times"
                    ))
                    final_answer = (
                        f"Unable to complete task. "
                        f"Tool '{parsed.tool_name}' failed repeatedly: {result.error}"
                    )
                    stop_reason = "repeated_failure"
                    break

                failed_tools.append(failure_sig)
            else:

                import hashlib
                args_hash = hashlib.md5(str(parsed.tool_args).encode()).hexdigest()[:8]

                is_no_op = False
                if parsed.tool_name == 'edit_file':

                    if result.summary and ('+0' in result.summary and '-0' in result.summary):
                        is_no_op = True
                        self.bus.emit(Thought(text=f"  Warning: edit_file made no changes"))
                elif parsed.tool_name == 'read_file':

                    file_path = parsed.tool_args.get('path', '')
                    all_reads.append(file_path)

                    recent_same_reads = [p for p in all_reads[-10:] if p == file_path]
                    if len(recent_same_reads) >= max_same_read:
                        self.bus.emit(Thought(
                            text=f"  Stopping: read {file_path} {len(recent_same_reads)} times"
                        ))
                        final_answer = (
                            f"Unable to make progress. "
                            f"Repeatedly reading '{file_path}' without taking action. "
                            f"The file content is already available - please analyze it and proceed."
                        )
                        stop_reason = "repeated_read"
                        break
                    elif len(recent_same_reads) >= 2:
                        is_no_op = True
                        self.bus.emit(Thought(text=f"  Warning: reading {file_path} again ({len(recent_same_reads)}/{max_same_read})"))
                elif parsed.tool_name == 'update_todos':

                    recent_updates = [op for op in no_op_tools[-3:] if op[0] == 'update_todos']
                    if len(recent_updates) >= 1:
                        is_no_op = True
                        self.bus.emit(Thought(text=f"  Warning: repeatedly updating todos"))

                if is_no_op:
                    no_op_sig = (parsed.tool_name, args_hash, parsed.tool_args.get('path', ''))
                    same_no_ops = sum(1 for op in no_op_tools if op[:2] == no_op_sig[:2])

                    if same_no_ops >= max_same_no_op:
                        self.bus.emit(Thought(
                            text=f"  Stopping: {parsed.tool_name} repeated {same_no_ops + 1} times with no changes"
                        ))
                        final_answer = (
                            f"Unable to make progress. "
                            f"Tool '{parsed.tool_name}' executed repeatedly without making changes. "
                            f"Please provide more specific instructions or check if the task is already complete."
                        )
                        stop_reason = "repeated_no_op"
                        break

                    no_op_tools.append(no_op_sig)
                    self.bus.emit(Thought(text=f"  Warning: {parsed.tool_name} made no changes ({same_no_ops + 1}/{max_same_no_op})"))
                else:

                    no_op_tools.clear()

            self.bus.emit(ToolEnd(
                name=parsed.tool_name,
                display_verb=verb,
                args=parsed.tool_args,
                ok=result.ok,
                summary=result.summary,
                body=result.body,
                body_lang=result.body_lang,
                meta=dict(result.meta),
                error=result.error,
            ))

            todo_items = result.meta.get("_todo_items") if result.meta else None
            if todo_items:
                self.bus.emit(TodoUpdated(items=list(todo_items)))

            obs = (
                f"# OBSERVATION ({parsed.tool_name})\n"
                f"{result.to_llm_observation()}\n"
            )
            observations.append(obs)
            transcript.append({"role": "tool", "content": obs})

            self.bus.emit(IterationCompleted(
                n=self.budget.iterations,
                budget_iterations=self.budget.max_iterations,
                budget_tokens=self.budget.max_tokens,
                used_tokens=self.budget.total_tokens,
            ))

        summary = self._generate_summary(query, transcript, final_answer, stop_reason)

        elapsed = self.budget.elapsed
        self.bus.emit(TaskFinished(
            ok=stop_reason in {"final_answer", "ok"},
            final_answer=final_answer,
            iterations=self.budget.iterations,
            tokens=self.budget.total_tokens,
            elapsed=elapsed,
        ))

        if summary:
            self.bus.emit(Thought(text=summary))

        return RunResult(
            ok=stop_reason in {"final_answer", "ok"},
            final_answer=final_answer,
            iterations=self.budget.iterations,
            tokens=self.budget.total_tokens,
            elapsed=elapsed,
            transcript=transcript,
            stop_reason=stop_reason,
        )

    def _generate_summary(
        self,
        query: str,
        transcript: List[Dict[str, Any]],
        final_answer: Optional[str],
        stop_reason: str,
    ) -> str:
        """Generate a concise summary using LLM."""

        if len(transcript) < 3:
            return ""

        actions_summary = []
        for turn in transcript:
            if turn.get("role") == "assistant":
                content = turn.get("content", "")
                if "<tool_call>" in content:
                    import re
                    match = re.search(r'"name":\s*"([^"]+)"', content)
                    if match:
                        actions_summary.append(match.group(1))

        from ...reasoning import llm

        summary_prompt = f"""
Task: {query}
Actions taken: {', '.join(actions_summary[:10])}
Result: {stop_reason}

Be concise and clear. No bullet points, just plain text."""

        try:
            samples, _ = llm.generate_samples(
                summary_prompt,
                n=1,
                temperature=0.3,
                mode="DIRECT"
            )
            if samples:
                return samples[0]['solution'].strip()
        except Exception:
            pass

        return ""

    def _verb_for(self, name: str) -> str:
        try:
            return self.registry.get(name).display_verb or name
        except Exception:
            return name

def _assemble_prompt(system_prompt: str, observations: List[str]) -> str:
    """Concatenate the system prompt and rolling observation log.

    Used as a single combined string only when the caller-supplied
    ``llm_call`` accepts one positional argument (the legacy contract
    used by tests and stub LLMs). The default LLM caller bypasses this
    function and talks to the provider directly with a proper
    system/user split.
    """
    body = "\n".join(observations)
    return f"{system_prompt}\n\n---\n\n{body}\n\nNext turn:"

def _make_final_answer_streamer(
    on_text: Callable[[str], None],
) -> Callable[[str], None]:
    """Filter LLM token deltas to only what's inside ``<final_answer>``.

    The model emits a mix of prose, ``<tool_call>{...}</tool_call>`` JSON
    and ``<final_answer>...</final_answer>`` Markdown. We only want to
    typewriter-stream the answer to the user; everything else stays
    buffered until the loop's parser handles it.

    The returned callable maintains a small rolling buffer so that
    opening/closing tags split across multiple deltas (``<final_`` then
    ``answer>``) are still detected.
    """
    state = {
        "buf": "",
        "in_final": False,
        "done": False,
        "open_tag": "<final_answer>",
        "close_tag": "</final_answer>",
    }

    TAIL = 24

    def _push(delta: str) -> None:
        if state["done"]:
            return
        state["buf"] += delta

        if not state["in_final"]:
            idx = state["buf"].find(state["open_tag"])
            if idx >= 0:

                rest = state["buf"][idx + len(state["open_tag"]):]

                if rest.startswith("\n"):
                    rest = rest[1:]
                state["buf"] = rest
                state["in_final"] = True
            else:

                if len(state["buf"]) > TAIL:
                    state["buf"] = state["buf"][-TAIL:]
                return

        idx = state["buf"].find(state["close_tag"])
        if idx >= 0:
            chunk = state["buf"][:idx]
            if chunk:
                on_text(chunk)
            state["buf"] = ""
            state["done"] = True
            return

        if len(state["buf"]) > TAIL:
            emit = state["buf"][:-TAIL]
            state["buf"] = state["buf"][-TAIL:]
            on_text(emit)

    return _push

def _default_llm_call(
    system_prompt: str,
    user_prompt: str,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Default LLM caller — talks to Anthropic (or a compatible proxy).

    The user's proxy returns SSE (``event: ...\\ndata: {...}``) even for
    non-streaming requests, so we always parse the body as SSE first
    and fall back to plain JSON. When ``stream_callback`` is provided
    we also enable ``stream=True`` server-side and forward each
    ``content_block_delta`` token to the callback in real time.

    The endpoint shape varies between providers:
      - Anthropic public API:    ``/v1/messages``
      - Most reverse proxies:    ``/messages``
    We try ``/messages`` first (matches the user's proxy and the
    legacy ``nare/reasoning/llm.py``) and fall back to ``/v1/messages``
    on 404.
    """
    import json
    import os
    import urllib.error
    import urllib.request

    base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    key = os.getenv("ANTHROPIC_API_KEY", "")
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. The agent loop needs an LLM key. "
            "Add it to your .env or use /agent off to fall back to the legacy router."
        )

    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0.2,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 2000
        },
        "system": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if stream_callback is not None:
        payload["stream"] = True

    headers = {
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "prompt-caching-2024-07-31,extended-thinking-2024-12-12",
        "accept": "text/event-stream" if stream_callback is not None else "application/json",
    }
    data = json.dumps(payload).encode("utf-8")

    def _extract_sse_text(body_str: str) -> str:
        """Walk an SSE body and concatenate all content_block_delta texts."""
        out: List[str] = []
        for line in body_str.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if not chunk or chunk == "[DONE]":
                continue
            try:
                ev = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "content_block_delta":
                delta = ev.get("delta", {}) or {}
                t = delta.get("text", "")
                if t:
                    out.append(t)
        return "".join(out)

    def _post(path: str) -> str:
        req = urllib.request.Request(f"{base}{path}", data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            if stream_callback is not None:

                parts: List[str] = []
                thinking_parts: List[str] = []
                current_block_type = None

                while True:
                    raw_line = r.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[len("data:"):].strip()
                    if not chunk or chunk == "[DONE]":
                        continue
                    try:
                        ev = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue

                    if ev.get("type") == "content_block_start":
                        block = ev.get("content_block", {})
                        current_block_type = block.get("type")

                    if ev.get("type") == "content_block_delta":
                        delta = ev.get("delta", {}) or {}
                        t = delta.get("text", "")
                        if t:

                            if current_block_type == "thinking":
                                thinking_parts.append(t)

                                try:
                                    stream_callback(f"[thinking] {t}")
                                except Exception:
                                    pass
                            else:
                                parts.append(t)
                                try:
                                    stream_callback(t)
                                except Exception:
                                    pass

                result = "".join(parts)

                if not result and thinking_parts:

                    return ""

                return result
            else:
                raw = r.read().decode("utf-8", errors="ignore")
                if not raw:
                    raise RuntimeError(f"empty response from {base}{path}")

                stripped = raw.lstrip()
                if stripped.startswith("event:") or stripped.startswith("data:"):
                    return _extract_sse_text(raw)
                obj = json.loads(raw)
                parts = obj.get("content") or []
                if isinstance(parts, list):
                    text = "".join(
                        p.get("text", "")
                        for p in parts
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                    if text:
                        return text

                choices = obj.get("choices") or []
                if choices:
                    msg = choices[0].get("message", {}) or {}
                    text = msg.get("content", "") or ""
                    if text:
                        return text
                raise RuntimeError(f"LLM returned no text content; raw body: {raw[:300]}")

    try:
        return _post("/messages")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return _post("/v1/messages")
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            err_body = ""
        raise RuntimeError(f"LLM HTTP {e.code}: {err_body}") from None

def build_loop(
    working_dir: str = ".",
    *,
    bus: Optional[EventBus] = None,
    budget: Optional[Budget] = None,
) -> AgentLoop:
    return AgentLoop(
        registry=build_default_registry(working_dir=working_dir),
        bus=bus or EventBus(),
        working_dir=working_dir,
        budget=budget or Budget(),
    )
