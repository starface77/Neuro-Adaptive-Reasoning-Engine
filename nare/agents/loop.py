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

from nare.agents.events import (
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
from nare.agents.tools import DEFAULT_REGISTRY, ToolRegistry, build_default_registry
from nare.agents.tools.base import ToolResult

log = logging.getLogger("nare.agents.loop")


# ─────────────────────────────────────────────────────────────────────
# Budget
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Budget:
    """Hard caps on a single agent run."""

    max_iterations: int = 20
    max_tokens: int = 50_000
    max_wall_clock: float = 600.0  # seconds

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


# ─────────────────────────────────────────────────────────────────────
# Run result
# ─────────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    ok: bool
    final_answer: Optional[str] = None
    iterations: int = 0
    tokens: int = 0
    elapsed: float = 0.0
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""


# ─────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT_HEAD = """\
You are a senior software engineer working inside a CLI agent loop.

OUTPUT CONTRACT — read carefully, this is non-negotiable:

Each turn you MUST respond with EXACTLY ONE of these two blocks, and
NOTHING else (no prose before, no commentary after, no Markdown around
the block):

  (1) A single tool call:
      <tool_call>
      {"name": "<tool_name>", "args": {"<param>": "<value>", ...}}
      </tool_call>

  (2) A final answer:
      <final_answer>
      Concise reply to the user. Markdown allowed inside the block.
      </final_answer>

Examples of CORRECT output:

      <tool_call>
      {"name": "list_dir", "args": {"path": "."}}
      </tool_call>

      <tool_call>
      {"name": "write_file", "args": {"path": "snake/main.py", "content": "import pygame\\n..."}}
      </tool_call>

      <final_answer>
      Created `snake/` with main.py, snake.py, food.py and a TODO.
      Run with `python -m snake`.
      </final_answer>

Examples of INCORRECT output (DO NOT DO THIS):

      <write_file><path>x.py</path><content>...</content></write_file>     ← WRONG, use <tool_call>
      I will now create the file. <tool_call>...</tool_call>               ← WRONG, no prose outside
      <reasoning>...</reasoning><solution>...</solution>                   ← WRONG, that's a different protocol

Rules:
- After every tool call you'll receive an OBSERVATION block; use it
  to decide the next step.
- Do NOT emit both a tool call and a final answer in the same turn.
- Do NOT invent tools — only use the names listed in TOOLS below.
- For multi-step tasks, start by calling `update_todos` with a
  checklist; mark each item done as you complete it.
- Use real, exact paths and patterns. The working directory is set
  for you; relative paths are fine.
- For destructive tools (write_file, edit_file, bash) keep operations
  small and verifiable.
- When you've gathered enough information, emit <final_answer>.
- Be concise in <final_answer>; the user reads it on a terminal.

"""


def render_system_prompt(registry: ToolRegistry, working_dir: str) -> str:
    return (
        SYSTEM_PROMPT_HEAD
        + f"\nWorking directory: {working_dir}\n\n"
        + registry.schema_block()
    )


# ─────────────────────────────────────────────────────────────────────
# Output parsing
# ─────────────────────────────────────────────────────────────────────


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)


@dataclass
class ParsedTurn:
    """Outcome of parsing one LLM turn."""

    kind: str  # 'tool_call' | 'final_answer' | 'malformed'
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[str] = None
    raw: str = ""
    error: Optional[str] = None


def parse_turn(text: str) -> ParsedTurn:
    """Parse the LLM's raw output for a tool call or final answer."""
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

    return ParsedTurn(
        kind="malformed",
        error=(
            "no <tool_call> or <final_answer> block found. "
            "Respond with exactly one of those two formats."
        ),
        raw=text,
    )


# ─────────────────────────────────────────────────────────────────────
# Loop
# ─────────────────────────────────────────────────────────────────────


@dataclass
class AgentLoop:
    """Tool-calling loop driven by an LLM and an EventBus."""

    registry: ToolRegistry = field(default_factory=lambda: DEFAULT_REGISTRY)
    bus: EventBus = field(default_factory=EventBus)
    working_dir: str = "."
    budget: Budget = field(default_factory=Budget)

    # Injected for testability — defaults wire in nare.reasoning.llm.
    llm_call: Optional[Callable[[str], str]] = None

    # Optional triage / planning agents; loaded lazily to avoid hard
    # imports during unit tests.
    triage: Optional[Any] = None
    planner: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.llm_call is None:
            self.llm_call = _default_llm_call
        if self.triage is None:
            try:
                from nare.agents.triage import TriageAgent
                self.triage = TriageAgent()
            except Exception as e:
                log.warning(f"[Loop] TriageAgent unavailable: {e}")
        if self.planner is None:
            try:
                from nare.agents.planning import PlanningAgent
                self.planner = PlanningAgent()
            except Exception as e:
                log.warning(f"[Loop] PlanningAgent unavailable: {e}")

    # ── Public API ───────────────────────────────────────────────

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

        # 1) Triage intent (best-effort).
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

        # 2) Plan (only for EDIT intent and when planner is available).
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

        # 3) Tool-calling loop.
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

            # Build a token-level callback that emits TokenStreamed events
            # only for text inside <final_answer> tags.
            stream_cb: Optional[Callable[[str], None]] = None

            def emit_token(text: str) -> None:
                self.bus.emit(TokenStreamed(text=text))

            if on_final_token is not None:
                # Wrap both callbacks: emit TokenStreamed AND call on_final_token
                final_streamer = _make_final_answer_streamer(on_final_token)
                def combined_callback(text: str) -> None:
                    final_streamer(text)
                stream_cb = combined_callback
            else:
                # Only emit TokenStreamed for final answer content
                stream_cb = _make_final_answer_streamer(emit_token)

            try:
                # Three calling conventions for `llm_call`:
                #   (system, user, stream_callback)  → preferred
                #   (system, user)                   → simple direct API
                #   (combined,)                      → legacy / test stubs
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

            # Crude token estimate; the underlying llm module doesn't
            # surface counts here. We multiply chars by 0.25 ≈ tokens.
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

            # tool_call branch
            assert parsed.tool_name is not None
            verb = self._verb_for(parsed.tool_name)
            self.bus.emit(ToolStart(
                name=parsed.tool_name,
                display_verb=verb,
                args=parsed.tool_args,
            ))

            result = self.registry.call(parsed.tool_name, parsed.tool_args)

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

            # If the tool emitted todos, surface them on the bus so the
            # CLI renderer can print the checklist panel.
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

        elapsed = self.budget.elapsed
        self.bus.emit(TaskFinished(
            ok=stop_reason in {"final_answer", "ok"},
            final_answer=final_answer,
            iterations=self.budget.iterations,
            tokens=self.budget.total_tokens,
            elapsed=elapsed,
        ))

        return RunResult(
            ok=stop_reason in {"final_answer", "ok"},
            final_answer=final_answer,
            iterations=self.budget.iterations,
            tokens=self.budget.total_tokens,
            elapsed=elapsed,
            transcript=transcript,
            stop_reason=stop_reason,
        )

    # ── Internals ────────────────────────────────────────────────

    def _verb_for(self, name: str) -> str:
        try:
            return self.registry.get(name).display_verb or name
        except Exception:
            return name


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


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
    # Keep a small tail of the buffer to handle tags split across deltas.
    TAIL = 24

    def _push(delta: str) -> None:
        if state["done"]:
            return
        state["buf"] += delta

        # Look for the opening tag.
        if not state["in_final"]:
            idx = state["buf"].find(state["open_tag"])
            if idx >= 0:
                # Drop everything before the tag, then start streaming.
                rest = state["buf"][idx + len(state["open_tag"]):]
                # Skip a single leading newline so the answer starts
                # cleanly on its own line in the terminal.
                if rest.startswith("\n"):
                    rest = rest[1:]
                state["buf"] = rest
                state["in_final"] = True
            else:
                # Trim buffer to the rolling tail; we don't need the
                # whole prefix.
                if len(state["buf"]) > TAIL:
                    state["buf"] = state["buf"][-TAIL:]
                return

        # Inside <final_answer>: stream what we can, watch for </final_answer>.
        idx = state["buf"].find(state["close_tag"])
        if idx >= 0:
            chunk = state["buf"][:idx]
            if chunk:
                on_text(chunk)
            state["buf"] = ""
            state["done"] = True
            return

        # Hold back the last few chars in case the closing tag is being split.
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
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if stream_callback is not None:
        payload["stream"] = True

    headers = {
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
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
                # Read line-by-line so the callback fires in real time.
                parts: List[str] = []
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
                    if ev.get("type") == "content_block_delta":
                        delta = ev.get("delta", {}) or {}
                        t = delta.get("text", "")
                        if t:
                            parts.append(t)
                            try:
                                stream_callback(t)
                            except Exception:
                                # Never let a renderer bug kill the LLM call.
                                pass
                return "".join(parts)
            else:
                raw = r.read().decode("utf-8", errors="ignore")
                if not raw:
                    raise RuntimeError(f"empty response from {base}{path}")
                # The proxy returns SSE for both streaming and
                # non-streaming requests (it just buffers the SSE
                # frames into one body). Detect either shape.
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
                # OpenAI-style fallback: choices[0].message.content
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


# Convenience: build a loop with a working-dir-bound registry.
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
