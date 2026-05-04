"""Tiny event bus for the NARE agent loop.

The agent loop is decoupled from rendering: each step emits an event,
and the CLI display module subscribes to those events and prints the
matching tool block in real time.

Event types are plain dataclasses; the bus is a list of typed callbacks.
No threading, no async — events are dispatched synchronously on the
calling thread, which is what the REPL expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

@dataclass
class Event:
    """Base class for all events. `kind` is set by subclasses."""

    kind: str = ""

@dataclass
class TaskStarted(Event):
    """The agent received a new user request."""

    kind: str = "task_started"
    query: str = ""
    intent: Optional[str] = None

@dataclass
class TaskFinished(Event):
    """The agent finished a request (success or otherwise)."""

    kind: str = "task_finished"
    ok: bool = True
    final_answer: Optional[str] = None
    iterations: int = 0
    tokens: int = 0
    elapsed: float = 0.0

@dataclass
class Thought(Event):
    """The agent emitted a free-form thought / status update."""

    kind: str = "thought"
    text: str = ""

@dataclass
class PlanProposed(Event):
    """The PlanningAgent produced a plan (only in EDIT intent)."""

    kind: str = "plan_proposed"
    steps: List[str] = field(default_factory=list)
    target_files: List[str] = field(default_factory=list)
    complexity: str = "moderate"

@dataclass
class ToolStart(Event):
    """A tool is about to run."""

    kind: str = "tool_start"
    name: str = ""
    display_verb: str = ""
    args: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolEnd(Event):
    """A tool finished running."""

    kind: str = "tool_end"
    name: str = ""
    display_verb: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    summary: Optional[str] = None
    body: Optional[str] = None
    body_lang: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class TokensConsumed(Event):
    """Token usage update from the LLM call."""

    kind: str = "tokens_consumed"
    delta_in: int = 0
    delta_out: int = 0

@dataclass
class TokenStreamed(Event):
    """A token was streamed from the LLM."""

    kind: str = "token_streamed"
    text: str = ""

@dataclass
class IterationCompleted(Event):
    """One inner iteration of the act-observe-reflect loop finished."""

    kind: str = "iteration_completed"
    n: int = 0
    budget_iterations: int = 0
    budget_tokens: int = 0
    used_tokens: int = 0

@dataclass
class TodoUpdated(Event):
    """The agent (or a tool) updated its persistent task list.

    Items are ``(state, text)`` tuples where ``state`` is one of
    ``'pending'`` / ``'in_progress'`` / ``'done'``. The renderer prints
    a single ``● Update todos`` panel per emit.
    """

    kind: str = "todo_updated"
    items: List[tuple] = field(default_factory=list)
    title: str = "Update todos"

Listener = Callable[[Event], None]

class EventBus:
    """Synchronous, in-process event bus."""

    def __init__(self) -> None:
        self._listeners: List[Listener] = []

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Register a listener. Returns an unsubscribe callable."""
        self._listeners.append(listener)

        def unsub() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return unsub

    def emit(self, event: Event) -> None:
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:

                continue
