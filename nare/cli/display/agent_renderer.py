"""Bridge between the agent EventBus and the CLI display blocks.

`attach_renderer(bus, console)` subscribes to every relevant event and
prints the corresponding Phase-2 block in real time.

The renderer keeps no state across tasks — each `TaskStarted` resets
its internal counters.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from rich.console import Console
from rich.text import Text
from rich.markup import escape

from nare.agents.events import (
    Event,
    EventBus,
    IterationCompleted,
    PlanProposed,
    TaskFinished,
    TaskStarted,
    Thought,
    TodoUpdated,
    TokensConsumed,
    TokenStreamed,
    ToolEnd,
    ToolStart,
)
from . import blocks


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def attach_renderer(
    bus: EventBus,
    console: Optional[Console] = None,
) -> Callable[[], None]:
    """Wire the bus into the CLI; returns an unsubscribe callable."""
    console = console or Console()
    state: dict = {"started": 0.0, "tokens": 0, "streaming": False}

    def on_event(ev: Event) -> None:
        if isinstance(ev, TaskStarted):
            _on_task_started(console, state, ev)
        elif isinstance(ev, PlanProposed):
            _on_plan(console, ev)
        elif isinstance(ev, Thought):
            _on_thought(console, ev)
        elif isinstance(ev, ToolStart):
            _on_tool_start(console, ev)
        elif isinstance(ev, ToolEnd):
            _on_tool_end(console, ev)
        elif isinstance(ev, TokensConsumed):
            state["tokens"] += ev.delta_in + ev.delta_out
        elif isinstance(ev, TokenStreamed):
            _on_token_streamed(console, state, ev)
        elif isinstance(ev, IterationCompleted):
            # Reserved for a future progress bar; no immediate render.
            pass
        elif isinstance(ev, TodoUpdated):
            _on_todos(console, ev)
        elif isinstance(ev, TaskFinished):
            _on_task_finished(console, state, ev)

    return bus.subscribe(on_event)


# ─────────────────────────────────────────────────────────────────────
# Per-event handlers
# ─────────────────────────────────────────────────────────────────────


def _on_task_started(console: Console, state: dict, ev: TaskStarted) -> None:
    import time
    state["started"] = time.time()
    state["tokens"] = 0

    # The REPL prompt already echoed the user's input, so we skip the
    # duplicate `❯ <query>` line and only show the (optional) intent
    # tag for debugging context.
    console.print()
    if ev.intent:
        tag = Text()
        tag.append("  ● ", style=blocks.ACCENT)
        tag.append(f"Intent: ", style=blocks.TEXT)
        tag.append(ev.intent.lower(), style=blocks.TEXT_MUTED)
        console.print(tag)
        console.print()


def _on_plan(console: Console, ev: PlanProposed) -> None:
    if not ev.steps:
        return
    head = Text()
    head.append("  ● ", style=blocks.ACCENT)
    head.append(f"Plan ", style=blocks.TEXT)
    head.append(f"({ev.complexity})", style=blocks.TEXT_MUTED)
    console.print(head)
    for i, step in enumerate(ev.steps, 1):
        line = Text()
        line.append(f"    {i}. ", style=blocks.TEXT_FAINT)
        line.append(escape(step), style=blocks.TEXT_MUTED)
        console.print(line)
    if ev.target_files:
        files = Text()
        files.append("    files: ", style=blocks.TEXT_FAINT)
        files.append(", ".join(escape(f) for f in ev.target_files), style=blocks.TEXT_MUTED)
        console.print(files)
    console.print()


def _on_thought(console: Console, ev: Thought) -> None:
    if not ev.text.strip():
        return
    line = Text()
    line.append("  ● ", style=blocks.ACCENT)
    line.append(escape(ev.text), style=blocks.TEXT)
    console.print(line)
    console.print()


def _on_todos(console: Console, ev: TodoUpdated) -> None:
    if not ev.items:
        return
    # Coerce items to (state, text) tuples; tolerate dict input from tools.
    coerced: list[tuple[str, str]] = []
    for it in ev.items:
        if isinstance(it, dict):
            coerced.append((str(it.get("state", "pending")), str(it.get("text", ""))))
        else:
            try:
                state, text = it
            except (TypeError, ValueError):
                continue
            coerced.append((str(state), str(text)))
    if coerced:
        blocks.render_todos(console, coerced, title=ev.title or "Update todos")
        console.print()


def _on_tool_start(console: Console, ev: ToolStart) -> None:
    target = _format_target(ev.name, ev.args)
    blocks.render_running(console, ev.display_verb or _verb_for(ev.name), target)


def _on_tool_end(console: Console, ev: ToolEnd) -> None:
    name = ev.name
    args = ev.args or {}
    verb = ev.display_verb or _verb_for(name)
    state = "ok" if ev.ok else "error"

    if name == "read_file":
        blocks.render_read(
            console,
            args.get("path", ""),
            num_lines=ev.meta.get("lines"),
            expandable=bool(ev.body),
        )
    elif name == "write_file":
        blocks.render_write(
            console,
            args.get("path", ""),
            ev.body or args.get("content", ""),
        )
    elif name == "edit_file":
        blocks.render_edit(
            console,
            args.get("path", ""),
            ev.body or "",
            additions=ev.meta.get("additions"),
            deletions=ev.meta.get("deletions"),
        )
    elif name == "bash":
        blocks.render_bash(
            console,
            args.get("command", ""),
            output=ev.body,
            exit_code=ev.meta.get("exit_code"),
        )
    elif name == "grep":
        blocks.render_grep(
            console,
            args.get("pattern", ""),
            path=args.get("path"),
            matches=ev.meta.get("matches"),
        )
    elif name == "list_dir":
        # Reuse the batch-header shape for parity with the reference UI.
        path = args.get("path", ".")
        blocks.render_listing_directory(console, path)
        if ev.body:
            for line in ev.body.splitlines()[:8]:
                console.print(f"        {escape(line)}", style=blocks.TEXT_MUTED)
            extra = max(0, len(ev.body.splitlines()) - 8)
            if extra:
                tail = Text()
                tail.append(f"    … +{extra} lines", style=blocks.TEXT_FAINT)
                console.print(tail)
    elif name == "find_files":
        target = args.get("glob", "")
        blocks.ToolBlock(
            verb=verb,
            target=target,
            summary=ev.summary,
            body=ev.body,
            state=state,
        ).render(console)
    else:
        # Generic fallback for any other tool.
        target = _format_target(name, args)
        blocks.ToolBlock(
            verb=verb,
            target=target,
            summary=ev.summary or ("ok" if ev.ok else (ev.error or "error")),
            body=ev.body,
            body_lang=ev.body_lang,
            state=state,
        ).render(console)
    console.print()


def _on_token_streamed(console: Console, state: dict, ev: TokenStreamed) -> None:
    if not state.get("streaming"):
        console.print()
        console.print("  ", end="")  # Small left padding for the answer text
        state["streaming"] = True
    console.print(ev.text, end="", style="white")


def _on_task_finished(console: Console, state: dict, ev: TaskFinished) -> None:
    import time
    if state.get("streaming"):
        console.print()
        console.print()
        state["streaming"] = False

    # Render token count and elapsed time in one beautiful line
    tokens = ev.tokens or state.get("tokens", 0)
    elapsed = ev.elapsed or (time.time() - state.get("started", time.time()))

    # Format tokens nicely: 782, 1.2k, 51k, etc.
    if tokens >= 1000:
        token_str = f"{tokens / 1000:.1f}k".replace(".0k", "k")
    else:
        token_str = str(tokens)

    status_line = Text()
    status_line.append("  ● ", style=blocks.ACCENT)
    status_line.append(token_str + " tokens", style=blocks.TEXT_MUTED)
    status_line.append("  ·  ", style=blocks.TEXT_FAINT)
    status_line.append(f"{elapsed:.1f}s", style=blocks.TEXT)
    console.print(status_line)
    console.print()


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _verb_for(name: str) -> str:
    return {
        "read_file": "Read",
        "write_file": "Write",
        "edit_file": "Edit",
        "bash": "Bash",
        "grep": "Grep",
        "list_dir": "List",
        "find_files": "Find",
        "git_status": "Git",
    }.get(name, name.replace("_", " ").title().replace(" ", ""))


def _format_target(name: str, args: dict) -> str:
    """Pick the best single-line representation of the tool call args."""
    if name in ("read_file", "write_file", "edit_file"):
        return str(args.get("path", ""))
    if name == "bash":
        return str(args.get("command", ""))
    if name == "grep":
        pat = repr(args.get("pattern", ""))
        path = args.get("path")
        return f"{pat} in {path}" if path and path != "." else pat
    if name == "list_dir":
        return str(args.get("path", "."))
    if name == "find_files":
        return str(args.get("glob", ""))
    if name == "git_status":
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in args.items())
