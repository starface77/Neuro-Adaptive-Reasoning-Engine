
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

ACCENT = "#D77757"
ACCENT_DIM = "#a25a40"
ACCENT_LIGHT = "#EB9F7F"
TEXT = "#FFFFFF"
TEXT_MUTED = "#999999"
TEXT_SUBTLE = "#666666"
TEXT_FAINT = "#444444"
SUCCESS = "#4EBA65"
ERROR = "#FF6B80"
WARNING = "#FFC107"
INFO = "#5599FF"
BORDER_CHAR = "─"

ROUTE_PALETTE = {
    "DIRECT":         "#888888",
    "FAST":           "#4EBA65",
    "COMPILED_SKILL": "#4EBA65",
    "REFLEX":         "#B1B9F9",
    "HYBRID":         "#FFC107",
    "SLOW":           "#D77757",
    "SLOW-RETRY":     "#D77757",
    "SLOW-PATH-FIX":  "#D77757",
    "AGENT":          "#D77757",
}

ROUTE_ICONS = {
    "DIRECT":         "→",
    "FAST":           "✦",
    "COMPILED_SKILL": "★",
    "REFLEX":         "↯",
    "HYBRID":         "◈",
    "SLOW":           "✻",
    "SLOW-RETRY":     "✻",
    "SLOW-PATH-FIX":  "✻",
    "AGENT":          "◆",
}

def render_banner(console: Console, repo_path: str, mode: str = "Manual") -> None:
    repo = os.path.basename(os.path.abspath(repo_path)) or repo_path
    safe_repo = escape(repo)
    safe_full = escape(repo_path)
    safe_mode = escape(mode)

    width = max(20, console.size.width if hasattr(console, "size") else 80)
    rule_len = min(width - 4, 60)

    console.print()
    console.print(f"  [dim]{BORDER_CHAR * rule_len}[/dim]")
    console.print()
    console.print(
        Text.assemble(
            ("  ◆ ", f"bold {ACCENT}"),
            ("NARE", f"bold {TEXT}"),
            ("  ", ""),
            ("reasoning agent for software engineering", TEXT_SUBTLE),
        )
    )
    console.print(
        Text.assemble(
            ("    NareCLI ", TEXT_MUTED),
            (safe_full, TEXT_FAINT),
        )
    )
    console.print()
    console.print(
        Text.assemble(
            ("    ", ""),
            (safe_mode + " mode", TEXT_SUBTLE),
            ("  ·  ", TEXT_FAINT),
            ("type ", TEXT_FAINT),
            ("/help", TEXT_MUTED),
            (" for commands, ", TEXT_FAINT),
            ("Tab", TEXT_MUTED),
            (" to cycle modes", TEXT_FAINT),
        )
    )
    console.print(
        Text.assemble(
            ("    ", ""),
            ("Use ", TEXT_FAINT),
            ("/agent", f"bold {ACCENT}"),
            (" for deep autonomous reasoning — ", TEXT_FAINT),
            ("higher accuracy, longer runtime, increased API usage", TEXT_MUTED),
        )
    )
    console.print()
    console.print(f"  [dim]{BORDER_CHAR * rule_len}[/dim]")
    console.print()

@dataclass
class ToolBlock:

    verb: str
    target: str
    summary: Optional[str] = None
    body: Optional[str] = None
    body_lang: Optional[str] = None
    state: str = "ok"
    expandable: bool = False
    body_numbered: bool = False

    _STATE_DOT = {
        "ok":      ("●", SUCCESS),
        "running": ("◌", WARNING),
        "error":   ("✕", ERROR),
    }

    def render(self, console: Console, max_body_lines: int = 8) -> None:
        dot, dot_color = self._STATE_DOT.get(self.state, self._STATE_DOT["ok"])

        width = max(20, console.size.width if hasattr(console, "size") else 80)

        budget = max(8, width - len(self.verb) - 6)
        target = self.target
        if len(target) > budget:
            head = budget // 3
            tail = budget - head - 1
            target = target[:head] + "…" + target[-tail:]
        safe_target = escape(target)

        header = Text()
        header.append(f"  {dot} ", style=dot_color)
        header.append(f"{self.verb}(", style=TEXT)
        header.append(safe_target, style=ACCENT)
        header.append(")", style=TEXT)

        console.print(header)

        if self.summary:
            summary = Text()
            summary.append("    └ ", style=TEXT_FAINT)
            summary.append(escape(self.summary), style=TEXT_MUTED)
            console.print(summary)

        if self.body:
            lines = self.body.splitlines()
            shown = lines[:max_body_lines]
            extra = len(lines) - len(shown)
            indent = "      "
            if self.body_lang == "diff":
                for line in shown:
                    if line.startswith("+++") or line.startswith("---"):
                        continue
                    if line.startswith("@@"):
                        console.print(f"{indent}{escape(line)}", style=TEXT_FAINT)
                    elif line.startswith("+"):
                        console.print(f"{indent}{escape(line)}", style="#4EBA65")
                    elif line.startswith("-"):
                        console.print(f"{indent}{escape(line)}", style="#FF6B80")
                    else:
                        console.print(f"{indent}{escape(line)}", style=TEXT_MUTED)
            elif self.body_numbered:
                width = len(str(len(lines)))
                for i, line in enumerate(shown, 1):
                    num = Text()
                    num.append(f"{indent}{i:>{width}} ", style=TEXT_FAINT)
                    num.append(escape(line), style=TEXT_MUTED)
                    console.print(num)
            else:
                for line in shown:
                    console.print(f"{indent}{escape(line)}", style=TEXT_MUTED)
            if extra > 0:
                tail = Text()
                tail.append(f"    … +{extra} lines", style=TEXT_FAINT)
                console.print(tail)

def render_running(console: Console, verb: str, target: str) -> None:
    ToolBlock(verb, target, summary="Running…", state="running").render(console)

def render_read(
    console: Console,
    path: str,
    num_lines: Optional[int] = None,
    *,
    expandable: bool = False,
) -> None:
    summary = f"{num_lines} lines" if num_lines is not None else None
    ToolBlock(
        "Read", path, summary=summary, expandable=expandable,
    ).render(console)

def render_write(
    console: Console,
    path: str,
    content: str,
    *,
    lang: Optional[str] = None,
    preview_lines: int = 9,
) -> None:
    line_count = content.count("\n") + 1 if content else 0
    inferred = lang or _ext_lang(path)
    ToolBlock(
        "Write",
        path,
        summary=f"Wrote {line_count} lines to {path}",
        body=content,
        body_lang=inferred,
        body_numbered=True,
    ).render(console, max_body_lines=preview_lines)

def render_edit(
    console: Console,
    path: str,
    diff: str,
    *,
    additions: Optional[int] = None,
    deletions: Optional[int] = None,
) -> None:
    if additions is None or deletions is None:
        additions = sum(
            1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
        )
        deletions = sum(
            1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")
        )
    summary = f"+{additions}  -{deletions}"
    ToolBlock(
        "Edit",
        path,
        summary=summary,
        body=diff,
        body_lang="diff",
    ).render(console)

def render_hunks(
    console: Console,
    summary: str,
    hunks: str,
) -> None:
    additions = sum(
        1 for line in hunks.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    deletions = sum(
        1 for line in hunks.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )

    change_summary = f"{summary}  (+{additions} -{deletions})"

    ToolBlock(
        "Patch",
        "hunks",
        summary=change_summary,
        body=hunks,
        body_lang="diff",
    ).render(console)

def render_bash(
    console: Console,
    command: str,
    *,
    output: Optional[str] = None,
    exit_code: Optional[int] = None,
) -> None:
    state = "error" if exit_code not in (None, 0) else "ok"
    has_output = bool(output and output.strip())

    if state == "error":

        summary = f"exit {exit_code}"
    elif has_output:
        summary = None
    else:
        summary = "Done"

    ToolBlock(
        "Bash",
        command,
        summary=summary,
        body=output if has_output else None,
        state=state,
    ).render(console)

def render_grep(
    console: Console,
    pattern: str,
    *,
    path: Optional[str] = None,
    matches: Optional[int] = None,
) -> None:
    target = pattern if not path else f'{pattern}  in {path}'
    summary = None if matches is None else (
        f"{matches} match" if matches == 1 else f"{matches} matches"
    )
    ToolBlock("Grep", target, summary=summary).render(console)

def _pluralize(n: int, singular: str, plural: Optional[str] = None) -> str:
    return singular if n == 1 else (plural or singular + "s")

def render_batch_header(
    console: Console,
    text: str,
    *,
    expandable: bool = True,
) -> None:
    line = Text("  ")
    line.append(escape(text), style=TEXT)

    _ = expandable
    console.print(line)

def render_reading_files(
    console: Console,
    paths: Sequence[str],
    *,
    expandable: bool = True,
) -> None:
    n = len(paths)
    render_batch_header(
        console,
        f"Reading {n} {_pluralize(n, 'file')}…",
        expandable=expandable,
    )
    for path in paths:
        console.print(
            Text.assemble(
                ("    └ ", TEXT_FAINT),
                (escape(path), TEXT_MUTED),
            )
        )

def render_searching(
    console: Console,
    pattern: str,
    *,
    in_files: Optional[int] = None,
    expandable: bool = True,
) -> None:
    head = "Searching for 1 pattern"
    if in_files is not None:
        head += f", reading {in_files} {_pluralize(in_files, 'file')}"
    render_batch_header(console, head + "…", expandable=expandable)
    console.print(
        Text.assemble(
            ("    └ ", TEXT_FAINT),
            (escape(repr(pattern)), TEXT_MUTED),
        )
    )

def render_listing_directory(
    console: Console,
    path: str,
    *,
    expandable: bool = True,
) -> None:
    render_batch_header(
        console,
        "Listing 1 directory…",
        expandable=expandable,
    )
    console.print(
        Text.assemble(
            ("    └ ", TEXT_FAINT),
            ("$ ls -la ", TEXT_MUTED),
            (escape(path), TEXT),
        )
    )

def render_diff(
    console: Console,
    path: str,
    old: str,
    new: str,
    *,
    context: int = 2,
    max_lines: int = 30,
) -> None:
    import difflib

    safe_path = escape(path)
    console.print()
    console.print(Text.assemble(("  Modified  ", ACCENT), (safe_path, TEXT)))
    console.print()

    old_lines = old.splitlines() if old else []
    new_lines = new.splitlines() if new else []

    diff = list(
        difflib.unified_diff(
            old_lines, new_lines, lineterm="", n=context,
        )
    )
    if not diff:
        console.print("    (no changes)", style=TEXT_FAINT)
        console.print()
        return

    shown = diff[:max_lines]
    for line in shown:
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            console.print(f"    {escape(line)}", style=TEXT_FAINT)
        elif line.startswith("+"):
            console.print(f"    {escape(line)}", style="#4EBA65")
        elif line.startswith("-"):
            console.print(f"    {escape(line)}", style="#FF6B80")
        else:
            console.print(f"    {escape(line)}", style=TEXT_MUTED)

    if len(diff) > max_lines:
        console.print(
            f"    ⋯ {len(diff) - max_lines} more diff lines",
            style=TEXT_FAINT,
        )
    console.print()

def render_status_line(
    console: Console,
    *,
    mode: Optional[str] = None,
    model: Optional[str] = None,
    repo: Optional[str] = None,
    route: Optional[str] = None,
    elapsed: Optional[float] = None,
    tokens: Optional[int] = None,
    episodes: Optional[int] = None,
    skills: Optional[int] = None,
) -> None:
    entries: list[tuple[int, Text]] = []

    if mode:
        entries.append((3, Text(mode, style=ACCENT)))
    if model:
        entries.append((2, Text(model, style=TEXT_MUTED)))
    if repo:
        entries.append((2, Text(repo, style=TEXT_SUBTLE)))
    if skills is not None and skills > 0:
        sk_text = Text()
        sk_text.append("★ ", style=SUCCESS)
        sk_text.append(f"{skills} skills", style=TEXT_FAINT)
        entries.append((1, sk_text))
    if episodes is not None and episodes > 0:
        ep_text = Text()
        ep_text.append(f"{episodes} episodes", style=TEXT_FAINT)
        entries.append((1, ep_text))
    if route:
        color = ROUTE_PALETTE.get(route, TEXT_MUTED)
        icon = ROUTE_ICONS.get(route, "◆")
        right = Text()
        right.append(f"{icon} ", style=color)
        right.append(route, style=color)
        if elapsed is not None:
            right.append(f"  {elapsed:.1f}s", style=TEXT_FAINT)

        entries.append((4, right))

    if not entries:
        return

    width = max(20, console.size.width if hasattr(console, "size") else 80)

    def _build(items: list[Text]) -> Text:
        line = Text("  ")
        for i, p in enumerate(items):
            if i:
                line.append("  ·  ", style=TEXT_FAINT)
            line.append(p)
        return line

    sorted_entries = sorted(range(len(entries)), key=lambda i: entries[i][0])
    drop = 0
    while True:
        keep_idxs = set(sorted_entries[drop:])
        items = [t for i, (_, t) in enumerate(entries) if i in keep_idxs]
        line = _build(items)
        if line.cell_len <= width or drop >= len(sorted_entries) - 1:
            break
        drop += 1
    console.print(line, soft_wrap=True)

def render_todos(
    console: Console,
    items: Sequence[tuple[str, str]],
    *,
    title: str = "Update todos",
) -> None:
    header = Text()
    header.append("  ● ", style=ACCENT)
    header.append(title, style=TEXT)
    console.print(header)

    done_count = sum(1 for s, _ in items if s == "done")
    total = len(items)
    if total > 0:
        progress = Text()
        progress.append(f"    {done_count}/{total} completed", style=TEXT_FAINT)
        console.print(progress)

    width = len(str(len(items))) if items else 1
    for i, (state, text) in enumerate(items, 1):
        line = Text()
        line.append(f"    {i:>{width}}. ", style=TEXT_FAINT)
        if state == "done":
            line.append("[", style=TEXT_FAINT)
            line.append("✓", style=SUCCESS)
            line.append("] ", style=TEXT_FAINT)
            line.append(escape(text), style=TEXT_MUTED)
        elif state == "in_progress":
            line.append("[", style=TEXT_FAINT)
            line.append("▸", style=ACCENT)
            line.append("] ", style=TEXT_FAINT)
            line.append(escape(text), style=TEXT)
        else:
            line.append("[ ] ", style=TEXT_FAINT)
            line.append(escape(text), style=TEXT_MUTED)
        console.print(line)

def render_command_table(
    console: Console, commands: Iterable[tuple[str, Sequence[str], str]]
) -> None:
    console.print()
    console.print(
        Text.assemble(
            ("  ◆ ", f"bold {ACCENT}"),
            ("Available Commands", f"bold {TEXT}"),
        )
    )
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False)
    table.add_column("cmd", style=ACCENT, min_width=18)
    table.add_column("desc", style=TEXT_MUTED)

    for name, aliases, help_text in commands:
        label = f"/{name}"
        if aliases:
            label += "  " + " ".join(f"/{a}" for a in aliases)
        table.add_row(label, help_text or "")

    console.print(Padding(table, (0, 0, 0, 2)))
    console.print()
    console.print(
        Text.assemble(
            ("    ", ""),
            ("Tip: ", TEXT_MUTED),
            ("Tab", f"bold {TEXT_MUTED}"),
            (" cycles modes, ", TEXT_FAINT),
            ("Ctrl+L", f"bold {TEXT_MUTED}"),
            (" clears screen, ", TEXT_FAINT),
            ("Ctrl+D", f"bold {TEXT_MUTED}"),
            (" exits", TEXT_FAINT),
        )
    )
    console.print()

def render_separator(console: Console, width: Optional[int] = None) -> None:
    w = width or min((console.size.width if hasattr(console, "size") else 80) - 4, 60)
    console.print(f"  [dim]{BORDER_CHAR * w}[/dim]")

def confirm(console: Console, prompt: str, default_yes: bool = True) -> bool:
    suffix = "(Y/n)" if default_yes else "(y/N)"
    try:
        raw = console.input(
            Text.assemble(
                ("  ? ", ACCENT),
                (prompt, TEXT),
                ("  ", ""),
                (suffix, TEXT_FAINT),
                ("  ", ""),
            )
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
    if not raw:
        return default_yes
    return raw in ("y", "yes", "д", "да")

def confirm_action(
    console: Console,
    *,
    title: str,
    body_lines: Sequence[str],
    description: Optional[str] = None,
    options: Sequence[str] = ("Yes", "No"),
    default_index: int = 0,
    footer_hints: Sequence[str] = (
        "Esc to cancel",
        "Tab to amend",
        "ctrl+e to explain",
    ),
) -> int:
    console.print()
    console.print(Text.assemble((f"  {title}", f"bold {ACCENT}")))
    console.print()
    for line in body_lines:
        console.print(f"  {escape(line)}", style=TEXT)
    if description:
        console.print(f"  {escape(description)}", style=TEXT_MUTED)
    console.print()
    console.print("  Do you want to proceed?", style=TEXT)
    for i, opt in enumerate(options):
        marker = "▸ " if i == default_index else "  "
        marker_style = ACCENT if i == default_index else TEXT_FAINT
        line = Text()
        line.append(f"  {marker}", style=marker_style)
        line.append(f"{i + 1}. ", style=ACCENT if i == default_index else TEXT_MUTED)
        line.append(opt, style=TEXT if i == default_index else TEXT_MUTED)
        console.print(line)
    console.print()
    if footer_hints:
        hint = Text("  ")
        for i, h in enumerate(footer_hints):
            if i:
                hint.append("  ·  ", style=TEXT_FAINT)
            hint.append(h, style=TEXT_FAINT)
        console.print(hint)
    console.print()

    try:
        raw = console.input(
            Text.assemble(("  > ", ACCENT), ("", TEXT))
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return -1

    if not raw:
        return default_index
    if raw in ("q", "esc", "n", "no", "cancel"):
        return -1
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return idx

    for i, opt in enumerate(options):
        if opt.lower().startswith(raw):
            return i
    return -1

class LiveStatus:

    SPINNER_FRAMES = ["+", "✦", "*", "·"]

    DEFAULT_VERBS = (
        "Thinking",
        "Reasoning",
        "Routing",
        "Synthesizing",
        "Compiling",
        "Reflecting",
        "Working",
    )

    def __init__(
        self,
        verb: str = "Thinking",
        *,
        console: Optional[Console] = None,
        refresh_per_second: int = 8,
    ):
        from rich.live import Live

        self._console = console
        self._verb = verb
        self._tokens_in = 0
        self._tokens_out = 0
        self._start: Optional[float] = None
        self._frame = 0
        self._live: Optional[Live] = None
        self._refresh = refresh_per_second

    def update(self, *, verb: Optional[str] = None) -> None:
        if verb is not None:
            self._verb = verb
        self._render()

    def bump_tokens(self, out: int = 0, in_: int = 0) -> None:
        self._tokens_in += in_
        self._tokens_out += out
        self._render()

    def __enter__(self) -> "LiveStatus":
        import time
        from rich.live import Live

        self._start = time.time()
        self._live = Live(
            self._render_text(),
            console=self._console,
            refresh_per_second=self._refresh,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc_val, exc_tb)
        self._live = None

    def _render(self) -> None:
        if self._live is not None:
            self._live.update(self._render_text())

    def _render_text(self) -> Text:
        import time

        elapsed = 0.0 if self._start is None else (time.time() - self._start)
        self._frame = (self._frame + 1) % len(self.SPINNER_FRAMES)
        spinner = self.SPINNER_FRAMES[self._frame]

        line = Text("  ")
        line.append(f"{spinner} ", style=ACCENT)
        line.append(self._verb + "… ", style=ACCENT_DIM)
        line.append("(", style=TEXT_FAINT)
        line.append(f"{elapsed:.0f}s", style=TEXT_MUTED)
        if self._tokens_in or self._tokens_out:
            total = self._tokens_in + self._tokens_out
            line.append(" · ", style=TEXT_FAINT)
            line.append("↑ ", style=TEXT_FAINT)
            line.append(f"{_human_tokens(total)} tokens", style=TEXT_MUTED)
        line.append(")", style=TEXT_FAINT)
        return line

def _ext_lang(path: str) -> Optional[str]:
    ext = (os.path.splitext(path)[1] or "").lower().lstrip(".")
    mapping = {
        "py": "python",
        "ts": "typescript", "tsx": "tsx",
        "js": "javascript", "jsx": "jsx",
        "rs": "rust", "go": "go",
        "c": "c", "h": "c", "cpp": "cpp", "hpp": "cpp",
        "java": "java",
        "rb": "ruby", "php": "php", "sh": "bash", "bash": "bash",
        "yml": "yaml", "yaml": "yaml",
        "json": "json", "toml": "toml", "ini": "ini",
        "md": "markdown", "rst": "rst",
        "html": "html", "css": "css",
        "sql": "sql",
    }
    return mapping.get(ext)

def _human_tokens(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        v = n / 1000
        return f"{v:.1f}k" if v < 10 else f"{int(v)}k"
    v = n / 1_000_000
    return f"{v:.1f}M"
