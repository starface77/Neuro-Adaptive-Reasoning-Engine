"""High-level UI helpers for the NARE CLI.

Bold, clean, professional. Orange accent (#D77757), no clutter.
This module is the canonical place for top-level rendering helpers
(banner, status, plan, solution, etc.).
"""

from contextlib import contextmanager
from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.theme import Theme as RichTheme
from rich.padding import Padding
from rich.spinner import Spinner as RichSpinner
from rich.table import Table  # re-exported for callers

from .theme import get_theme, ThemeName
from .animations import get_shimmer_color
from . import blocks

# Global theme state
_current_theme: ThemeName = "dark"

# Padding configuration
PADDING_LEFT = 0
PADDING_RIGHT = 0


def _build_rich_theme():
    """Build Rich theme from current NARA theme."""
    theme = get_theme(_current_theme)
    return RichTheme({
        "accent": theme.accent,
        "accent_shimmer": theme.accent_shimmer,
        "text": theme.text,
        "text_muted": theme.text_muted,
        "text_subtle": theme.text_subtle,
        "success": theme.success,
        "error": theme.error,
        "warning": theme.warning,
        "info": theme.info,
        "route_fast": theme.route_fast,
        "route_reflex": theme.route_reflex,
        "route_hybrid": theme.route_hybrid,
        "route_slow": theme.route_slow,
    })


console = Console(theme=_build_rich_theme())


def set_theme(name: ThemeName):
    """Set active theme and rebuild console."""
    global _current_theme, console
    _current_theme = name
    console = Console(theme=_build_rich_theme())


def _theme():
    """Get current theme."""
    return get_theme(_current_theme)


# ── Banner ────────────────────────────────────────────────────────────

def print_banner(repo_path: str = ".", mode: str = "Manual"):
    """Professional banner."""
    blocks.render_banner(console, repo_path=repo_path, mode=mode)


# ── Status & Info ─────────────────────────────────────────────────────

def print_status(label: str, value: str, style: str = "text"):
    """Print status line."""
    import time
    from .animations import get_shimmer_color

    current_time = time.time()
    shimmer = get_shimmer_color(current_time, speed=4.0)

    console.print(f"[{shimmer}]{label}:[/] [white]{value}[/]")


def print_intent(intent: str):
    """Silent — intent is shown via triage spinner."""
    pass


# ── Plan ──────────────────────────────────────────────────────────────

def print_plan(plan: dict):
    """Render execution plan with shimmer."""
    import time
    from .animations import get_shimmer_color
    from rich.markup import escape

    current_time = time.time()

    complexity = plan.get("complexity", "moderate")

    console.print()
    shimmer = get_shimmer_color(current_time, speed=3.0)
    console.print(f"[{shimmer}]Plan[/] [white]{complexity}[/]")

    steps = plan.get("plan_steps", [])
    if steps:
        console.print()
        for i, step in enumerate(steps, 1):
            step_shimmer = get_shimmer_color(current_time + i * 0.1, speed=3.0)
            safe_step = escape(step)
            console.print(f"[{step_shimmer}]{i}.[/] [white]{safe_step}[/]")

    console.print()


# ── Solution ──────────────────────────────────────────────────────────

def print_solution(answer: str, route: str, elapsed: float):
    """Print solution with parsed tool calls."""
    import time
    import re
    from .animations import get_shimmer_color
    from rich.markup import escape
    from rich.syntax import Syntax

    current_time = time.time()
    route_color = get_shimmer_color(current_time, speed=3.0)

    console.print()

    # Remove all XML tags and show clean output
    cleaned = answer

    # Parse <edit_file> tags. We accept both fully-closed and
    # truncated/streaming variants — when the LLM cuts off mid-content
    # we still want to render the partial block instead of dumping raw
    # XML to the terminal.
    edit_pattern = re.compile(
        r'<edit_file>\s*<path>(.*?)</path>\s*<diff>(.*?)(?:</diff>\s*</edit_file>|$)',
        re.DOTALL,
    )
    edits = edit_pattern.findall(answer)

    write_pattern = re.compile(
        r'<write_file>\s*<path>(.*?)</path>\s*<content>(.*?)(?:</content>\s*</write_file>|$)',
        re.DOTALL,
    )
    writes = write_pattern.findall(answer)

    read_pattern = re.compile(
        r'<read_file>\s*<path>(.*?)(?:</path>\s*</read_file>|$)',
        re.DOTALL,
    )
    reads = read_pattern.findall(answer)

    bash_pattern = re.compile(
        r'<bash_command>\s*<command>(.*?)(?:</command>\s*</bash_command>|$)',
        re.DOTALL,
    )
    bash_cmds = bash_pattern.findall(answer)

    # Strip all tool-call XML (closed or truncated) from the prose text.
    cleaned = edit_pattern.sub('', cleaned)
    cleaned = write_pattern.sub('', cleaned)
    cleaned = read_pattern.sub('', cleaned)
    cleaned = bash_pattern.sub('', cleaned)
    # Defensive: drop any remaining lone tool-tag fragments so the
    # terminal never shows raw `<write_file>...` to the user.
    cleaned = re.sub(
        r'</?(?:edit_file|write_file|read_file|bash_command|path|content|diff|command)>',
        '',
        cleaned,
    )
    cleaned = cleaned.strip()

    # Show clean text first
    if cleaned:
        safe_text = escape(cleaned)
        console.print(f"[white]{safe_text}[/]")
        console.print()

    # Show edits
    for path, diff in edits:
        blocks.render_edit(console, path.strip(), diff.strip())

    # Show writes
    for path, content in writes:
        blocks.render_write(console, path.strip(), content.strip())

    # Show reads
    for path in reads:
        blocks.render_read(console, path.strip(), num_lines=content_line_count(path.strip()))

    # Show bash commands
    for cmd in bash_cmds:
        blocks.render_bash(console, cmd.strip())

    blocks.render_status_line(console, route=route, elapsed=elapsed)
    console.print()


def content_line_count(path: str) -> int:
    """Best-effort line count for a file path; 0 if unreadable."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# ── Utilities ─────────────────────────────────────────────────────────

def print_file_loaded(path: str, lines: int):
    import time
    from .animations import get_shimmer_color
    from rich.markup import escape

    current_time = time.time()
    shimmer = get_shimmer_color(current_time, speed=4.0)

    safe_path = escape(path)
    console.print(f"[{shimmer}]{safe_path}[/] [white]{lines} lines[/]")


def print_error(msg: str):
    import time
    from .animations import get_shimmer_color
    from rich.markup import escape

    current_time = time.time()
    shimmer = get_shimmer_color(current_time, speed=3.0)
    safe_msg = escape(msg)

    console.print()
    console.print(f"[{shimmer}]Error:[/] [white]{safe_msg}[/]")
    console.print()


def print_warning(msg: str):
    import time
    from .animations import get_shimmer_color
    from rich.markup import escape

    current_time = time.time()
    shimmer = get_shimmer_color(current_time, speed=3.0)

    safe_msg = escape(msg)
    console.print(f"[{shimmer}]{safe_msg}[/]")


def print_success(msg: str):
    from rich.markup import escape
    safe_msg = escape(msg)
    console.print(f"[white]{safe_msg}[/]")


def print_code_changes(file_path: str, added_lines: list, line_number: int = None):
    """Show code changes with green background for added lines."""
    import time
    from .animations import get_shimmer_color
    from rich.markup import escape

    current_time = time.time()
    shimmer = get_shimmer_color(current_time, speed=3.0)

    console.print()
    safe_path = escape(file_path)
    console.print(f"[{shimmer}]Changes:[/]  [white]{safe_path}[/]")
    console.print()

    if line_number:
        console.print(f"[#999999]Line {line_number}[/]")

    for line in added_lines:
        # Green background for added lines - escape the line content
        safe_line = escape(line)
        console.print(f"[black on green]+ {safe_line}[/]")

    console.print()


def print_file_diff(file_path: str, old_content: str, new_content: str):
    """Show file diff with green background for additions."""
    import time
    from .animations import get_shimmer_color
    from rich.markup import escape

    current_time = time.time()
    shimmer = get_shimmer_color(current_time, speed=3.0)

    console.print()
    safe_path = escape(file_path)
    console.print(f"[{shimmer}]Modified:[/]  [white]{safe_path}[/]")
    console.print()

    old_lines = old_content.split('\n') if old_content else []
    new_lines = new_content.split('\n') if new_content else []

    # Simple diff - show last 3 lines of context + additions
    if len(new_lines) > len(old_lines):
        # Lines were added
        added_count = len(new_lines) - len(old_lines)
        start_idx = max(0, len(old_lines) - 3)

        # Show context
        for i in range(start_idx, len(old_lines)):
            safe_line = escape(old_lines[i])
            console.print(f"[#999999]{i+1:4d}[/]  [#505050]{safe_line}[/]")

        # Show additions with green background
        for i in range(len(old_lines), len(new_lines)):
            safe_line = escape(new_lines[i])
            console.print(f"[#999999]{i+1:4d}[/]  [black on green]+ {safe_line}[/]")
    else:
        # Show last 5 lines
        for i in range(max(0, len(new_lines) - 5), len(new_lines)):
            safe_line = escape(new_lines[i])
            console.print(f"[#999999]{i+1:4d}[/]  [white]{safe_line}[/]")

    console.print()


def confirm_plan() -> bool:
    """Ask user to confirm plan."""
    import time
    from .animations import get_shimmer_color

    current_time = time.time()
    shimmer = get_shimmer_color(current_time, speed=4.0)

    try:
        result = console.input(f"[{shimmer}]>[/] [white]Proceed?[/] [#999999](Y/n)[/] ").strip().lower()
        return result not in ('n', 'no')
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False


@contextmanager
def spinner(msg: str, animation: str = "dots"):
    """Non-blocking animated spinner with shimmer."""
    import time
    from .animations import get_shimmer_color
    start = time.time()

    class ShimmerSpinner:
        def __init__(self, msg):
            self.msg = msg
            self.start = start
            # Use ASCII spinner
            self.frames = ["|", "/", "-", "\\"]
            self.interval = 0.1

        def __rich__(self):
            elapsed = time.time() - self.start
            spinner_color = get_shimmer_color(elapsed, speed=5.0)
            text_color = get_shimmer_color(elapsed + 0.2, speed=4.0)

            text = Text()
            frame_idx = int(elapsed / self.interval) % len(self.frames)
            text.append(self.frames[frame_idx], style=spinner_color)
            text.append("  ", style="default")  # Two spaces for distance
            text.append(self.msg, style=text_color)
            return text

    shimmer = ShimmerSpinner(msg)
    with Live(shimmer, console=console, refresh_per_second=20, transient=True) as live:
        yield live

    # Final completion message
    final_time = time.time()
    final_shimmer = get_shimmer_color(final_time, speed=3.0)
    from rich.markup import escape
    safe_msg = escape(msg)
    console.print(f"[{final_shimmer}]Done[/] [white]{safe_msg}[/]")
