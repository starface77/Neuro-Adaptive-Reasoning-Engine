
from contextlib import contextmanager
from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.theme import Theme as RichTheme
from rich.padding import Padding
from rich.spinner import Spinner as RichSpinner
from rich.table import Table

from .theme import get_theme, ThemeName
from .animations import get_shimmer_color
from . import blocks

_current_theme: ThemeName = "dark"

PADDING_LEFT = 0
PADDING_RIGHT = 0

def _build_rich_theme():
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
        "route_compiled_skill": theme.route_compiled_skill,
        "route_reflex": theme.route_reflex,
        "route_hybrid": theme.route_hybrid,
        "route_slow": theme.route_slow,
    })

console = Console(theme=_build_rich_theme(), force_terminal=True, force_interactive=True)

def set_theme(name: ThemeName):
    global _current_theme, console
    _current_theme = name
    console = Console(theme=_build_rich_theme(), force_terminal=True, force_interactive=True)

def _theme():
    return get_theme(_current_theme)

def print_banner(repo_path: str = ".", mode: str = "Manual"):
    blocks.render_banner(console, repo_path=repo_path, mode=mode)

def print_status(label: str, value: str, style: str = "text"):
    from rich.markup import escape
    safe_value = escape(value)
    style_color = {
        "info": "#5599FF",
        "success": "#4EBA65",
        "warning": "#FFC107",
        "error": "#FF6B80",
    }.get(style, "#FFFFFF")
    console.print(f"  [#D77757]{label}:[/] [{style_color}]{safe_value}[/]")

def print_intent(intent: str):
    pass

def print_plan(plan: dict):
    from rich.markup import escape

    complexity = plan.get("complexity", "moderate")
    complexity_colors = {
        "simple": "#4EBA65",
        "moderate": "#FFC107",
        "complex": "#D77757",
        "very_complex": "#FF6B80",
    }
    c_color = complexity_colors.get(complexity, "#FFC107")

    console.print()
    console.print(f"  [#D77757]◆[/]  [bold white]Plan[/]  [{c_color}]{complexity}[/]")
    console.print()

    steps = plan.get("plan_steps", [])
    if steps:
        for i, step in enumerate(steps, 1):
            safe_step = escape(step)
            console.print(f"    [#D77757]{i}.[/] [white]{safe_step}[/]")

    console.print()

def print_solution(answer: str, route: str, elapsed: float):
    import re
    from rich.markup import escape

    console.print()

    cleaned = answer

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

    cleaned = edit_pattern.sub('', cleaned)
    cleaned = write_pattern.sub('', cleaned)
    cleaned = read_pattern.sub('', cleaned)
    cleaned = bash_pattern.sub('', cleaned)

    cleaned = re.sub(
        r'</?(?:edit_file|write_file|read_file|bash_command|path|content|diff|command)>',
        '',
        cleaned,
    )
    cleaned = cleaned.strip()

    if cleaned:
        safe_text = escape(cleaned)
        console.print(f"[white]{safe_text}[/]")
        console.print()

    for path, diff in edits:
        blocks.render_edit(console, path.strip(), diff.strip())

    for path, content in writes:
        blocks.render_write(console, path.strip(), content.strip())

    for path in reads:
        blocks.render_read(console, path.strip(), num_lines=content_line_count(path.strip()))

    for cmd in bash_cmds:
        blocks.render_bash(console, cmd.strip())

    blocks.render_status_line(console, route=route, elapsed=elapsed)
    console.print()

def content_line_count(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0

def print_file_loaded(path: str, lines: int):
    from rich.markup import escape
    safe_path = escape(path)
    console.print(f"  [#4EBA65]●[/]  [#D77757]{safe_path}[/]  [#666666]{lines} lines[/]")

def print_error(msg: str):
    from rich.markup import escape
    safe_msg = escape(msg)
    console.print()
    console.print(f"  [#FF6B80]✕[/]  [#FF6B80]Error:[/] [white]{safe_msg}[/]")
    console.print()

def print_warning(msg: str):
    from rich.markup import escape
    safe_msg = escape(msg)
    console.print(f"  [#FFC107]△[/]  [#999999]{safe_msg}[/]")

def print_success(msg: str):
    from rich.markup import escape
    safe_msg = escape(msg)
    console.print(f"  [#4EBA65]✓[/]  [white]{safe_msg}[/]")

def print_code_changes(file_path: str, added_lines: list, line_number: int = None):
    from rich.markup import escape

    console.print()
    safe_path = escape(file_path)
    console.print(f"[#D77757]Changes:[/]  [white]{safe_path}[/]")
    console.print()

    if line_number:
        console.print(f"[#999999]Line {line_number}[/]")

    for line in added_lines:

        safe_line = escape(line)
        console.print(f"[black on green]+ {safe_line}[/]")

    console.print()

def print_file_diff(file_path: str, old_content: str, new_content: str):
    from rich.markup import escape

    console.print()
    safe_path = escape(file_path)
    console.print(f"[#D77757]Modified:[/]  [white]{safe_path}[/]")
    console.print()

    old_lines = old_content.split('\n') if old_content else []
    new_lines = new_content.split('\n') if new_content else []

    if len(new_lines) > len(old_lines):

        added_count = len(new_lines) - len(old_lines)
        start_idx = max(0, len(old_lines) - 3)

        for i in range(start_idx, len(old_lines)):
            safe_line = escape(old_lines[i])
            console.print(f"[#999999]{i+1:4d}[/]  [#505050]{safe_line}[/]")

        for i in range(len(old_lines), len(new_lines)):
            safe_line = escape(new_lines[i])
            console.print(f"[#999999]{i+1:4d}[/]  [black on green]+ {safe_line}[/]")
    else:

        for i in range(max(0, len(new_lines) - 5), len(new_lines)):
            safe_line = escape(new_lines[i])
            console.print(f"[#999999]{i+1:4d}[/]  [white]{safe_line}[/]")

    console.print()

def confirm_plan() -> bool:
    try:
        result = console.input("[#D77757]>[/] [white]Proceed?[/] [#999999](Y/n)[/] ").strip().lower()
        return result not in ('n', 'no')
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False

@contextmanager
def spinner(msg: str, animation: str = "dots"):
    import time
    start = time.time()

    class SimpleSpinner:
        FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        def __init__(self, msg):
            self.msg = msg
            self.start = start

        def __rich__(self):
            elapsed = time.time() - self.start
            text = Text()
            frame_idx = int(elapsed / 0.08) % len(self.FRAMES)
            text.append(f"  {self.FRAMES[frame_idx]} ", style="#D77757")
            text.append(self.msg, style="#999999")
            if elapsed > 2.0:
                text.append(f"  {elapsed:.0f}s", style="#555555")
            return text

    s = SimpleSpinner(msg)
    with Live(s, console=console, refresh_per_second=12, transient=True) as live:
        yield live

    from rich.markup import escape
    elapsed = time.time() - start
    safe_msg = escape(msg)
    console.print(f"  [#4EBA65]✓[/]  [white]{safe_msg}[/]  [#555555]{elapsed:.1f}s[/]")
