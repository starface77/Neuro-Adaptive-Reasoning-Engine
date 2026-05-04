"""Display components for NARE CLI.

Public surface:

- High-level helpers (`ui.print_*`, `console`, `show_thinking`).
- Reusable visual blocks (`blocks.render_*`) for tool calls, diffs,
  status lines, and the slash-command help table.
"""

from .thinking import ThinkingDisplay, get_thinking_display, show_thinking
from .file_writing import FileWritingDisplay, get_file_writing_display, show_file_writing
from .ui import (
    console,
    set_theme,
    print_banner,
    print_status,
    print_intent,
    print_plan,
    print_solution,
    print_file_loaded,
    print_error,
    print_warning,
    print_success,
    confirm_plan,
    spinner,
)
from . import blocks
from .blocks import (
    ToolBlock,
    LiveStatus,
    render_banner,
    render_read,
    render_write,
    render_edit,
    render_bash,
    render_grep,
    render_running,
    render_batch_header,
    render_reading_files,
    render_searching,
    render_listing_directory,
    render_diff,
    render_status_line,
    render_command_table,
    confirm,
    confirm_action,
)

__all__ = [
    "ThinkingDisplay",
    "get_thinking_display",
    "show_thinking",
    "FileWritingDisplay",
    "get_file_writing_display",
    "show_file_writing",
    "console",
    "set_theme",
    "print_banner",
    "print_status",
    "print_intent",
    "print_plan",
    "print_solution",
    "print_file_loaded",
    "print_error",
    "print_warning",
    "print_success",
    "confirm_plan",
    "spinner",
    # New: blocks module + renderers
    "blocks",
    "ToolBlock",
    "LiveStatus",
    "render_banner",
    "render_read",
    "render_write",
    "render_edit",
    "render_bash",
    "render_grep",
    "render_running",
    "render_batch_header",
    "render_reading_files",
    "render_searching",
    "render_listing_directory",
    "render_diff",
    "render_status_line",
    "render_command_table",
    "confirm",
    "confirm_action",
]
