"""
Real-time file writing display with shimmer animations.

Shows file path and content being written with smooth animations.
"""

from contextlib import contextmanager
from rich.live import Live
from rich.text import Text
from rich.console import Group
from rich.syntax import Syntax
from rich.padding import Padding
from . import ui
from .animations import get_shimmer_color
import time
from nare.utils.logger import get_logger

log = get_logger(__name__)

class FileWritingDisplay:
    """Manages real-time display of file writing operations with animations."""

    def __init__(self):
        self.live = None
        self.filepath = ""
        self.content_buffer = ""

        self.spinner_frames = ["|", "/", "-", "\\"]
        self.frame_index = 0
        self.is_active = False
        self.start_time = time.time()

    def start_writing(self, filepath: str):
        """Start showing file writing for a specific file."""

        import os
        try:
            cwd = os.getcwd()
            if filepath.startswith(cwd):
                display_path = os.path.relpath(filepath, cwd)
            else:
                display_path = filepath
        except Exception as e:
            log.warning(f"[FileWriting] Failed to get relative path: {e}")
            display_path = filepath

        self.filepath = filepath
        self.display_path = display_path
        self.content_buffer = ""
        self.is_active = True
        self.start_time = time.time()
        self._update()

    def stream_content(self, chunk: str):
        """Stream a chunk of content being written."""
        if not self.is_active:
            return

        self.content_buffer += chunk
        self._update()

    def finish_writing(self):
        """Finish writing and show completion."""
        self.is_active = False

        if self.filepath and self.content_buffer:
            self._show_final_diff()

    def _show_final_diff(self):
        """Render a minimalist diff summary after a write completes."""
        import os
        from .animations import get_shimmer_color
        import time

        current_time = time.time()
        shimmer = get_shimmer_color(current_time, speed=3.0)

        display_path = getattr(self, 'display_path', self.filepath)

        old_content = ""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    old_content = f.read()
            except Exception as e:
                log.warning(f"[FileWriting] Failed to read existing file: {e}")

        from . import ui

        ui.console.print()
        if old_content:
            ui.console.print(f"  [{shimmer}]+  Modified[/]  [white]{display_path}[/]")
        else:
            ui.console.print(f"  [{shimmer}]+  Created[/]  [white]{display_path}[/]")

        ui.console.print()

    def _update(self):
        """Refresh the live display with the spinner and recent buffer lines."""
        if not self.live or not self.is_active:
            return

        renderables = []

        self.frame_index = (self.frame_index + 1) % len(self.spinner_frames)
        spinner_char = self.spinner_frames[self.frame_index]

        display_path = getattr(self, 'display_path', self.filepath)

        elapsed = time.time() - self.start_time
        spinner_style = get_shimmer_color(elapsed, speed=5.0)

        status = Text()
        status.append(spinner_char, style=spinner_style)
        status.append("  ", style="default")

        text_color = get_shimmer_color(elapsed + 0.3, speed=4.0)
        status.append(display_path, style=text_color)

        renderables.append(status)
        renderables.append(Text(""))

        if self.content_buffer:
            lines = self.content_buffer.split('\n')
            show_lines = lines[-5:] if len(lines) > 5 else lines
            for line in show_lines:
                renderables.append(Text(f"  {line}", style="#999999"))

        self.live.update(Group(*renderables))

    @contextmanager
    def show(self):
        """Context manager for live display."""
        self.filepath = ""
        self.content_buffer = ""
        self.is_active = False
        self.frame_index = 0
        self.start_time = time.time()
        self.shimmer = None

        self.live = Live(
            Text("", style="#999999"),
            console=ui.console,
            refresh_per_second=30,
            transient=True,
        )
        with self.live:
            yield self

        self.live = None

_file_writing = FileWritingDisplay()

def get_file_writing_display() -> FileWritingDisplay:
    """Get the global file writing display."""
    return _file_writing

@contextmanager
def show_file_writing():
    """Show file writing display for a block of code."""
    with _file_writing.show():
        yield _file_writing
