"""
Real-time thinking display — premium streaming UI.

Shows LLM reasoning in gray, solution in white, with beautiful
animated transitions between phases. Thread-safe and flicker-free.
"""

import time
import math
from contextlib import contextmanager
from rich.text import Text
from . import ui
from .spinner import WaitingSpinner


class ThinkingDisplay:
    """Manages real-time display of LLM reasoning and solution tokens.
    
    Phases:
    1. THINKING — gray text, reasoning tokens stream
    2. SOLUTION — white text, final answer streams
    
    Transitions between phases are animated with color shifts.
    """

    def __init__(self):
        self.live = None
        self.buffer = ""
        self.mode = "thinking"
        self.solution_lines_printed = 0
        self.max_solution_lines = 50
        self.in_code_block = False
        self.code_block_lines = 0
        self.start_time = time.time()
        self.frame_idx = 0
        self.progress_current = 0
        self.progress_total = 0
        self.progress_message = ""
        self.waiting_spinner = None
        self.in_xml_tag = False  # Track if we're inside XML tag
        self.xml_buffer = ""  # Buffer for XML tag content

    def _stop_live_and_spinner(self):
        """Helper to stop any running spinners/live displays before printing."""
        if self.waiting_spinner:
            self.waiting_spinner.stop()
            self.waiting_spinner = None

    def switch_to_solution(self):
        """Switch from thinking mode to solution streaming mode."""
        self._stop_live_and_spinner()
        self.mode = "solution"
        self.solution_lines_printed = 0
        self.in_code_block = False
        self.code_block_lines = 0

    def stream_token(self, token: str):
        """Stream a single token from LLM output."""
        # Stop spinner as soon as we start streaming tokens to avoid console clashes
        self._stop_live_and_spinner()

        # Filter out closing tags
        if token.strip() in ['</solution>', '</reasoning>', '</abstract_signature>', '</delta_reasoning>']:
            return

        if self.mode == "solution":
            # CRITICAL: Filter out XML tool call tags during streaming
            # Check for XML tag start
            if '<read_file>' in token or '<edit_file>' in token or '<write_file>' in token or '<bash_command>' in token:
                self.in_xml_tag = True
                self.xml_buffer = token
                return

            # If inside XML tag, buffer it and don't print
            if self.in_xml_tag:
                self.xml_buffer += token
                # Check for closing tags
                if '</read_file>' in self.xml_buffer or '</edit_file>' in self.xml_buffer or '</write_file>' in self.xml_buffer or '</bash_command>' in self.xml_buffer:
                    self.in_xml_tag = False
                    self.xml_buffer = ""
                return

            # Check for code block markers or XML content tags
            if '```' in token or '<content>' in token:
                self.in_code_block = True
                self.code_block_lines = 0
            if '```' in token and self.in_code_block and self.code_block_lines > 0:
                self.in_code_block = False
            if '</content>' in token:
                self.in_code_block = False

            # Count lines
            if '\n' in token:
                lines_in_token = token.count('\n')
                self.solution_lines_printed += lines_in_token
                if self.in_code_block:
                    self.code_block_lines += lines_in_token

            # Truncate long code blocks
            if self.in_code_block and self.code_block_lines > 40:
                if self.code_block_lines == 41:
                    ui.console.print("\n[dim]  ⋯ (long code truncated, full file saved)[/dim]", style="#555555")
                    ui.console.file.flush()
                return

            # Truncate very long output
            if self.solution_lines_printed > self.max_solution_lines:
                if self.solution_lines_printed == self.max_solution_lines + 1:
                    ui.console.print("\n[dim]  ⋯ (output truncated)[/dim]", style="#555555")
                    ui.console.file.flush()
                return

            # Print solution tokens in white
            ui.console.print(token, end='', style="white")
            ui.console.file.flush()
        else:
            # Thinking mode — show ALL content in dim style
            self.buffer += token

            ui.console.print(token, end='', style="#666666")
            ui.console.file.flush()

    def print_action(self, action: str):
        """Print an action label with animated dot."""
        self._stop_live_and_spinner()
        
        elapsed = time.time() - self.start_time
        # Breathing dot color
        phase = (math.sin(elapsed * 2.0) + 1.0) / 2.0
        r = int(180 + 75 * phase)
        g = int(100 + 55 * phase)
        b = int(60 + 40 * phase)
        dot_color = f"#{r:02x}{g:02x}{b:02x}"
        
        ui.console.print(f"  [{dot_color}]●[/] [{dot_color}]{action}[/]")

    def start_waiting(self, message: str = "Thinking"):
        """Start smooth waiting animation.

        Args:
            message: Message to show during wait
        """
        self._stop_live_and_spinner()
        self.waiting_spinner = WaitingSpinner(message, delay=0.08, color="#D77757")
        self.waiting_spinner.start()

    def update_waiting(self, message: str):
        """Update the message of the currently running waiting spinner."""
        if self.waiting_spinner:
            self.waiting_spinner.text = message
        else:
            self.start_waiting(message)

    def stop_waiting(self):
        """Stop waiting animation."""
        self._stop_live_and_spinner()

    def update_progress(self, current: int, total: int, message: str = ""):
        """Update progress indicator.

        Args:
            current: Current step
            total: Total steps
            message: Progress message
        """
        self._stop_live_and_spinner()

        self.progress_current = current
        self.progress_total = total
        self.progress_message = message

        if self.mode == "solution":
            return  # Don't show progress in solution mode

        # Show progress bar
        percent = int((current / total) * 100) if total > 0 else 0
        bar_length = 20
        filled = int((current / total) * bar_length) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        elapsed = time.time() - self.start_time
        phase = (math.sin(elapsed * 2.0) + 1.0) / 2.0
        r = int(180 + 75 * phase)
        g = int(100 + 55 * phase)
        b = int(60 + 40 * phase)
        color = f"#{r:02x}{g:02x}{b:02x}"

        ui.console.print(
            f"  [{color}]▸[/] [#999999]{message} [{bar}] {percent}% ({current}/{total})[/]"
        )
        ui.console.file.flush()

    def update_progress(self, current: int, total: int, message: str = ""):
        """Update progress indicator.

        Args:
            current: Current step
            total: Total steps
            message: Progress message
        """
        self.progress_current = current
        self.progress_total = total
        self.progress_message = message

        if self.mode == "solution":
            return  # Don't show progress in solution mode

        self._stop_live_and_spinner()

        # Animated progress bar
        percent = int((current / total) * 100) if total > 0 else 0
        bar_length = 24
        filled = int((current / total) * bar_length) if total > 0 else 0
        
        # Use gradient characters for the bar
        bar = "█" * filled + "░" * (bar_length - filled)
        
        elapsed = time.time() - self.start_time
        phase = (math.sin(elapsed * 2.0) + 1.0) / 2.0
        r = int(180 + 75 * phase)
        g = int(100 + 55 * phase)
        b = int(60 + 40 * phase)
        bar_color = f"#{r:02x}{g:02x}{b:02x}"

        ui.console.print(
            f"  [{bar_color}]▸[/] [#888888]{message} [{bar_color}]{bar}[/{bar_color}] {percent}% ({current}/{total})[/]"
        )

    def _update(self):
        """Update the live display."""
        pass

    @contextmanager
    def show(self):
        """Context manager for live display."""
        self.buffer = ""
        self.mode = "thinking"
        self.solution_lines_printed = 0
        self.in_code_block = False
        self.code_block_lines = 0
        self.start_time = time.time()
        self.frame_idx = 0

        try:
            yield self
        finally:
            self._stop_live_and_spinner()


# Global thinking display instance
_thinking = ThinkingDisplay()


def get_thinking_display() -> ThinkingDisplay:
    """Get the global thinking display."""
    return _thinking


@contextmanager
def show_thinking():
    """Show thinking display for a block of code."""
    with _thinking.show():
        yield _thinking
