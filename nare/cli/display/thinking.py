"""
Real-time thinking display — premium streaming UI.

Shows LLM reasoning in gray, solution in white, with beautiful
animated transitions between phases. Thread-safe and flicker-free.
"""

import re
import time
import math
from contextlib import contextmanager
from rich.text import Text
from . import ui
from .spinner import WaitingSpinner

_XML_TAGS = (
    '<solution>', '</solution>', '<reasoning>', '</reasoning>',
    '<abstract_signature>', '</abstract_signature>',
    '<delta_reasoning>', '</delta_reasoning>',
    '<tool_call>', '</tool_call>',
)


def _strip_xml_tags(text: str) -> str:
    """Remove all internal XML tags from text."""
    cleaned = re.sub(r'</?(?:solution|reasoning|abstract_signature|delta_reasoning|tool_call)\s*>', '', text)
    cleaned = re.sub(r'\{\s*"name"\s*:\s*"\w+"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\}', '', cleaned)
    return cleaned.strip()

class ThinkingDisplay:
    """Manages real-time display of LLM reasoning and solution tokens.
    
    Phases:
    1. THINKING — gray text, reasoning tokens stream
    2. SOLUTION — white text, final answer streams
    
    Transitions between phases are animated with color shifts.
    """

    def __init__(self, session=None):
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
        self.in_xml_tag = False
        self.xml_buffer = ""
        self.thinking_lines = []
        self.keep_last_n_lines = 15
        self.session = session

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
        """Stream a single token from LLM output.

        Modes:
        - thinking: Gray text, reasoning process
        - solution: White text, final answer
        """

        self._stop_live_and_spinner()

        stripped = token.strip()
        if stripped in _XML_TAGS or any(stripped == tag for tag in _XML_TAGS):
            return
        if re.search(r'^</?(?:solution|reasoning|abstract_signature|delta_reasoning|tool_call)\s*>$', stripped):
            return

        if self.mode == "thinking":
            self.thinking_lines.append(token)
            ui.console.print(token, style="#666666", end="")
            return

        if self.mode == "solution":
            if any(tag in token for tag in ('<read_file>', '<edit_file>', '<write_file>', '<bash_command>', '<tool_call>', '<create_file>')):
                self.in_xml_tag = True
                self.xml_buffer = token
                return

            if self.in_xml_tag:
                self.xml_buffer += token

                close_tags = ('</read_file>', '</edit_file>', '</write_file>', '</bash_command>', '</tool_call>', '</create_file>')
                if any(tag in self.xml_buffer for tag in close_tags):
                    self.in_xml_tag = False
                    self.xml_buffer = ""
                return

            if '```' in token or '<content>' in token:
                self.in_code_block = True
                self.code_block_lines = 0
            if '```' in token and self.in_code_block and self.code_block_lines > 0:
                self.in_code_block = False
            if '</content>' in token:
                self.in_code_block = False

            if '\n' in token:
                lines_in_token = token.count('\n')
                self.solution_lines_printed += lines_in_token
                if self.in_code_block:
                    self.code_block_lines += lines_in_token

            if self.in_code_block and self.code_block_lines > 200:
                if self.code_block_lines == 201:
                    ui.console.print("\n[dim]  ⋯ (long code truncated)[/dim]", style="#555555")
                    ui.console.file.flush()
                return

            if self.solution_lines_printed > 200:
                if self.solution_lines_printed == 201:
                    ui.console.print("\n[dim]  ⋯ (output truncated)[/dim]", style="#555555")
                    ui.console.file.flush()
                return

            ui.console.print(token, end='', style="white")
            ui.console.file.flush()
        else:

            self.buffer += token

            if '\n' in token:
                lines = (self.buffer).split('\n')
                self.thinking_lines.extend(lines[:-1])
                self.buffer = lines[-1]

            ui.console.print(token, end='', style="#666666")
            ui.console.file.flush()

    def print_action(self, action: str):
        """Print an action label with animated dot."""
        self._stop_live_and_spinner()

        elapsed = time.time() - self.start_time

        phase = (math.sin(elapsed * 2.0) + 1.0) / 2.0
        r = int(180 + 75 * phase)
        g = int(100 + 55 * phase)
        b = int(60 + 40 * phase)
        dot_color = f"#{r:02x}{g:02x}{b:02x}"

        ui.console.print(f"  [{dot_color}]●[/] [{dot_color}]{action}[/]")

    def start_waiting(self, message: str = "Thinking"):
        """Start smooth waiting animation.

        Shows spinner with message during long operations (routing, model loading).
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
            return

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
            return

        self._stop_live_and_spinner()

        percent = int((current / total) * 100) if total > 0 else 0
        bar_length = 24
        filled = int((current / total) * bar_length) if total > 0 else 0

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
        self.thinking_lines = []

        try:
            yield self
        finally:
            self._stop_live_and_spinner()

            # Removed "Key reasoning steps" summary - reasoning should not leak to user
            # User only sees solution output, not internal thinking process

_thinking = ThinkingDisplay()

def get_thinking_display() -> ThinkingDisplay:
    """Get the global thinking display."""
    return _thinking

@contextmanager
def show_thinking(session=None):
    """Show thinking display for a block of code."""
    display = ThinkingDisplay(session=session)
    with display.show():
        yield display
