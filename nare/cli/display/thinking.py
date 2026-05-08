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

_TOOL_OPEN_TAGS = (
    '<read_file>', '<edit_file>', '<write_file>',
    '<bash_command>', '<tool_call>', '<create_file>',
    '<list_files>',
)

_TOOL_CLOSE_TAGS = (
    '</read_file>', '</edit_file>', '</write_file>',
    '</bash_command>', '</tool_call>', '</create_file>',
    '</list_files>',
)


def _strip_xml_tags(text: str) -> str:
    cleaned = re.sub(r'</?(?:solution|reasoning|abstract_signature|delta_reasoning|tool_call)\s*>', '', text)
    cleaned = re.sub(r'\{\s*"name"\s*:\s*"\w+"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\}', '', cleaned)
    return cleaned.strip()


class ThinkingDisplay:

    def __init__(self, session=None):
        self.live = None
        self.buffer = ""
        self.mode = "thinking"
        self.solution_lines_printed = 0
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
        self.thinking_token_count = 0
        self.solution_started = False
        self.route_shown = False
        self.current_route = None
        self.session = session

    def _stop_live_and_spinner(self):
        if self.waiting_spinner:
            self.waiting_spinner.stop()
            self.waiting_spinner = None

    def _elapsed_str(self) -> str:
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        return f"{int(elapsed // 60)}m{int(elapsed % 60)}s"

    def _print_transition(self):
        if self.thinking_token_count > 0:
            elapsed = self._elapsed_str()
            ui.console.print()
            ui.console.print(
                f"  [#444444]\u2500[/] [#555555]thought for {elapsed}[/]",
            )
            ui.console.print()

    def show_route(self, route: str):
        from .blocks import ROUTE_PALETTE
        self.current_route = route
        color = ROUTE_PALETTE.get(route, "#999999")
        self._stop_live_and_spinner()
        ui.console.print(
            Text.assemble(
                ("  ", ""),
                ("\u25c6 ", f"bold {color}"),
                (route, f"bold {color}"),
            )
        )
        self.route_shown = True

    def switch_to_solution(self):
        self._stop_live_and_spinner()
        if self.mode == "thinking":
            self._print_transition()
        self.mode = "solution"
        self.solution_lines_printed = 0
        self.in_code_block = False
        self.code_block_lines = 0
        self.solution_started = True

    def stream_token(self, token: str):
        self._stop_live_and_spinner()

        stripped = token.strip()
        if stripped in _XML_TAGS or any(stripped == tag for tag in _XML_TAGS):
            return
        if re.search(r'^</?(?:solution|reasoning|abstract_signature|delta_reasoning|tool_call)\s*>$', stripped):
            return

        if self.mode == "thinking":
            self.thinking_lines.append(token)
            self.thinking_token_count += 1
            ui.console.print(token, style="#555555", end="")
            return

        if self.mode == "solution":
            if any(tag in token for tag in _TOOL_OPEN_TAGS):
                self.in_xml_tag = True
                self.xml_buffer = token
                return

            if self.in_xml_tag:
                self.xml_buffer += token
                if any(tag in self.xml_buffer for tag in _TOOL_CLOSE_TAGS):
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
                    ui.console.print("\n[dim]  ... (truncated)[/dim]", style="#555555")
                    ui.console.file.flush()
                return

            if self.solution_lines_printed > 200:
                if self.solution_lines_printed == 201:
                    ui.console.print("\n[dim]  ... (truncated)[/dim]", style="#555555")
                    ui.console.file.flush()
                return

            ui.console.print(token, end='', style="white")
            ui.console.file.flush()
        else:
            self.buffer += token
            if '\n' in token:
                lines = self.buffer.split('\n')
                self.thinking_lines.extend(lines[:-1])
                self.buffer = lines[-1]
            ui.console.print(token, end='', style="#555555")
            ui.console.file.flush()

    def print_action(self, action: str):
        self._stop_live_and_spinner()
        ui.console.print(f"  [#D77757]>[/] [#999999]{action}[/]")

    def start_waiting(self, message: str = "Thinking"):
        self._stop_live_and_spinner()
        self.waiting_spinner = WaitingSpinner(message, delay=0.08, color="#D77757")
        self.waiting_spinner.start()

    def update_waiting(self, message: str):
        if self.waiting_spinner:
            self.waiting_spinner.text = message
        else:
            self.start_waiting(message)

    def stop_waiting(self):
        self._stop_live_and_spinner()

    def update_progress(self, current: int, total: int, message: str = ""):
        self.progress_current = current
        self.progress_total = total
        self.progress_message = message

        if self.mode == "solution":
            return

        self._stop_live_and_spinner()

        percent = int((current / total) * 100) if total > 0 else 0
        bar_length = 24
        filled = int((current / total) * bar_length) if total > 0 else 0
        bar = "=" * filled + "-" * (bar_length - filled)

        elapsed = time.time() - self.start_time
        phase = (math.sin(elapsed * 2.0) + 1.0) / 2.0
        r = int(180 + 75 * phase)
        g = int(100 + 55 * phase)
        b = int(60 + 40 * phase)
        bar_color = f"#{r:02x}{g:02x}{b:02x}"

        ui.console.print(
            f"  [{bar_color}]>[/] [#888888]{message} [{bar_color}][{bar}][/{bar_color}] {percent}%[/]"
        )

    def _update(self):
        pass

    @contextmanager
    def show(self):
        self.buffer = ""
        self.mode = "thinking"
        self.solution_lines_printed = 0
        self.in_code_block = False
        self.code_block_lines = 0
        self.start_time = time.time()
        self.frame_idx = 0
        self.thinking_lines = []
        self.thinking_token_count = 0
        self.solution_started = False
        self.route_shown = False

        try:
            yield self
        finally:
            self._stop_live_and_spinner()
            if not self.solution_started and self.thinking_token_count > 0:
                ui.console.print()


_thinking = ThinkingDisplay()


def get_thinking_display() -> ThinkingDisplay:
    return _thinking


@contextmanager
def show_thinking(session=None):
    display = ThinkingDisplay(session=session)
    with display.show():
        yield display
