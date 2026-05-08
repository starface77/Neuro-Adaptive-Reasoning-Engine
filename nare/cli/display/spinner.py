import time
import math
import threading
from rich.live import Live
from rich.text import Text
from . import ui

DOTS_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

def _pulse_color(t: float) -> str:
    phase = (math.sin(t * 2.0) + 1.0) / 2.0
    r = int(160 + 55 * phase)
    g = int(90 + 29 * phase)
    b = int(60 + 27 * phase)
    return f"#{r:02x}{g:02x}{b:02x}"

class WaitingSpinner:

    def __init__(self, text: str = "Thinking", delay: float = 0.08, color: str = "#D77757"):
        self.text = text
        self.delay = delay
        self.base_color = color
        self._stop_event = threading.Event()
        self._thread = None
        self.live = None
        self._start_time = 0.0

    def _render_frame(self) -> Text:
        elapsed = time.time() - self._start_time
        result = Text()

        dot_idx = int(elapsed / self.delay) % len(DOTS_FRAMES)
        color = _pulse_color(elapsed)
        result.append(f"  {DOTS_FRAMES[dot_idx]} ", style=color)
        result.append(self.text, style="#999999")

        if elapsed > 2.0:
            result.append(f"  {elapsed:.0f}s", style="#555555")

        return result

    def _spin(self):
        try:
            self.live = Live(
                self._render_frame(),
                console=ui.console,
                refresh_per_second=12,
                transient=True,
            )
            with self.live:
                while not self._stop_event.is_set():
                    self.live.update(self._render_frame())
                    time.sleep(self.delay)
        except Exception:
            pass

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._start_time = time.time()
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
            time.sleep(0.05)

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
