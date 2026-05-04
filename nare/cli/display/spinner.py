"""
Premium animations for NARE CLI.

Features:
- Breathing pulse animation with smooth color gradients
- Neural-network inspired thinking animation
- Braille-based wave spinner
- All animations are thread-safe and flicker-free via Rich Live
"""

import sys
import time
import math
import threading
from rich.console import Console
from rich.live import Live
from rich.text import Text
from . import ui

WAVE_FRAMES = [
    "⠁⠂⠄⡀⢀⠠⠐⠈",
    "⠂⠄⡀⢀⠠⠐⠈⠁",
    "⠄⡀⢀⠠⠐⠈⠁⠂",
    "⡀⢀⠠⠐⠈⠁⠂⠄",
    "⢀⠠⠐⠈⠁⠂⠄⡀",
    "⠠⠐⠈⠁⠂⠄⡀⢀",
    "⠐⠈⠁⠂⠄⡀⢀⠠",
    "⠈⠁⠂⠄⡀⢀⠠⠐",
]

DOTS_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

BREATH_CHARS = " ░▒▓█▓▒░"

def _breathing_color(t: float) -> str:
    """Generate a smoothly pulsing orange color.
    
    Uses sine wave to create a breathing effect between
    dim amber and bright orange.
    """
    phase = (math.sin(t * 2.0) + 1.0) / 2.0

    r = int(139 + (255 - 139) * phase)
    g = int(69 + (153 - 69) * phase)
    b = int(19 + (102 - 19) * phase)
    return f"#{r:02x}{g:02x}{b:02x}"

def _wave_color(t: float, offset: float = 0.0) -> str:
    """Generate rainbow-shifted warm color for wave effect."""
    phase = (math.sin(t * 1.5 + offset) + 1.0) / 2.0

    r = int(200 + 55 * phase)
    g = int(100 + 80 * phase)
    b = int(60 + 40 * phase)
    return f"#{r:02x}{g:02x}{b:02x}"

class WaitingSpinner:
    """Premium thread-based spinner with breathing animation.

    Creates a beautiful, non-blocking waiting indicator that feels
    alive and responsive. Uses braille wave + breathing color pulse.

    Usage:
        spinner = WaitingSpinner("Thinking")
        spinner.start()
        # ... do work ...
        spinner.stop()

    Or as context manager:
        with WaitingSpinner("Processing"):
            # ... do work ...
    """

    def __init__(self, text: str = "Thinking", delay: float = 0.08, color: str = "#D77757"):
        """Initialize waiting spinner.

        Args:
            text: Message to display
            delay: Frame delay in seconds (lower = smoother)
            color: Base color (overridden by breathing animation)
        """
        self.text = text
        self.delay = delay
        self.base_color = color
        self._stop_event = threading.Event()
        self._thread = None
        self.live = None
        self._start_time = 0.0

    def _render_frame(self) -> Text:
        """Render a single animation frame."""
        elapsed = time.time() - self._start_time
        result = Text()

        dot_idx = int(elapsed / self.delay) % len(DOTS_FRAMES)
        dot_color = _breathing_color(elapsed)
        result.append(f"  {DOTS_FRAMES[dot_idx]} ", style=dot_color)

        msg_color = _breathing_color(elapsed + 0.5)
        result.append(self.text, style=msg_color)

        result.append("  ", style="default")
        wave_idx = int(elapsed / self.delay) % len(WAVE_FRAMES)
        wave = WAVE_FRAMES[wave_idx]
        for i, ch in enumerate(wave):
            ch_color = _wave_color(elapsed, offset=i * 0.4)
            result.append(ch, style=ch_color)

        if elapsed > 2.0:
            result.append(f"  {elapsed:.0f}s", style="#555555")

        return result

    def _spin(self):
        """Thread target — render frames until stopped."""
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
        """Start spinner in background thread."""
        if self._thread is None or not self._thread.is_alive():
            self._start_time = time.time()
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
            time.sleep(0.05)

    def stop(self):
        """Stop spinner and clear the line."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=1.0)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
