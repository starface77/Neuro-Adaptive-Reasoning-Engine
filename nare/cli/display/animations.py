"""Shimmer animation utilities for text.

Creates a wave effect that moves through text characters, highlighting
them with a shimmer color as the wave passes.
"""

import time
import math
from typing import Optional


class ShimmerAnimation:
    """Manages shimmer animation state for text."""

    def __init__(self, text: str, speed: float = 0.1):
        self.text = text
        self.speed = speed  # Speed of shimmer wave
        self.start_time = time.time()
        self.text_length = len(text)

    def get_shimmer_index(self) -> int:
        """Get current shimmer position based on elapsed time."""
        elapsed = time.time() - self.start_time
        # Wave moves from left to right
        position = (elapsed * self.speed * 10) % (self.text_length + 10)
        return int(position) - 5  # Start offscreen

    def get_char_style(self, index: int) -> str:
        """Get style for character at index based on shimmer position."""
        shimmer_pos = self.get_shimmer_index()

        # Characters within ±1 of shimmer position get highlighted
        distance = abs(index - shimmer_pos)

        if distance == 0:
            return "accent_shimmer"  # Brightest
        elif distance == 1:
            return "accent"  # Medium
        else:
            return "text_muted"  # Normal


def interpolate_color(color1: tuple, color2: tuple, t: float) -> tuple:
    """Interpolate between two RGB colors.

    Args:
        color1: (r, g, b) tuple for start color
        color2: (r, g, b) tuple for end color
        t: interpolation factor (0.0 to 1.0)

    Returns:
        Interpolated (r, g, b) tuple
    """
    r = int(color1[0] + (color2[0] - color1[0]) * t)
    g = int(color1[1] + (color2[1] - color1[1]) * t)
    b = int(color1[2] + (color2[2] - color1[2]) * t)
    return (r, g, b)


def pulse_opacity(elapsed: float, frequency: float = 2.0) -> float:
    """Calculate pulsing opacity based on elapsed time.

    Args:
        elapsed: Time elapsed in seconds
        frequency: Pulses per second

    Returns:
        Opacity value between 0.3 and 1.0
    """
    # Sine wave for smooth pulsing
    pulse = math.sin(elapsed * frequency * 2 * math.pi)
    # Map from [-1, 1] to [0.3, 1.0]
    return 0.65 + (pulse * 0.35)


def get_shimmer_color(elapsed: float, speed: float = 2.0) -> str:
    """Get shimmer color based on elapsed time.

    Creates a smooth color transition effect for text.

    Args:
        elapsed: Time elapsed in seconds
        speed: Speed of color transition

    Returns:
        Hex color string (e.g., "#D77757")
    """
    # Base orange accent for the NARE palette
    base_r, base_g, base_b = 0xD7, 0x77, 0x57

    # Shimmer between base and brighter version
    bright_r, bright_g, bright_b = 0xFF, 0x99, 0x77

    # Sine wave for smooth transition
    t = (math.sin(elapsed * speed) + 1) / 2  # Map to [0, 1]

    # Interpolate
    r = int(base_r + (bright_r - base_r) * t)
    g = int(base_g + (bright_g - base_g) * t)
    b = int(base_b + (bright_b - base_b) * t)

    return f"#{r:02x}{g:02x}{b:02x}"
