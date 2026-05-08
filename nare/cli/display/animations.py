import math


def interpolate_color(color1: tuple, color2: tuple, t: float) -> tuple:
    r = int(color1[0] + (color2[0] - color1[0]) * t)
    g = int(color1[1] + (color2[1] - color1[1]) * t)
    b = int(color1[2] + (color2[2] - color1[2]) * t)
    return (r, g, b)


def pulse_opacity(elapsed: float, frequency: float = 2.0) -> float:
    pulse = math.sin(elapsed * frequency * 2 * math.pi)
    return 0.65 + (pulse * 0.35)


def get_shimmer_color(elapsed: float, speed: float = 2.0) -> str:
    base_r, base_g, base_b = 0xD7, 0x77, 0x57
    bright_r, bright_g, bright_b = 0xFF, 0x99, 0x77
    t = (math.sin(elapsed * speed) + 1) / 2
    r = int(base_r + (bright_r - base_r) * t)
    g = int(base_g + (bright_g - base_g) * t)
    b = int(base_b + (bright_b - base_b) * t)
    return f"#{r:02x}{g:02x}{b:02x}"
