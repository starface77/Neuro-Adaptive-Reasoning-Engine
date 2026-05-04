"""Theme system for the NARE CLI.

Provides RGB-based themes with proper color definitions,
avoiding dependency on terminal ANSI color customizations.
"""

from dataclasses import dataclass
from typing import Literal

ThemeName = Literal["dark", "light", "dark-ansi", "light-ansi"]


@dataclass
class Theme:
    """Theme color palette."""

    # Primary colors
    accent: str              # Main accent (orange for NARA)
    accent_shimmer: str      # Lighter accent for animations

    # Text colors
    text: str                # Primary text
    text_inverse: str        # Inverse text (for highlights)
    text_muted: str          # Muted/secondary text
    text_subtle: str         # Very subtle text

    # UI colors
    border: str              # Border color
    border_shimmer: str      # Animated border
    background: str          # Background color

    # Semantic colors
    success: str             # Success/green
    error: str               # Error/red
    warning: str             # Warning/yellow
    info: str                # Info/blue

    # Route colors (for NARE routing display)
    route_fast: str          # FAST path
    route_reflex: str        # REFLEX path
    route_hybrid: str        # HYBRID path
    route_slow: str          # SLOW path

    # Agent colors
    agent_triage: str        # Triage agent
    agent_planning: str      # Planning agent
    agent_coding: str        # Coding agent

    # Diff colors
    diff_added: str          # Added lines
    diff_removed: str        # Removed lines
    diff_added_word: str     # Added word highlight
    diff_removed_word: str   # Removed word highlight


# Dark theme (default)
DARK_THEME = Theme(
    # Primary
    accent="rgb(215,119,87)",           # NARA orange
    accent_shimmer="rgb(235,159,127)",

    # Text
    text="rgb(255,255,255)",
    text_inverse="rgb(0,0,0)",
    text_muted="rgb(153,153,153)",
    text_subtle="rgb(80,80,80)",

    # UI
    border="rgb(136,136,136)",
    border_shimmer="rgb(166,166,166)",
    background="rgb(0,0,0)",

    # Semantic
    success="rgb(78,186,101)",
    error="rgb(255,107,128)",
    warning="rgb(255,193,7)",
    info="rgb(177,185,249)",

    # Routes
    route_fast="rgb(78,186,101)",       # Green - cached
    route_reflex="rgb(177,185,249)",    # Blue - skills
    route_hybrid="rgb(255,193,7)",      # Yellow - delta
    route_slow="rgb(215,119,87)",       # Orange - synthesis

    # Agents
    agent_triage="rgb(177,185,249)",    # Blue
    agent_planning="rgb(255,193,7)",    # Yellow
    agent_coding="rgb(215,119,87)",     # Orange

    # Diff
    diff_added="rgb(34,92,43)",
    diff_removed="rgb(122,41,54)",
    diff_added_word="rgb(56,166,96)",
    diff_removed_word="rgb(179,89,107)",
)


# Light theme
LIGHT_THEME = Theme(
    # Primary
    accent="rgb(215,119,87)",
    accent_shimmer="rgb(245,149,117)",

    # Text
    text="rgb(0,0,0)",
    text_inverse="rgb(255,255,255)",
    text_muted="rgb(102,102,102)",
    text_subtle="rgb(175,175,175)",

    # UI
    border="rgb(153,153,153)",
    border_shimmer="rgb(183,183,183)",
    background="rgb(255,255,255)",

    # Semantic
    success="rgb(44,122,57)",
    error="rgb(171,43,63)",
    warning="rgb(150,108,30)",
    info="rgb(87,105,247)",

    # Routes
    route_fast="rgb(44,122,57)",
    route_reflex="rgb(87,105,247)",
    route_hybrid="rgb(150,108,30)",
    route_slow="rgb(215,119,87)",

    # Agents
    agent_triage="rgb(87,105,247)",
    agent_planning="rgb(150,108,30)",
    agent_coding="rgb(215,119,87)",

    # Diff
    diff_added="rgb(105,219,124)",
    diff_removed="rgb(255,168,180)",
    diff_added_word="rgb(47,157,68)",
    diff_removed_word="rgb(209,69,75)",
)


# Dark ANSI theme (16-color terminals)
DARK_ANSI_THEME = Theme(
    accent="ansi:redBright",
    accent_shimmer="ansi:yellowBright",
    text="ansi:whiteBright",
    text_inverse="ansi:black",
    text_muted="ansi:white",
    text_subtle="ansi:blackBright",
    border="ansi:white",
    border_shimmer="ansi:whiteBright",
    background="ansi:black",
    success="ansi:greenBright",
    error="ansi:redBright",
    warning="ansi:yellowBright",
    info="ansi:blueBright",
    route_fast="ansi:greenBright",
    route_reflex="ansi:blueBright",
    route_hybrid="ansi:yellowBright",
    route_slow="ansi:redBright",
    agent_triage="ansi:blueBright",
    agent_planning="ansi:yellowBright",
    agent_coding="ansi:redBright",
    diff_added="ansi:green",
    diff_removed="ansi:red",
    diff_added_word="ansi:greenBright",
    diff_removed_word="ansi:redBright",
)


# Light ANSI theme
LIGHT_ANSI_THEME = Theme(
    accent="ansi:redBright",
    accent_shimmer="ansi:yellowBright",
    text="ansi:black",
    text_inverse="ansi:white",
    text_muted="ansi:blackBright",
    text_subtle="ansi:white",
    border="ansi:white",
    border_shimmer="ansi:whiteBright",
    background="ansi:white",
    success="ansi:green",
    error="ansi:red",
    warning="ansi:yellow",
    info="ansi:blue",
    route_fast="ansi:green",
    route_reflex="ansi:blue",
    route_hybrid="ansi:yellow",
    route_slow="ansi:redBright",
    agent_triage="ansi:blue",
    agent_planning="ansi:yellow",
    agent_coding="ansi:redBright",
    diff_added="ansi:green",
    diff_removed="ansi:red",
    diff_added_word="ansi:greenBright",
    diff_removed_word="ansi:redBright",
)


THEMES = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
    "dark-ansi": DARK_ANSI_THEME,
    "light-ansi": LIGHT_ANSI_THEME,
}


def get_theme(name: ThemeName = "dark") -> Theme:
    """Get theme by name."""
    return THEMES.get(name, DARK_THEME)


def rgb_to_ansi(color: str) -> str:
    """Convert rgb(r,g,b) to ANSI escape code.

    Examples:
        rgb(215,119,87) -> \\x1b[38;2;215;119;87m
        ansi:redBright -> \\x1b[91m
    """
    if color.startswith("rgb("):
        # Extract RGB values
        rgb = color[4:-1].split(",")
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        return f"\x1b[38;2;{r};{g};{b}m"

    elif color.startswith("ansi:"):
        # Map ANSI color names to codes
        ansi_map = {
            "black": "30", "red": "31", "green": "32", "yellow": "33",
            "blue": "34", "magenta": "35", "cyan": "36", "white": "37",
            "blackBright": "90", "redBright": "91", "greenBright": "92",
            "yellowBright": "93", "blueBright": "94", "magentaBright": "95",
            "cyanBright": "96", "whiteBright": "97",
        }
        name = color[5:]
        code = ansi_map.get(name, "37")
        return f"\x1b[{code}m"

    return "\x1b[0m"  # Reset


def colorize(text: str, color: str) -> str:
    """Colorize text with theme color.

    Args:
        text: Text to colorize
        color: Color from theme (rgb(...) or ansi:...)

    Returns:
        ANSI-colored text with reset at end
    """
    return f"{rgb_to_ansi(color)}{text}\x1b[0m"
