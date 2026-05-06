"""
Provides RGB-based themes with proper color definitions,
avoiding dependency on terminal ANSI color customizations.
"""

from dataclasses import dataclass
from typing import Literal

ThemeName = Literal["dark", "light", "dark-ansi", "light-ansi"]

@dataclass
class Theme:

    accent: str
    accent_shimmer: str

    text: str
    text_inverse: str
    text_muted: str
    text_subtle: str

    border: str
    border_shimmer: str
    background: str

    success: str
    error: str
    warning: str
    info: str

    route_fast: str
    route_compiled_skill: str
    route_reflex: str
    route_hybrid: str
    route_slow: str

    agent_triage: str
    agent_planning: str
    agent_coding: str

    diff_added: str
    diff_removed: str
    diff_added_word: str
    diff_removed_word: str

DARK_THEME = Theme(

    accent="rgb(215,119,87)",
    accent_shimmer="rgb(245,149,117)",

    text="rgb(255,255,255)",
    text_inverse="rgb(0,0,0)",
    text_muted="rgb(160,160,160)",
    text_subtle="rgb(100,100,100)",

    border="rgb(70,70,70)",
    border_shimmer="rgb(90,90,90)",
    background="rgb(0,0,0)",

    success="rgb(100,220,150)",
    error="rgb(255,100,120)",
    warning="rgb(255,200,100)",
    info="rgb(120,180,255)",

    route_fast="rgb(100,220,150)",
    route_compiled_skill="rgb(150,255,100)",
    route_reflex="rgb(120,180,255)",
    route_hybrid="rgb(255,200,100)",
    route_slow="rgb(215,119,87)",

    agent_triage="rgb(120,180,255)",
    agent_planning="rgb(255,200,100)",
    agent_coding="rgb(215,119,87)",

    diff_added="rgb(50,50,50)",
    diff_removed="rgb(40,40,40)",
    diff_added_word="rgb(80,80,80)",
    diff_removed_word="rgb(70,70,70)",
)

LIGHT_THEME = Theme(

    accent="rgb(215,119,87)",
    accent_shimmer="rgb(245,149,117)",

    text="rgb(0,0,0)",
    text_inverse="rgb(255,255,255)",
    text_muted="rgb(102,102,102)",
    text_subtle="rgb(175,175,175)",

    border="rgb(153,153,153)",
    border_shimmer="rgb(183,183,183)",
    background="rgb(255,255,255)",

    success="rgb(44,122,57)",
    error="rgb(171,43,63)",
    warning="rgb(150,108,30)",
    info="rgb(87,105,247)",

    route_fast="rgb(44,122,57)",
    route_compiled_skill="rgb(50,180,50)",
    route_reflex="rgb(87,105,247)",
    route_hybrid="rgb(150,108,30)",
    route_slow="rgb(215,119,87)",

    agent_triage="rgb(87,105,247)",
    agent_planning="rgb(150,108,30)",
    agent_coding="rgb(215,119,87)",

    diff_added="rgb(105,219,124)",
    diff_removed="rgb(255,168,180)",
    diff_added_word="rgb(47,157,68)",
    diff_removed_word="rgb(209,69,75)",
)

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
    route_compiled_skill="ansi:cyanBright",
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
    route_compiled_skill="ansi:cyan",
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
    return THEMES.get(name, DARK_THEME)

def rgb_to_ansi(color: str) -> str:
    """Convert RGB or ANSI-named color to escape sequence.

    Examples:
        rgb(215,119,87) -> \x1b[38;2;215;119;87m
        ansi:redBright -> \x1b[91m
    """
    if color.startswith("rgb("):

        rgb = color[4:-1].split(",")
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        return f"\x1b[38;2;{r};{g};{b}m"

    elif color.startswith("ansi:"):

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

    return "\x1b[0m"

def colorize(text: str, color: str) -> str:
    """Wrap text in ANSI color sequences.

    Args:
        text: Text to colorize
        color: Color from theme (rgb(...) or ansi:...)

    Returns:
        ANSI-colored text with reset at end
    """
    return f"{rgb_to_ansi(color)}{text}\x1b[0m"

RICH_STYLE_MAP = {
    "info": "bold cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "success": "bold green",

    "agent.name": "bold white",
    "agent.thinking": "italic bright_black",
    "agent.action": "bold blue",
    "agent.result": "white",

    "file.read": "cyan",
    "file.write": "green",
    "file.edit": "yellow",
    "file.delete": "red",

    "code.keyword": "bold magenta",
    "code.string": "green",
    "code.number": "cyan",
    "code.comment": "bright_black",

    "border": "bright_black",
    "title": "bold white",
}
