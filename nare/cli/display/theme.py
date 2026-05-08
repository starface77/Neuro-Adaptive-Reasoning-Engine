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
    accent_shimmer="rgb(235,159,127)",

    text="rgb(255,255,255)",
    text_inverse="rgb(0,0,0)",
    text_muted="rgb(140,140,140)",
    text_subtle="rgb(90,90,90)",

    border="rgb(60,60,60)",
    border_shimmer="rgb(80,80,80)",
    background="rgb(0,0,0)",

    success="rgb(78,186,101)",
    error="rgb(215,87,87)",
    warning="rgb(255,193,7)",
    info="rgb(85,153,255)",

    route_fast="rgb(78,186,101)",
    route_compiled_skill="rgb(78,200,120)",
    route_reflex="rgb(177,185,249)",
    route_hybrid="rgb(255,193,7)",
    route_slow="rgb(215,119,87)",

    agent_triage="rgb(177,185,249)",
    agent_planning="rgb(255,193,7)",
    agent_coding="rgb(215,119,87)",

    diff_added="rgb(35,60,35)",
    diff_removed="rgb(60,35,35)",
    diff_added_word="rgb(78,186,101)",
    diff_removed_word="rgb(215,87,87)",
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
