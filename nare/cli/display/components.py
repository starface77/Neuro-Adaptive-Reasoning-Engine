from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text
from rich.console import Console
from typing import Optional

from . import blocks

class StatusBar:

    ROUTE_STYLES = {
        "FAST":     ("#4EBA65", "✦"),
        "REFLEX":   ("#B1B9F9", "↯"),
        "HYBRID":   ("#FFC107", "◈"),
        "SLOW":     ("#D77757", "✻"),
        "DIRECT":   ("#888888", "→"),
        "COMPILED_SKILL": ("#4EBA65", "★"),
        "AGENT":    ("#D77757", "◆"),
    }

    ROUTE_COLORS = {
        "FAST": "#4EBA65",
        "REFLEX": "#B1B9F9",
        "HYBRID": "#FFC107",
        "SLOW": "#D77757",
        "DIRECT": "#888888",
        "COMPILED_SKILL": "#4EBA65",
    }

    @staticmethod
    def render(
        console: Console,
        route: str,
        elapsed: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        episodes: int = 0,
        skills: int = 0,
        *,
        mode: Optional[str] = None,
        model: Optional[str] = None,
        repo: Optional[str] = None,
    ) -> None:
        total_tokens = (tokens_in or 0) + (tokens_out or 0)
        blocks.render_status_line(
            console,
            mode=mode,
            model=model,
            repo=repo,
            route=route,
            elapsed=elapsed,
            tokens=total_tokens,
            episodes=episodes,
            skills=skills,
        )

class ResultCard:

    @staticmethod
    def render(
        console: Console,
        answer: str,
        route: str,
        elapsed: float,
        tokens: int = 0,
        streamed: bool = False
    ) -> None:
        if not streamed and answer:
            console.print(answer, style="white")
        console.print()

class ProgressIndicator:

    @staticmethod
    def create(console: Console, description: str) -> Progress:
        return Progress(
            SpinnerColumn(spinner_name="dots", style="#D77757"),
            TextColumn("[#D77757]{task.description}[/]"),
            BarColumn(bar_width=40, style="#D77757", complete_style="#D77757"),
            TextColumn("[#999999]{task.percentage:>3.0f}%[/]"),
            console=console,
            transient=True,
        )

class MemoryStats:

    @staticmethod
    def render(
        console: Console,
        episodes: int,
        high_quality_episodes: int,
        skills: int,
        mature_skills: int,
        cache_hit_rate: float
    ) -> None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("label", style="#999999", min_width=16)
        table.add_column("value", style="white")

        table.add_row(
            "Episodes",
            f"{episodes} [#666666]({high_quality_episodes} high-quality)[/]"
        )
        table.add_row(
            "Compiled Skills",
            f"{skills} [#666666]({mature_skills} mature)[/]"
        )

        hit_color = "#4EBA65" if cache_hit_rate > 0.5 else "#FFC107" if cache_hit_rate > 0.2 else "#FF6B80"
        table.add_row(
            "Cache hit rate",
            f"[{hit_color}]{cache_hit_rate:.0%}[/]"
        )

        panel = Panel(
            table,
            title="[#D77757]◆ Memory[/]",
            border_style="#444444",
            padding=(1, 2),
        )

        console.print(panel)

class IntentBadge:

    INTENT_COLORS = {
        "QUESTION": "#5599FF",
        "EXPLORE": "#AA55FF",
        "EDIT": "#FF9955",
        "DEBUG": "#FF6B80",
        "REFACTOR": "#FFC107",
    }

    INTENT_ICONS = {
        "QUESTION": "?",
        "EXPLORE": "▷",
        "EDIT": "✎",
        "DEBUG": "✖",
        "REFACTOR": "↻",
    }

    @staticmethod
    def render(console: Console, intent: str) -> None:
        color = IntentBadge.INTENT_COLORS.get(intent, "#999999")
        icon = IntentBadge.INTENT_ICONS.get(intent, "◆")

        badge = Text()
        badge.append(f"{icon} ", style=color)
        badge.append(intent, style=color)

        console.print(badge, end=" ")

class SessionStats:

    @staticmethod
    def render(
        console: Console,
        queries: int,
        cache_hits: int,
        avg_response_time: float,
        episodes: int,
        skills: int
    ) -> None:
        cache_rate = (cache_hits / queries * 100) if queries > 0 else 0

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("label", style="#999999", min_width=16)
        table.add_column("value", style="white")

        table.add_row("Queries", str(queries))
        hit_color = "#4EBA65" if cache_rate > 50 else "#FFC107" if cache_rate > 20 else "#999999"
        table.add_row("Cache hits", f"{cache_hits} [{hit_color}]({cache_rate:.0f}%)[/]")
        resp_color = "#4EBA65" if avg_response_time < 2 else "#FFC107" if avg_response_time < 5 else "#D77757"
        table.add_row("Avg response", f"[{resp_color}]{avg_response_time:.1f}s[/]")
        table.add_row("Memory", f"{episodes} episodes, {skills} skills")

        panel = Panel(
            table,
            title="[#D77757]◆ Session[/]",
            border_style="#444444",
            padding=(1, 2),
        )

        console.print(panel)
