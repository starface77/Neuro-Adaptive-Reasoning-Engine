"""Component library for the NARE CLI.

Provides composable UI primitives for the interactive REPL:
- StatusBar       — bottom status line (route, timing, memory stats)
- ResultCard      — final answer block with metadata footer
- ProgressIndicator — long-running progress bar
- MemoryStats     — memory statistics panel
- IntentBadge     — small badge showing the classified intent

Design principles: clean, minimal, no decorative noise. Routes are
color-coded (FAST=green, REFLEX=cyan, HYBRID=yellow, SLOW=orange).
"""

from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text
from rich.console import Console
from typing import Optional

from . import blocks

class StatusBar:
    """Bottom status bar with premium route indicators."""

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
        """Render the bottom status line.

        Layout::

            Manual  claude-sonnet-4  project  1.2k tok  ◆ SLOW · 1.4s
        """
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
    """Premium result card with animated route badge."""

    @staticmethod
    def render(
        console: Console,
        answer: str,
        route: str,
        elapsed: float,
        tokens: int = 0,
        streamed: bool = False
    ) -> None:
        """Render the model's answer.

        We deliberately do NOT print a status line here — the canonical
        bottom status bar is rendered by ``StatusBar.render`` once per
        turn. Printing it twice produced two ``◆ AGENT_LOOP 9.8s``
        lines stacked on top of each other.
        """
        if not streamed and answer:
            console.print(answer, style="white")
        console.print()

class ProgressIndicator:
    """Progress bar for long operations.

    ⠋ Generating candidates... [████████░░] 80%

    Usage:
        with ProgressIndicator.create(console, "Processing") as progress:
            task = progress.add_task("Generating", total=100)
            for i in range(100):
                progress.update(task, advance=1)
    """

    @staticmethod
    def create(console: Console, description: str) -> Progress:
        """Create progress indicator.

        Args:
            console: Rich console instance
            description: Task description

        Returns:
            Progress instance (use as context manager)
        """
        return Progress(
            SpinnerColumn(spinner_name="dots", style="#D77757"),
            TextColumn("[#D77757]{task.description}[/]"),
            BarColumn(bar_width=40, style="#D77757", complete_style="#D77757"),
            TextColumn("[#999999]{task.percentage:>3.0f}%[/]"),
            console=console,
            transient=True,
        )

class MemoryStats:
    """Memory statistics panel.

    Memory:
      Episodes: 45 (12 high-quality)
      Skills: 12 (8 mature)
      Cache hit rate: 67%
    """

    @staticmethod
    def render(
        console: Console,
        episodes: int,
        high_quality_episodes: int,
        skills: int,
        mature_skills: int,
        cache_hit_rate: float
    ) -> None:
        """Render memory statistics.

        Args:
            console: Rich console instance
            episodes: Total episodes in memory
            high_quality_episodes: High-quality episodes
            skills: Total compiled skills
            mature_skills: Mature skills (used 3+ times)
            cache_hit_rate: Cache hit rate (0.0-1.0)
        """
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("label", style="#999999")
        table.add_column("value", style="white")

        table.add_row(
            "Episodes:",
            f"{episodes} ({high_quality_episodes} high-quality)"
        )
        table.add_row(
            "Skills:",
            f"{skills} ({mature_skills} mature)"
        )
        table.add_row(
            "Cache hit rate:",
            f"{cache_hit_rate:.0%}"
        )

        panel = Panel(
            table,
            title="[#D77757]Memory[/]",
            border_style="#666666",
            padding=(1, 2),
        )

        console.print(panel)

class IntentBadge:
    """Badge showing query intent type.

    Displays: [QUESTION] [EXPLORE] [EDIT]

    Color coding:
    - QUESTION: blue - simple Q&A
    - EXPLORE: purple - codebase exploration
    - EDIT: orange - code modification
    """

    INTENT_COLORS = {
        "QUESTION": "#5599FF",
        "EXPLORE": "#AA55FF",
        "EDIT": "#FF9955",
    }

    @staticmethod
    def render(console: Console, intent: str) -> None:
        """Render intent badge.

        Args:
            console: Rich console instance
            intent: Intent type (QUESTION/EXPLORE/EDIT)
        """
        color = IntentBadge.INTENT_COLORS.get(intent, "#999999")

        badge = Text()
        badge.append("[", style="#666666")
        badge.append(intent, style=color)
        badge.append("]", style="#666666")

        console.print(badge, end=" ")

class SessionStats:
    """Session statistics display.

    ┌─ Session Stats ──────────────────┐
    │ Queries: 5                       │
    │ Cache hits: 3 (60%)              │
    │ Avg response: 2.1s               │
    │ Memory: 45 episodes, 12 skills   │
    └──────────────────────────────────┘
    """

    @staticmethod
    def render(
        console: Console,
        queries: int,
        cache_hits: int,
        avg_response_time: float,
        episodes: int,
        skills: int
    ) -> None:
        """Render session statistics.

        Args:
            console: Rich console instance
            queries: Total queries in session
            cache_hits: Number of cache hits
            avg_response_time: Average response time
            episodes: Memory episodes
            skills: Compiled skills
        """
        cache_rate = (cache_hits / queries * 100) if queries > 0 else 0

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("label", style="#999999")
        table.add_column("value", style="white")

        table.add_row("Queries:", str(queries))
        table.add_row("Cache hits:", f"{cache_hits} ({cache_rate:.0f}%)")
        table.add_row("Avg response:", f"{avg_response_time:.1f}s")
        table.add_row("Memory:", f"{episodes} episodes, {skills} skills")

        panel = Panel(
            table,
            title="[#D77757]Session Stats[/]",
            border_style="#666666",
            padding=(1, 2),
        )

        console.print(panel)
