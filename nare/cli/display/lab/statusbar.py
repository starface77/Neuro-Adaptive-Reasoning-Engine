"""Minimal laboratory-style status bar."""

from rich.text import Text

def print_statusbar(console, session_time: str, queries: int, avg_latency: float, memory_mb: float, cache_hits: int, cache_total: int):
    """Print minimal status bar."""

    status = Text()
    status.append(f"session {session_time}", style="#444444")
    status.append(f"  {queries} queries", style="#444444")

    console.print(status)
    console.print()
