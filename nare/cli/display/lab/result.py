"""
Laboratory-style result display - simple and working.
"""

from rich.text import Text
from ..truncate import smart_truncate_answer


def print_result(console, answer: str, route: str, latency: float, tokens_in: int = 0, tokens_out: int = 0, confidence: float = 100.0, verified: bool = True):
    """Print result with simple styling."""

    # Smart truncate long answers
    truncated_answer, hint = smart_truncate_answer(answer, max_lines=50)

    # Answer - clean white text
    console.print()
    console.print(truncated_answer, style="white")

    # Show hint if truncated
    if hint:
        console.print(hint)

    console.print()

    # Metadata line
    metadata = Text()

    # Route name
    metadata.append(route, style="#D77757")

    # Metrics in subtle gray
    metadata.append(f"  {latency:.1f}s", style="#666666")
    if tokens_in + tokens_out > 0:
        metadata.append(f"  {tokens_in + tokens_out} tokens", style="#666666")

    console.print(metadata)
    console.print()
