"""Render the demo blocks to an SVG screenshot.

Used to attach a visual reference to PRs and the README.

Usage:
    PYTHONPATH=. python scripts/render_demo_svg.py docs/cli_demo.svg
"""

from __future__ import annotations

import sys
from rich.console import Console

from nare.cli.display import blocks


SNAKE_PY = """import pygame
import random
import sys

pygame.init()

WIDTH, HEIGHT = 600, 400
CELL_SIZE = 20
FPS = 10
""" + "\n".join(f"# line {i}" for i in range(11, 148))


def main(out_path: str) -> None:
    console = Console(record=True, width=92)

    blocks.render_banner(console, repo_path=".", mode="Manual")

    console.print(
        f"  [#D77757]❯[/] [white]Создай Snake game в папке snakegame[/]"
    )
    console.print()
    console.print(
        "  [#D77757]●[/] [white]Создаю Snake game в папке "
        "[#D77757]snakegame[/].[/]"
    )
    console.print()

    blocks.render_bash(console, 'mkdir -p "snakegame"')
    console.print()
    blocks.render_write(console, "snakegame/snake.py", SNAKE_PY)
    console.print()
    blocks.render_write(
        console, "snakegame/requirements.txt", "pygame>=2.5.0\n"
    )
    console.print()

    blocks.render_bash(
        console,
        "git log --oneline -3",
        output=(
            "deccd26 Merge pull request #1\n"
            "cfc39de phase1: wire CLI to NARE core\n"
            "aa76be1 Initial commit"
        ),
        exit_code=0,
    )
    console.print()

    blocks.render_reading_files(
        console, ["snakegame/snake.py", "snakegame/README.md"]
    )
    console.print()

    blocks.render_edit(
        console,
        "snakegame/snake.py",
        "@@ -7,3 +7,3 @@\n WIDTH, HEIGHT = 600, 400\n CELL_SIZE = 20\n-FPS = 10\n+FPS = 12\n",
    )
    console.print()

    blocks.render_status_line(
        console,
        mode="Manual",
        route="SLOW",
        elapsed=1.43,
        tokens=1234,
        episodes=12,
        skills=3,
    )

    console.save_svg(out_path, title="NARE CLI")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/cli_demo.svg"
    main(out)
