"""Static visual demo of all CLI blocks.

Run with `python scripts/demo_blocks.py` to render every reusable
display block in sequence. This is a developer reference for the
look-and-feel of the CLI; the agent runtime hooks them up live in
later phases.
"""

from __future__ import annotations

import time

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


SNAKE_README = """# Snake Game

Classic Snake game built with Pygame.

## Installation

```bash
pip install pygame
```
""" + "\n".join(f"<!-- line {i} -->" for i in range(10, 37))


def main() -> None:
    console = Console()

    blocks.render_banner(console, repo_path=".", mode="Manual")

    # User query echo
    console.print(
        f"  [#D77757]❯[/] [white]Создай Snake game в папке snakegame[/]"
    )
    console.print()

    # Assistant text echo
    console.print("  [#D77757]●[/] [white]Создаю Snake game в папке "
                  "[#D77757]snakegame[/].[/]")
    console.print()

    # ── Bash mkdir → Done ────────────────────────────────────────
    blocks.render_bash(console, 'mkdir -p "snakegame"')
    console.print()

    # ── Write three files with numbered preview ──────────────────
    blocks.render_write(console, "snakegame/snake.py", SNAKE_PY)
    console.print()
    blocks.render_write(console, "snakegame/README.md", SNAKE_README)
    console.print()
    blocks.render_write(
        console, "snakegame/requirements.txt", "pygame>=2.5.0\n"
    )
    console.print()

    # ── Bash: short output ───────────────────────────────────────
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

    # ── Batch headers ────────────────────────────────────────────
    blocks.render_listing_directory(console, "snakegame")
    console.print()
    blocks.render_reading_files(
        console, ["snakegame/snake.py", "snakegame/README.md"]
    )
    console.print()
    blocks.render_searching(console, "**/*.py", in_files=2)
    console.print()

    # ── Edit (diff) ──────────────────────────────────────────────
    blocks.render_edit(
        console,
        "snakegame/snake.py",
        "@@ -7,3 +7,3 @@\n WIDTH, HEIGHT = 600, 400\n CELL_SIZE = 20\n-FPS = 10\n+FPS = 12\n",
    )
    console.print()

    # ── Todo panel ───────────────────────────────────────────────
    blocks.render_todos(
        console,
        [
            ("done",        "Read the existing snake game module"),
            ("done",        "Sketch a top-down design"),
            ("in_progress", "Implement the food spawn logic"),
            ("pending",     "Wire collision into the game loop"),
            ("pending",     "Write smoke tests"),
        ],
    )
    console.print()

    # ── Live status (animated) ───────────────────────────────────
    with blocks.LiveStatus("Synthesizing", console=console) as live:
        for _ in range(40):
            live.bump_tokens(out=4)
            time.sleep(0.05)

    # ── Status line ──────────────────────────────────────────────
    blocks.render_status_line(
        console,
        mode="Manual",
        route="SLOW",
        elapsed=1.43,
        tokens=1234,
        episodes=12,
        skills=3,
    )
    console.print()

    # ── Command table ────────────────────────────────────────────
    blocks.render_command_table(
        console,
        [
            ("help",   ["?"], "Show available commands"),
            ("status", [],    "Session summary"),
            ("repo",   [],    "Show or change working directory"),
            ("read",   [],    "Add a file to context"),
            ("mode",   [],    "Cycle autonomy mode"),
            ("clear",  [],    "Clear conversation"),
            ("memory", [],    "Inspect memory and skills"),
            ("diff",   [],    "Show uncommitted changes"),
            ("run",    [],    "Run shell command"),
            ("test",   [],    "Run test suite"),
            ("bench",  [],    "Run benchmarks"),
            ("exit",   ["q"], "Quit the REPL"),
        ],
    )

    # ── Confirmation panel preview ───────────────────────────────
    console.print()
    console.print("  [bold #D77757]Bash command[/]")
    console.print()
    console.print('  cd "/home/ubuntu/repos/NareCLI" && pwd')
    console.print("  Change to project directory", style="#999999")
    console.print()
    console.print("  Do you want to proceed?")
    console.print("  [bold #D77757]▸ 1.[/] Yes")
    console.print("    [#999999]2. Yes, allow reading from project[/]")
    console.print("    [#999999]3. No[/]")
    console.print()
    console.print(
        "  [#444444]Esc to cancel  ·  Tab to amend  ·  ctrl+e to explain[/]"
    )
    console.print()


if __name__ == "__main__":
    main()
