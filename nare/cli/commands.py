"""
Commands — Slash command handling via Command pattern

Component: CLI Command System
Purpose: Handles slash commands in REPL
Architecture: Command pattern with registry

Each command is a class with:
- name: Command name (e.g., "help")
- aliases: Alternative names (e.g., ["?"])
- help: Description shown in /help
- execute(): Command implementation

Commands are registered in COMMANDS list and mapped by name/alias.
"""

import os
import sys
import subprocess
from typing import Optional

from nare.cli.session import NareSession
from nare.cli.display import ui


class Command:
    """Base command interface.

    Attributes:
        name: Command name (without /)
        aliases: Alternative names
        help: Description for /help display

    Methods:
        execute: Run command with session and argument
    """
    name: str = ""
    aliases: list[str] = []
    help: str = ""

    def execute(self, session: NareSession, arg: str) -> None:
        """Execute command.

        Args:
            session: NareSession instance
            arg: Command argument string
        """
        raise NotImplementedError


class HelpCommand(Command):
    name = "help"
    aliases = ["?"]
    help = "Show available commands"

    def execute(self, session, arg):
        from nare.cli.display import blocks
        rows = [(cmd.name, list(cmd.aliases or []), cmd.help) for cmd in COMMANDS]
        blocks.render_command_table(ui.console, rows)


class AgentCommand(Command):
    """Toggle the Phase-3 tool-calling agent loop.

    Off (default) → legacy ReasoningRouter (5-tier routing + verified synthesis).
    On            → AgentLoop with live tool-call rendering and budgets.
    """

    name = "agent"
    help = "Toggle tool-calling agent loop on/off"

    def execute(self, session, arg):
        arg = (arg or "").strip().lower()
        if arg in ("on", "true", "1", "yes"):
            session.use_agent_loop = True
        elif arg in ("off", "false", "0", "no"):
            session.use_agent_loop = False
        else:
            session.use_agent_loop = not session.use_agent_loop
        state = "on" if session.use_agent_loop else "off"
        ui.print_status("Agent loop", state, "info" if session.use_agent_loop else "warning")


class StatusCommand(Command):
    name = "status"
    help = "Show agent status"

    def execute(self, session, arg):
        info = session.get_status()
        ui.console.print()
        ui.print_status("Repo", info["repo"], "info")
        ui.print_status("Context files", str(info["context_files"]))
        ui.print_status("Model", info["model"])
        if info["agent_ready"]:
            ui.print_status("Episodes", str(info.get("episodes", 0)))
            ui.print_status("Skills", str(info.get("skills", 0)))
        else:
            ui.print_status("Agent", "not initialized yet", "warning")
        ui.console.print()


class RepoCommand(Command):
    name = "repo"
    help = "Set/show working repository"

    def execute(self, session, arg):
        if not arg:
            ui.print_status("Current repo", session.repo_path, "info")
            return
        if session.set_repo(arg):
            ui.print_success(f"Repo set to {session.repo_path}")
        else:
            ui.print_error(f"Not a directory: {os.path.abspath(arg)}")


class FilesCommand(Command):
    name = "files"
    aliases = ["ls"]
    help = "List files in repo"

    def execute(self, session, arg):
        from nare.cli.display.spinner import WaitingSpinner

        try:
            with WaitingSpinner("Loading files", delay=0.15, color="bright_yellow"):
                result = subprocess.run(
                    ["git", "ls-files"],
                    cwd=session.repo_path,
                    capture_output=True, text=True, timeout=10,
                )

            if result.returncode == 0:
                files = result.stdout.strip().split("\n")
                # Group by top-level dir
                dirs: dict[str, list[str]] = {}
                for f in files[:200]:
                    parts = f.split("/")
                    d = parts[0] if len(parts) > 1 else "."
                    dirs.setdefault(d, []).append("/".join(parts[1:]) if len(parts) > 1 else parts[0])

                ui.console.print()
                for d in sorted(dirs.keys())[:25]:
                    ui.console.print(f"  [info]{d}/[/]")
                    for fname in sorted(dirs[d])[:8]:
                        ui.console.print(f"    [muted]{fname}[/]")
                    if len(dirs[d]) > 8:
                        ui.console.print(f"    [dim]... +{len(dirs[d]) - 8} more[/]")
                ui.console.print(f"\n  [muted]Total: {len(files)} files[/]")
                ui.console.print()
            else:
                ui.print_warning("Not a git repository")
        except Exception as e:
            ui.print_error(str(e))


class ReadCommand(Command):
    name = "read"
    help = "Read a file into context"

    def execute(self, session, arg):
        from nare.cli.display.spinner import WaitingSpinner

        if not arg:
            ui.print_warning("Usage: /read <filepath>")
            return

        with WaitingSpinner("Reading file", delay=0.15, color="bright_yellow"):
            content = session.read_file(arg)

        if content:
            ui.print_file_loaded(arg, content.count("\n"))
        else:
            ui.print_error(f"File not found: {arg}")


class ClearCommand(Command):
    name = "clear"
    help = "Clear context and history"

    def execute(self, session, arg):
        session.clear_context()
        ui.print_success("Context cleared")


class BenchCommand(Command):
    name = "bench"
    help = "Run SWE-bench (e.g. /bench 10)"

    def execute(self, session, arg):
        from nare.cli.display.spinner import WaitingSpinner

        n = int(arg) if arg and arg.isdigit() else 5
        ui.console.print(f"\n  [accent]*[/] Starting SWE-bench with {n} tasks...\n")
        cmd = [
            sys.executable,
            "benchmarks/swe_bench_official.py",
            "--max-tasks", str(n),
            "--output", "predictions_cli.jsonl",
        ]
        try:
            with WaitingSpinner(f"Running benchmark ({n} tasks)", delay=0.15, color="bright_yellow"):
                subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except KeyboardInterrupt:
            ui.print_warning("Benchmark interrupted")


class ThemeCommand(Command):
    name = "theme"
    help = "Change color theme (dark/light/dark-ansi/light-ansi)"

    def execute(self, session, arg):
        from nare.cli.theme import THEMES
        from nare.cli import ui

        if not arg:
            # Show current theme
            current = ui._current_theme
            ui.console.print()
            ui.console.print(f"  Current theme: [accent]{current}[/]")
            ui.console.print(f"  Available: {', '.join(THEMES.keys())}")
            ui.console.print()
            return

        if arg not in THEMES:
            ui.print_error(f"Unknown theme: {arg}. Available: {', '.join(THEMES.keys())}")
            return

        ui.set_theme(arg)
        ui.print_success(f"Theme set to {arg}")


class CdCommand(Command):
    name = "cd"
    help = "Change working directory"

    def execute(self, session, arg):
        if not arg:
            ui.print_status("Current directory", session.repo_path, "info")
            return

        # Expand ~ and resolve path
        path = os.path.expanduser(arg)
        path = os.path.abspath(path)

        if not os.path.isdir(path):
            ui.print_error(f"Not a directory: {path}")
            return

        if session.set_repo(path):
            ui.print_success(f"Changed to {session.repo_path}")
        else:
            ui.print_error(f"Failed to change directory")


class ExitCommand(Command):
    name = "exit"
    aliases = ["quit", "q"]
    help = "Quit NARE"

    def execute(self, session, arg):
        pass  # Handled by REPL loop


class ModeCommand(Command):
    """Switch CLI mode."""
    name = "mode"
    aliases = ["m"]
    help = "Switch CLI mode (manual/research/autopilot/focus/verbose/interactive)"

    def execute(self, session, arg):
        from nare.cli.modes import get_mode_manager, Mode, MODE_DESCRIPTIONS

        mode_manager = get_mode_manager()

        if not arg:
            # Show current mode
            current = mode_manager.current_mode
            description = MODE_DESCRIPTIONS.get(current, "")

            ui.console.print()
            ui.console.print(f"  Current mode: [#D77757]{current.value}[/]")
            ui.console.print(f"  {description}", style="#999999")
            ui.console.print()
            ui.console.print("  Available modes:", style="#666666")
            for mode in Mode:
                desc = MODE_DESCRIPTIONS.get(mode, "")
                ui.console.print(f"    [#D77757]{mode.value.lower()}[/] - {desc}", style="#999999")
            ui.console.print()
            return

        # Parse mode name
        mode_name = arg.strip().upper()
        try:
            mode = Mode[mode_name]
            mode_manager.set_mode(mode)
            description = MODE_DESCRIPTIONS.get(mode, "")
            ui.console.print()
            ui.console.print(f"  Mode set to [#D77757]{mode.value}[/]")
            ui.console.print(f"  {description}", style="#999999")
            ui.console.print()
        except KeyError:
            ui.print_error(f"Unknown mode: {arg}")
            ui.console.print("  Available: manual, research, autopilot, focus, verbose, interactive", style="#666666")


class MemoryCommand(Command):
    """Show memory statistics."""
    name = "memory"
    aliases = ["mem"]
    help = "Show memory statistics"

    def execute(self, session, arg):
        from nare.cli.display.components import MemoryStats

        info = session.get_status()

        if not info.get("agent_ready"):
            ui.print_warning("Agent not initialized yet")
            return

        episodes = info.get("episodes", 0)
        skills = info.get("skills", 0)

        # Estimate high-quality and mature counts
        high_quality = int(episodes * 0.3)
        mature = int(skills * 0.6)
        cache_hit_rate = 0.0

        ui.console.print()
        MemoryStats.render(
            ui.console,
            episodes=episodes,
            high_quality_episodes=high_quality,
            skills=skills,
            mature_skills=mature,
            cache_hit_rate=cache_hit_rate
        )
        ui.console.print()


class AddCommand(Command):
    """Add file to context (alias for /read)."""
    name = "add"
    help = "Add file to context"

    def execute(self, session, arg):
        from nare.cli.display.spinner import WaitingSpinner

        if not arg:
            # Show current context files
            if session.context_files:
                ui.console.print()
                ui.console.print("  Context files:", style="#FFA500")
                for path in session.context_files:
                    ui.console.print(f"    [#999999]{path}[/]")
                ui.console.print()
            else:
                ui.console.print()
                ui.console.print("  [#666666]No files in context[/]")
                ui.console.print()
            return

        with WaitingSpinner("Adding file", delay=0.15, color="bright_yellow"):
            content = session.read_file(arg)

        if content:
            ui.print_file_loaded(arg, content.count("\n"))
        else:
            ui.print_error(f"File not found: {arg}")


class DropCommand(Command):
    """Remove file from context."""
    name = "drop"
    help = "Remove file from context"

    def execute(self, session, arg):
        if not arg:
            ui.print_warning("Usage: /drop <filepath>")
            return

        if arg in session.context_files:
            del session.context_files[arg]
            ui.print_success(f"Removed from context: {arg}")
        else:
            ui.print_warning(f"File not in context: {arg}")


class TokensCommand(Command):
    """Show token usage statistics."""
    name = "tokens"
    help = "Show token usage statistics"

    def execute(self, session, arg):
        total_in = getattr(session, '_total_tokens_in', 0)
        total_out = getattr(session, '_total_tokens_out', 0)
        queries = getattr(session, '_query_count', 0)

        ui.console.print()
        ui.console.print("  Token Usage:", style="#D77757")
        ui.console.print(f"    Queries: {queries}", style="#999999")
        ui.console.print(f"    Tokens in: {total_in:,}", style="#999999")
        ui.console.print(f"    Tokens out: {total_out:,}", style="#999999")
        ui.console.print(f"    Total: {total_in + total_out:,}", style="#999999")

        if queries > 0:
            avg = (total_in + total_out) / queries
            ui.console.print(f"    Avg per query: {avg:,.0f}", style="#999999")
        ui.console.print()


class UndoCommand(Command):
    """Undo last change via git."""
    name = "undo"
    help = "Undo last change"

    def execute(self, session, arg):
        history = getattr(session, '_history', [])
        if not history:
            ui.print_warning("No changes to undo")
            return

        last_action = history[-1]

        # Revert via git
        try:
            result = subprocess.run(
                ["git", "revert", "--no-edit", last_action['commit']],
                cwd=session.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            history.pop()
            ui.print_success(f"Undone: {last_action.get('message', 'last change')}")
        except subprocess.CalledProcessError as e:
            ui.print_error(f"Failed to undo: {e.stderr if e.stderr else str(e)}")
        except FileNotFoundError:
            ui.print_error("Git not found. Make sure git is installed.")


class DiffCommand(Command):
    """Show git diff of changes."""
    name = "diff"
    help = "Show uncommitted changes"

    def execute(self, session, arg):
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=session.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            if result.stdout:
                ui.console.print()
                ui.console.print(result.stdout)
            else:
                ui.console.print()
                ui.console.print("  [#666666]No uncommitted changes[/]")
                ui.console.print()
        except FileNotFoundError:
            ui.print_error("Git not found. Make sure git is installed.")
        except subprocess.CalledProcessError as e:
            ui.print_error(f"Git diff failed: {e.stderr if e.stderr else str(e)}")


class RunCommand(Command):
    """Run shell command."""
    name = "run"
    help = "Run shell command"

    def execute(self, session, arg):
        from nare.cli.display.spinner import WaitingSpinner

        if not arg:
            ui.print_warning("Usage: /run <command>")
            return

        ui.console.print()
        ui.console.print(f"  Running: {arg}", style="#FFA500")
        ui.console.print()

        try:
            with WaitingSpinner(f"Executing", delay=0.15, color="bright_yellow"):
                result = subprocess.run(
                    arg,
                    shell=True,
                    cwd=session.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

            if result.stdout:
                ui.console.print(result.stdout)
            if result.stderr:
                ui.console.print(result.stderr, style="#FFA500")

            if result.returncode != 0:
                ui.console.print()
                ui.console.print(f"  [#FFA500]Exit code: {result.returncode}[/]")
                ui.console.print()
        except subprocess.TimeoutExpired:
            ui.print_error("Command timed out (30s limit)")
        except Exception as e:
            ui.print_error(f"Failed to run command: {str(e)}")


class TestCommand(Command):
    """Run tests."""
    name = "test"
    help = "Run tests"

    def execute(self, session, arg):
        from nare.cli.display.spinner import WaitingSpinner

        # Try common test commands
        test_commands = [
            arg if arg else None,
            "pytest",
            "python -m pytest",
            "npm test",
            "cargo test",
            "go test ./...",
        ]

        for cmd in test_commands:
            if cmd is None:
                continue

            ui.console.print()
            ui.console.print(f"  Running: {cmd}", style="#FFA500")
            ui.console.print()

            try:
                with WaitingSpinner(f"Running tests", delay=0.15, color="bright_yellow"):
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        cwd=session.repo_path,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )

                if result.stdout:
                    ui.console.print(result.stdout)
                if result.stderr:
                    ui.console.print(result.stderr, style="#FFA500")

                if result.returncode == 0:
                    ui.print_success("Tests passed")
                    return
                else:
                    ui.console.print()
                    ui.console.print(f"  [#FFA500]Tests failed (exit code: {result.returncode})[/]")
                    ui.console.print()
                    return
            except FileNotFoundError:
                continue  # Try next command
            except subprocess.TimeoutExpired:
                ui.print_error("Tests timed out (60s limit)")
                return
            except Exception:
                continue  # Try next command

        ui.print_warning("No test command found. Try: /test <command>")


# Command Registry
COMMANDS: list[Command] = [
    HelpCommand(),
    AgentCommand(),
    StatusCommand(),
    RepoCommand(),
    CdCommand(),
    FilesCommand(),
    ReadCommand(),
    AddCommand(),
    DropCommand(),
    ClearCommand(),
    ThemeCommand(),
    ModeCommand(),
    MemoryCommand(),
    TokensCommand(),
    UndoCommand(),
    DiffCommand(),
    RunCommand(),
    TestCommand(),
    BenchCommand(),
    ExitCommand(),
]

COMMAND_MAP: dict[str, Command] = {}
for cmd in COMMANDS:
    COMMAND_MAP[cmd.name] = cmd
    for alias in cmd.aliases:
        COMMAND_MAP[alias] = cmd


def dispatch(session: NareSession, raw: str) -> Optional[str]:
    """Dispatch a slash command.

    Args:
        session: NareSession instance
        raw: Raw command string (e.g., "/help" or "/read file.py")

    Returns:
        'exit' if should quit, else None
    """
    parts = raw.split(maxsplit=1)
    name = parts[0].lstrip("/").lower()
    arg = parts[1] if len(parts) > 1 else ""

    if name in ("exit", "quit", "q"):
        return "exit"

    cmd = COMMAND_MAP.get(name)
    if cmd:
        cmd.execute(session, arg)
    else:
        ui.print_warning(f"Unknown command: /{name}. Type /help")
    return None
