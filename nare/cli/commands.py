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
            # Reset cached AgentLoop instance to ensure clean state
            session._agent_loop = None
        else:
            session.use_agent_loop = not session.use_agent_loop
            # Reset AgentLoop when turning off
            if not session.use_agent_loop:
                session._agent_loop = None
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
        pass

class ModeCommand(Command):
    """Switch CLI mode."""
    name = "mode"
    aliases = ["m"]
    help = "Switch CLI mode (manual/research/autopilot/focus/verbose/interactive)"

    def execute(self, session, arg):
        from nare.cli.modes import get_mode_manager, Mode, MODE_DESCRIPTIONS

        mode_manager = get_mode_manager()

        if not arg:

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

        # Compute real stats from memory instead of fake ratios
        memory = session.agent.memory
        high_quality = sum(1 for ep in memory.episodes if ep.get('score', 0) >= 0.80)
        mature = sum(1 for sk in memory.compiled_skills if sk.get('use_count', 0) >= 3)

        # Real cache hit rate from router metrics
        cache_hit_rate = 0.0
        if hasattr(session.agent, 'router') and hasattr(session.agent.router, 'route_metrics'):
            rm_stats = session.agent.router.route_metrics.get_stats()
            total_q = rm_stats.get('total_queries', 0)
            if total_q > 0:
                fast_count = rm_stats.get('route_distribution', {}).get('FAST', 0)
                cache_hit_rate = fast_count

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
                continue
            except subprocess.TimeoutExpired:
                ui.print_error("Tests timed out (60s limit)")
                return
            except Exception:
                continue

        ui.print_warning("No test command found. Try: /test <command>")

class SkillsCommand(Command):
    """Show compiled skills."""
    name = "skills"
    help = "Show compiled skills library"

    def execute(self, session, arg):
        from rich.table import Table
        from rich.text import Text
        import json
        from pathlib import Path

        arg = (arg or "").strip().lower()

        # Handle /skills compile subcommand
        if arg == "compile":
            if not (hasattr(session, 'agent') and session.agent and hasattr(session.agent, 'evolution')):
                ui.print_error("Evolution engine not available — agent not initialized")
                return

            memory = session.agent.memory
            episodes = memory.episodes
            high_score = [ep for ep in episodes if ep.get('score', 0) >= 0.80]
            with_emb = [ep for ep in high_score if 'embedding' in ep]

            ui.console.print()
            ui.print_status("Episodes in memory", str(len(episodes)))
            ui.print_status("High-score (≥0.80)", str(len(high_score)))
            ui.print_status("With embeddings", str(len(with_emb)))

            if len(with_emb) < 3:
                ui.console.print()
                ui.print_warning(
                    f"Need ≥3 episodes with embeddings for compilation (have {len(with_emb)})"
                )
                ui.console.print("  [#666666]Keep using NARE — episodes are saved automatically from SLOW/HYBRID/REFLEX routes[/]")
                ui.console.print()
                return

            def _on_compile_done(before, after, error):
                if error:
                    ui.print_error(f"Compilation failed: {error}")
                elif after > before:
                    ui.print_success(f"Compiled {after - before} new skill(s) — total: {after}")
                    ui.console.print("  [#666666]Run /skills to inspect[/]")
                else:
                    ui.print_warning("No new skills compiled — episodes may be too diverse for clustering")

            ui.console.print()
            ui.print_status("Compilation", "starting", "info")
            session.agent.evolution.run_compilation_cycle(on_complete=_on_compile_done)
            ui.print_success("Compilation running in background")
            ui.console.print("  [#666666]Results will appear when complete[/]")
            ui.console.print()
            return

        # Try to get skills from agent first
        skills = None
        if hasattr(session, 'agent') and session.agent and hasattr(session.agent, 'memory'):
            skills = session.agent.memory.compiled_skills

        # If agent not initialized, try to load from file directly
        if not skills:
            skills_file = Path(session.repo_path) / '.nare_memory' / 'compiled_skills.json'
            if skills_file.exists():
                try:
                    with open(skills_file, 'r', encoding='utf-8') as f:
                        skills = json.load(f)
                except Exception as e:
                    ui.print_error(f"Failed to load skills: {e}")
                    return

        if not skills:
            ui.console.print()
            ui.console.print("  [#666666]No compiled skills yet[/]")
            ui.console.print("  [#666666]Skills are learned from successful task patterns[/]")
            ui.console.print("  [#666666]Tip: Use /skills compile to trigger manual compilation[/]")
            ui.console.print()
            return

        ui.console.print()
        ui.console.print(f"  [#D77757]★ Compiled Skills Library[/] ({len(skills)} skills)")
        ui.console.print()

        table = Table(show_header=True, header_style="#D77757", border_style="#666666")
        table.add_column("#", style="#666666", width=4)
        table.add_column("Pattern", style="#FFFFFF", width=40)
        table.add_column("Confidence", style="#999999", width=12)
        table.add_column("Uses", style="#999999", width=8)

        for idx, skill in enumerate(skills):
            pattern = skill.get('pattern', 'unknown')
            confidence = skill.get('confidence', 0)
            uses = skill.get('use_count', 0)

            # Truncate long patterns
            if len(pattern) > 37:
                pattern = pattern[:34] + "..."

            # Color code confidence
            if confidence >= 0.8:
                conf_color = "#4EBA65"  # green
            elif confidence >= 0.6:
                conf_color = "#FFC107"  # yellow
            else:
                conf_color = "#D77757"  # orange

            conf_text = Text(f"{confidence:.0%}", style=conf_color)

            table.add_row(
                str(idx),
                pattern,
                conf_text,
                str(uses)
            )

        ui.console.print(table)
        ui.console.print()

        if arg and arg.isdigit():
            # Show detailed view of specific skill
            skill_id = int(arg)
            if 0 <= skill_id < len(skills):
                skill = skills[skill_id]
                ui.console.print(f"  [#D77757]Skill #{skill_id} Details:[/]")
                ui.console.print()
                ui.console.print(f"  Pattern: {skill.get('pattern', 'unknown')}", style="#FFFFFF")
                ui.console.print(f"  Confidence: {skill.get('confidence', 0):.0%}", style="#999999")
                ui.console.print(f"  Uses: {skill.get('use_count', 0)}", style="#999999")
                ui.console.print(f"  Success: {skill.get('success_count', 0)}", style="#999999")
                ui.console.print(f"  Failures: {skill.get('failure_count', 0)}", style="#999999")
                ui.console.print()
                ui.console.print("  Code:", style="#D77757")
                ui.console.print()
                code = skill.get('code', 'No code available')
                for line in code.split('\n')[:20]:  # Show first 20 lines
                    ui.console.print(f"    {line}", style="#999999")
                if code.count('\n') > 20:
                    ui.console.print(f"    [#666666]... +{code.count(chr(10)) - 20} more lines[/]")
                ui.console.print()
            else:
                ui.print_error(f"Skill #{skill_id} not found")
        else:
            ui.console.print("  [#666666]Tip: Use /skills <number> to see skill details[/]")
            ui.console.print()

class MetricsCommand(Command):
    """Show routing metrics."""
    name = "metrics"
    help = "Show routing metrics and skill usage"

    def execute(self, session, arg):
        if not hasattr(session, 'agent') or not session.agent:
            ui.print_warning("Agent not initialized yet")
            return

        if not hasattr(session.agent, 'router') or not hasattr(session.agent.router, 'route_metrics'):
            ui.print_warning("Metrics not available")
            return

        stats = session.agent.router.route_metrics.get_stats()

        if stats["total_queries"] == 0:
            ui.console.print()
            ui.console.print("  [#666666]No queries yet[/]")
            ui.console.print()
            return

        ui.console.print()
        ui.console.print(f"  [#D77757]Routing Metrics[/] ({stats['total_queries']} queries)")
        ui.console.print()

        # Route distribution
        ui.console.print("  Route Distribution:", style="#D77757")
        for route, pct in sorted(stats["route_distribution"].items(), key=lambda x: x[1], reverse=True):
            bar_length = int(pct * 30)
            bar = "█" * bar_length
            ui.console.print(f"    {route:20s} {pct:5.1%} {bar}", style="#999999")

        # Amortization stats
        if hasattr(session.agent, 'get_amortization_stats'):
            amor = session.agent.get_amortization_stats()
            ui.console.print()
            ui.console.print("  Amortization:", style="#D77757")
            ui.console.print(f"    α_t (empirical)     {amor.get('alpha_t', 0):.1%}", style="#999999")
            ui.console.print(f"    α_t (theoretical)   {amor.get('alpha_t_theoretical', 0):.1%}", style="#999999")
            ui.console.print(f"    Blended cost        {amor.get('blended_cost', 0):.1f}", style="#999999")
            ui.console.print(f"    Amortized queries   {amor.get('amortized_queries', 0)} / {amor.get('total_queries', 0)}", style="#999999")
            ui.console.print(f"    Memory size         {amor.get('memory_size', 0)} episodes", style="#999999")
            ui.console.print(f"    Skills count        {amor.get('skills_count', 0)}", style="#999999")

        # Top skills
        if stats["top_skills"]:
            ui.console.print()
            ui.console.print("  Top Skills:", style="#D77757")
            for pattern, count in stats["top_skills"]:
                if len(pattern) > 50:
                    pattern = pattern[:47] + "..."
                ui.console.print(f"    {pattern:50s} {count:3d} uses", style="#999999")

        ui.console.print()

class APIKeyCommand(Command):
    """Manage API keys for LLM providers."""
    name = "apikey"
    aliases = ["api", "key"]
    help = "Manage API keys (Anthropic, OpenAI, Google)"

    def execute(self, session: NareSession, arg: str):
        from nare.config.api_keys import get_api_key_manager

        manager = get_api_key_manager()

        if not arg:
            # Show current status
            ui.console.print()
            ui.console.print("  [#D77757]API Keys Status[/]")
            ui.console.print()

            for provider, info in manager.SUPPORTED_PROVIDERS.items():
                has_key = manager.get_key(provider) is not None
                status = "[#4EBA65]✓[/]" if has_key else "[#666666]✗[/]"
                model = manager.get_model(provider)
                model_text = f" → {model}" if model else ""
                ui.console.print(f"  {status} {info['name']}{model_text}")
                if not has_key:
                    ui.console.print(f"      [#666666]Get key: {info['url']}[/]")

            ui.console.print()
            ui.console.print("  [#999999]Usage: /apikey <provider> <key>[/]")
            ui.console.print("  [#999999]Example: /apikey anthropic sk-ant-...[/]")
            ui.console.print()
            return

        # Parse provider and key
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            ui.print_error("Usage: /apikey <provider> <key>")
            ui.console.print("  Providers: anthropic, openai, google")
            return

        provider, key = parts

        if provider not in manager.SUPPORTED_PROVIDERS:
            ui.print_error(f"Unknown provider: {provider}")
            ui.console.print("  Available: anthropic, openai, google")
            return

        # Set key
        try:
            manager.set_key(provider, key, save=True)
            info = manager.SUPPORTED_PROVIDERS[provider]
            ui.console.print()
            ui.console.print(f"  [#4EBA65]✓ {info['name']} API key saved[/]")
            ui.console.print()
        except Exception as e:
            ui.print_error(f"Failed to save key: {e}")

class ResumeCommand(Command):
    """Resume an interrupted AgentLoop session."""
    name = "resume"
    help = "Resume an interrupted agent session"

    def execute(self, session: NareSession, arg: str):
        import json
        import os
        from nare.cli.display.spinner import WaitingSpinner

        state_path = os.path.join(session.repo_path, ".nare_memory", "session_state.json")
        if not os.path.exists(state_path):
            ui.print_warning("No saved session state found to resume.")
            return

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            ui.console.print()
            ui.console.print("  [#D77757]Resuming AgentLoop Session...[/]")
            ui.console.print(f"  [#999999]Query:[/] {state.get('query', '')[:100]}...")
            ui.console.print(f"  [#999999]Iterations:[/] {state.get('budget_iterations', 0)}")
            ui.console.print(f"  [#999999]Tokens:[/] {state.get('budget_tokens_in', 0) + state.get('budget_tokens_out', 0)}")
            ui.console.print()

            from nare.cli.display.agent_renderer import ThinkingDisplay
            with ThinkingDisplay() as display:
                session.solve(query="", thinking_display=display, resume_state=state)
                
        except Exception as e:
            ui.print_error(f"Failed to resume session: {e}")

class AutonomyCommand(Command):
    name = "autonomy"
    aliases = ["auto"]
    help = "Change agent autonomy level (supervised/assisted/autonomous)"

    def execute(self, session: NareSession, arg: str):
        from nare.cli.autonomy_level import AutonomyLevel, AUTONOMY_DESCRIPTIONS

        if not arg:
            # Show current level
            current = session.autonomy_level
            ui.console.print()
            ui.console.print(f"  [#D77757]Current autonomy level:[/] [bold]{current.value}[/]")
            ui.console.print(f"  [#999999]{AUTONOMY_DESCRIPTIONS[current]}[/]")
            ui.console.print()
            ui.console.print("  [#666666]Available levels:[/]")
            for level in AutonomyLevel:
                marker = "→" if level == current else " "
                ui.console.print(f"  {marker} [#D77757]{level.value}[/] - {AUTONOMY_DESCRIPTIONS[level]}")
            ui.console.print()
            ui.console.print("  [#666666]Usage: /autonomy <level>[/]")
            return

        # Set new level
        level_name = arg.strip().lower()
        try:
            new_level = AutonomyLevel(level_name)
            session.autonomy_level = new_level
            ui.console.print()
            ui.console.print(f"  [#4EBA65]✓[/] Autonomy level set to: [bold]{new_level.value}[/]")
            ui.console.print(f"  [#999999]{AUTONOMY_DESCRIPTIONS[new_level]}[/]")
            ui.console.print()
        except ValueError:
            ui.print_error(f"Invalid autonomy level: {level_name}")
            ui.console.print("  [#666666]Valid levels: supervised, assisted, autonomous[/]")

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
    AutonomyCommand(),
    MemoryCommand(),
    SkillsCommand(),
    MetricsCommand(),
    APIKeyCommand(),
    TokensCommand(),
    UndoCommand(),
    DiffCommand(),
    RunCommand(),
    TestCommand(),
    BenchCommand(),
    ResumeCommand(),
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
