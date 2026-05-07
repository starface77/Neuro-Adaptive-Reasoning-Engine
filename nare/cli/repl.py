"""
REPL — Interactive Read-Eval-Print Loop

Component: Main CLI Interface
Purpose: Provides interactive shell for NARE queries
Architecture: Thin UI layer over NARE (Neural Amortized Reasoning Engine)

Responsibilities:
- Accept user input with auto-completion
- Dispatch slash commands
- Execute queries through NARE
- Display results with proper formatting
- Manage CLI modes (Manual/Research/Autopilot/Focus/Verbose/Interactive)

Dependencies:
- NareSession: NARE agent wrapper
- ModeManager: CLI mode management
- ThinkingDisplay: Real-time token streaming
- Components: Professional UI elements
"""

import os
import time
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console

from nare.cli.session import NareSession
from nare.cli.commands import dispatch, COMMAND_MAP
from nare.cli.display import ui, show_thinking
from nare.cli.display.components import ResultCard, StatusBar
from nare.cli.modes import get_mode_manager, Mode, MODE_DESCRIPTIONS, MODE_SYMBOLS
from nare.cli.autonomous_runner import AutonomousRunner
from nare.cli.interactive import ask_yes_no

_session_for_toolbar: 'NareSession | None' = None

def get_bottom_toolbar():
    """Bottom toolbar with mode + repo + model + key hints.

    Layout:
        Manual  NareCLI  model  Tab: mode  Ctrl+L: clear  Ctrl+D: exit
    """
    mode_manager = get_mode_manager()
    name = mode_manager.current_mode.value

    parts = [f'<style fg="#D77757">{name}</style>']

    if _session_for_toolbar is not None:
        repo = os.path.basename(_session_for_toolbar.repo_path)
        parts.append(f'<style fg="#999999">{repo}</style>')
        try:
            info = _session_for_toolbar.get_status()
            model = info.get("model")
            if model:
                parts.append(f'<style fg="#666666">{model}</style>')
        except Exception:
            pass

    parts.append('<style fg="#505050">Tab·mode  Ctrl+L·clear  Ctrl+D·exit</style>')
    return HTML("  ".join(parts))

class NareCompleter(Completer):
    """Auto-complete slash commands and file paths.

    Features:
    - Completes slash commands with help text
    - Completes file paths after /read command
    - Filters hidden files (starting with .)

    Usage:
        completer = NareCompleter(session)
        prompt_session = PromptSession(completer=completer)
    """

    def __init__(self, session: NareSession):
        """Initialize completer.

        Args:
            session: NareSession instance for file path resolution
        """
        self.session = session

    def get_completions(self, document, complete_event):
        """Generate completions for current input.

        Args:
            document: Current document state
            complete_event: Completion event

        Yields:
            Completion objects
        """
        text = document.text_before_cursor

        if text.startswith("/"):
            word = text.lstrip("/")
            for name in COMMAND_MAP:
                if name.startswith(word):
                    yield Completion(
                        f"/{name}",
                        start_position=-len(text),
                        display_meta=COMMAND_MAP[name].help,
                    )
            return

        if text.startswith("/read "):
            partial = text[6:]
            base = os.path.join(self.session.repo_path, partial)
            parent = os.path.dirname(base) or self.session.repo_path
            prefix = os.path.basename(base)
            if os.path.isdir(parent):
                try:
                    for entry in os.listdir(parent):
                        if entry.startswith(prefix) and not entry.startswith('.'):
                            full = os.path.join(parent, entry)
                            display = entry + "/" if os.path.isdir(full) else entry
                            yield Completion(
                                display,
                                start_position=-len(prefix),
                            )
                except PermissionError:
                    pass

def run_query(session: NareSession, query: str):
    """Execute query through NARE pipeline.

    Pipeline:
    1. Display query with intent badge
    2. Check if autonomous mode needed
    3. Stream thinking process in real-time (if mode allows)
    4. Execute through NARE (routing, planning, tools, memory)
    5. Clean and display result
    6. Show status bar with metrics

    Args:
        session: NareSession instance
        query: User query string
    """
    mode_manager = get_mode_manager()
    mode_config = mode_manager.get_config()
    console = Console()

    console.print()

    added_files = []
    common_exts = ('.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.md', '.txt', '.json', '.yaml', '.yml', '.rs', '.go', '.c', '.cpp', '.h', '.java', '.sh')

    for token in query.split():

        word = token.strip('",\'.;:()[]{}?!<>`')
        if word and len(word) > 2 and ('.' in word or '/' in word or '\\' in word):

            if any(word.endswith(ext) for ext in common_exts) or '/' in word:

                if not any(word in path for path in session.context_files):
                    content = session.read_file(word)
                    if content:
                        # Show UI block for auto-loaded file
                        from nare.cli.display import blocks
                        num_lines = content.count('\n') + 1 if content else 0
                        blocks.render_read(console, word, num_lines=num_lines)
                        console.print()
                        added_files.append(word)

    if added_files:
        console.print(f"[#666666]Auto-context loaded:[/] [#00FFFF]{', '.join(added_files)}[/]")
        console.print()

        # Add file contents to query so model can reference them
        file_contents = []
        for file_path in added_files:
            content = session.context_files.get(file_path, "")
            if content:
                # Show first 2000 chars
                preview = content[:2000]
                if len(content) > 2000:
                    preview += f"\n\n... ({len(content) - 2000} more characters)"
                file_contents.append(f"\n## File: {file_path}\n```\n{preview}\n```")

        if file_contents:
            query = f"{query}\n\n[Context: The following files are loaded:]\n{''.join(file_contents)}"

    autonomous = AutonomousRunner(session)
    if autonomous.should_run_autonomously(query):
        console.print()
        console.print(f"  [#FFA500]◆ This looks like a multi-step task[/]")
        console.print()

        user_choice = ask_yes_no("Work on this autonomously?", default=False)

        if user_choice:
            if mode_config.show_thinking:
                with show_thinking(session=session) as thinking:
                    thinking.start_waiting("Thinking")
                    result = autonomous.run(query, thinking_display=thinking)
                    thinking.stop_waiting()
            else:
                result = autonomous.run(query, thinking_display=None)
        else:
            console.print()
            console.print(f"  [#666666]Continuing in manual mode...[/]")
            console.print()

            if mode_config.show_thinking:
                with show_thinking(session=session) as thinking:
                    thinking.start_waiting("Thinking")
                    result = session.solve(query, thinking_display=thinking)
                    thinking.stop_waiting()
            else:
                from nare.cli.display.spinner import WaitingSpinner
                with WaitingSpinner("Processing", delay=0.15, color="bright_yellow"):
                    result = session.solve(query, thinking_display=None)
    else:

        if mode_config.show_thinking:
            with show_thinking(session=session) as thinking:

                thinking.start_waiting("Thinking")
                result = session.solve(query, thinking_display=thinking)
                thinking.stop_waiting()
        else:

            from nare.cli.display.spinner import WaitingSpinner
            with WaitingSpinner("Processing", delay=0.15, color="bright_yellow"):
                result = session.solve(query, thinking_display=None)

    elapsed = result.get("_elapsed", 0)
    console.print()

    answer = result.get("final_answer") or result.get("best_solution") or "No answer."

    import re
    import logging
    logging.info(f"[REPL] Raw answer before filtering (first 500 chars): {answer[:500]}")

    # Split by lines and filter out lines containing <tool_call> tags
    lines = answer.split('\n')
    filtered_lines = []
    skip_until_close = False

    for i, line in enumerate(lines):
        if '<tool_call' in line:
            skip_until_close = True
            # Remove previous line ONLY if it's a short prefix (< 20 chars and looks like a tag prefix)
            if filtered_lines:
                prev = filtered_lines[-1].strip()
                if len(prev) < 20 and prev.endswith('>'):
                    filtered_lines.pop()
        if skip_until_close:
            if '</tool_call' in line:
                skip_until_close = False
            continue
        filtered_lines.append(line)

    answer = '\n'.join(filtered_lines)

    # Remove other XML tags
    answer = re.sub(r'<reasoning>.*?</reasoning>', '', answer, flags=re.DOTALL)
    answer = re.sub(r'<abstract_signature>.*?</abstract_signature>', '', answer, flags=re.DOTALL)
    answer = re.sub(r'<delta_reasoning>.*?</delta_reasoning>', '', answer, flags=re.DOTALL)
    answer = re.sub(r'</?solution>', '', answer)
    answer = re.sub(r'</?final_answer>', '', answer)
    answer = re.sub(
        r'\{\s*"name"\s*:\s*"(?:create_file|edit_file|read_file|list_files|list_dir|write_file)"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\}',
        '', answer
    )
    answer = re.sub(r'<[^>]+>', '', answer)
    answer = re.sub(r'\n{3,}', '\n\n', answer).strip()

    route = result.get("route_decision", "FAST")
    tokens_in = result.get("tokens_in", 0)
    tokens_out = result.get("tokens_out", 0)
    total_tokens = tokens_in + tokens_out

    was_streamed = mode_config.show_thinking and route in (
        "FAST", "SLOW", "HYBRID", "SLOW-RETRY", "SLOW-PATH-FIX", "DIRECT", "AGENT",
    )

    ResultCard.render(console, answer, route, elapsed, total_tokens, streamed=was_streamed)

    if getattr(mode_config, 'auto_commit', False):
        import subprocess
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain"], 
                cwd=session.repo_path, 
                capture_output=True, 
                text=True
            )
            if res.stdout.strip():

                msg_base = query.split('\n')[0].strip()
                if len(msg_base) > 50:
                    msg_base = msg_base[:47] + "..."
                msg = f"NARE({route}): {msg_base}"

                commit_hash = session.git_commit(msg, ["."])
                if commit_hash:
                    console.print(f"  [#666666]└─[/] [#00FF00]Auto-committed changes:[/] {commit_hash[:7]}", style="dim")
                    console.print()
        except Exception as e:
            ui.print_warning(f"Auto-commit failed: {e}")

    session_queries = getattr(session, '_query_count', 0) + 1
    session._query_count = session_queries

    session._total_tokens_in = getattr(session, '_total_tokens_in', 0) + tokens_in
    session._total_tokens_out = getattr(session, '_total_tokens_out', 0) + tokens_out

    session_start = getattr(session, '_start_time', time.time())
    if not hasattr(session, '_start_time'):
        session._start_time = session_start

    if route != "AGENT":
        # Show tokens line before status bar (like in agent loop)
        if total_tokens > 0:
            from rich.text import Text
            from nare.cli.display import blocks

            if total_tokens >= 1000:
                token_str = f"{total_tokens / 1000:.1f}k".replace(".0k", "k")
            else:
                token_str = str(total_tokens)

            token_line = Text()
            token_line.append("  ● ", style=blocks.ACCENT)
            token_line.append(token_str + " tokens", style=blocks.TEXT_MUTED)
            token_line.append("  ·  ", style=blocks.TEXT_FAINT)
            token_line.append(f"{elapsed:.1f}s", style=blocks.TEXT)
            console.print(token_line)
            console.print()

        info = session.get_status()
        StatusBar.render(
            console, route, elapsed, tokens_in, tokens_out,
            info.get("episodes", 0), info.get("skills", 0),
        )

def repl(session: NareSession):
    """Main interactive loop.

    Features:
    - Command history with file persistence
    - Auto-completion for commands and paths
    - Mode cycling with Tab key
    - Bottom toolbar showing current mode
    - Graceful exit on Ctrl+D or /exit

    Args:
        session: NareSession instance
    """
    console = Console()

    session._query_count = 0
    session._start_time = time.time()

    import os
    global _session_for_toolbar
    _session_for_toolbar = session
    mode_manager = get_mode_manager()
    ui.print_banner(repo_path=session.repo_path, mode=mode_manager.current_mode.value)

    history_file = os.path.join(session.repo_path, ".nare_history")

    kb = KeyBindings()

    @kb.add('tab')
    def _(event):
        """Cycle mode on Tab key."""
        mode_manager = get_mode_manager()
        mode_manager.cycle_mode()

    @kb.add('c-l')
    def _(event):
        """Clear screen on Ctrl+L."""
        ui.console.clear()

    try:
        from prompt_toolkit.styles import Style

        toolbar_style = Style.from_dict({
            'bottom-toolbar': 'noreverse',
        })

        prompt_session = PromptSession(
            history=FileHistory(history_file),
            completer=NareCompleter(session),
            complete_while_typing=False,
            bottom_toolbar=get_bottom_toolbar,
            key_bindings=kb,
            style=toolbar_style,
        )
    except Exception:
        prompt_session = PromptSession(
            completer=NareCompleter(session),
            complete_while_typing=False,
            bottom_toolbar=get_bottom_toolbar,
            key_bindings=kb,
        )

    while True:
        try:
            raw = prompt_session.prompt(HTML('<style fg="#EB9F7F">></style> ')).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        if not raw:
            continue

        if raw.startswith("/"):
            action = dispatch(session, raw)
            if action == "exit":
                console.print("Goodbye.")
                break
            continue

        try:
            run_query(session, raw)
        except KeyboardInterrupt:
            ui.print_warning("Interrupted")
        except Exception as e:
            ui.print_error(str(e))
