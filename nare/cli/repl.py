
import os
import re
import time
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.text import Text

from nare.cli.session import NareSession
from nare.cli.commands import dispatch, COMMAND_MAP
from nare.cli.display import ui, show_thinking
from nare.cli.display import blocks
from nare.cli.modes import get_mode_manager, Mode, MODE_DESCRIPTIONS, MODE_SYMBOLS
from nare.cli.autonomous_runner import AutonomousRunner
from nare.cli.interactive import ask_yes_no

_session_for_toolbar: 'NareSession | None' = None

def get_bottom_toolbar():
    mode_manager = get_mode_manager()
    name = mode_manager.current_mode.value

    parts = [f'<style fg="#D77757">◆ {name}</style>']

    if _session_for_toolbar is not None:
        repo = os.path.basename(_session_for_toolbar.repo_path)
        parts.append(f'<style fg="#999999">{repo}</style>')
        try:
            info = _session_for_toolbar.get_status()
            model = info.get("model")
            if model:
                parts.append(f'<style fg="#666666">{model}</style>')
            if info.get("agent_ready"):
                ep = info.get("episodes", 0)
                sk = info.get("skills", 0)
                if ep > 0 or sk > 0:
                    parts.append(f'<style fg="#555555">{ep} ep · {sk} sk</style>')
        except Exception:
            pass

    parts.append('<style fg="#444444">Tab mode  ·  Ctrl+L clear  ·  Ctrl+D exit</style>')
    return HTML("  ".join(parts))

class NareCompleter(Completer):

    def __init__(self, session: NareSession):
        self.session = session

    def get_completions(self, document, complete_event):
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

def _clean_answer(answer):
    lines = answer.split('\n')
    filtered_lines = []
    skip_until_close = False

    for line in lines:
        if '<tool_call' in line:
            skip_until_close = True
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
    return answer


ROUTE_DESCRIPTIONS = {
    "FAST": "cached from episodic memory",
    "COMPILED_SKILL": "executed compiled skill",
    "REFLEX": "matched semantic rule",
    "HYBRID": "delta reasoning with memory context",
    "SLOW": "verified synthesis (best-of-N)",
    "SLOW-RETRY": "verified synthesis retry",
    "SLOW-PATH-FIX": "verified synthesis with path fix",
    "DIRECT": "direct response",
    "AGENT": "autonomous agent loop",
}

def _render_status(console, route, elapsed, tokens_in, tokens_out, result=None, session=None):
    total_tokens = tokens_in + tokens_out
    color = blocks.ROUTE_PALETTE.get(route, blocks.TEXT_MUTED)
    icon = blocks.ROUTE_ICONS.get(route, "◆")

    line = Text()
    line.append(f"  {icon} ", style=f"bold {color}")
    line.append(route, style=f"bold {color}")
    line.append("  ", style="")
    line.append(f"{elapsed:.1f}s", style=blocks.TEXT_MUTED)

    if total_tokens > 0:
        if total_tokens >= 1000:
            token_str = f"{total_tokens / 1000:.1f}k".replace(".0k", "k")
        else:
            token_str = str(total_tokens)
        line.append("  ·  ", style=blocks.TEXT_FAINT)
        line.append(f"{token_str} tokens", style=blocks.TEXT_MUTED)

    # Show amortization ratio if available
    if result:
        alpha = result.get("amortization_ratio", 0)
        if alpha > 0:
            line.append("  ·  ", style=blocks.TEXT_FAINT)
            line.append(f"α={alpha:.0%}", style="#5599FF")

    console.print(line)

    # Show route description
    desc = ROUTE_DESCRIPTIONS.get(route)
    if desc:
        console.print(f"  [#555555]  {desc}[/]")

    # Show memory context info
    if session and hasattr(session, 'agent') and session.agent:
        try:
            mem = session.agent.memory
            ep_count = len(mem.episodes)
            sk_count = len(mem.compiled_skills)
            if ep_count > 0 or sk_count > 0:
                mem_line = Text()
                mem_line.append("    ", style="")
                mem_line.append(f"{ep_count}", style="#5599FF")
                mem_line.append(" episodes", style="#555555")
                if sk_count > 0:
                    mem_line.append("  ·  ", style="#444444")
                    mem_line.append(f"{sk_count}", style="#4EBA65")
                    mem_line.append(" skills", style="#555555")
                console.print(mem_line)
        except Exception:
            pass


def run_query(session: NareSession, query: str):
    mode_manager = get_mode_manager()
    mode_config = mode_manager.get_config()
    console = Console()

    console.print()

    added_files = []
    common_exts = ('.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.md',
                   '.txt', '.json', '.yaml', '.yml', '.rs', '.go', '.c', '.cpp',
                   '.h', '.java', '.sh')

    for token in query.split():
        word = token.strip(',"\'\'.;:()[]{}?!<>`')
        if word and len(word) > 2 and ('.' in word or '/' in word or '\\' in word):
            if any(word.endswith(ext) for ext in common_exts) or '/' in word:
                if not any(word in path for path in session.context_files):
                    content = session.read_file(word)
                    if content:
                        num_lines = content.count('\n') + 1 if content else 0
                        blocks.render_read(console, word, num_lines=num_lines)
                        console.print()
                        added_files.append(word)

    if added_files:
        console.print(f"[#666666]Context:[/] [#999999]{', '.join(added_files)}[/]")
        console.print()

        file_contents = []
        for file_path in added_files:
            content = session.context_files.get(file_path, "")
            if content:
                preview = content[:2000]
                if len(content) > 2000:
                    preview += f"\n\n... ({len(content) - 2000} more characters)"
                file_contents.append(f"\n## File: {file_path}\n```\n{preview}\n```")

        if file_contents:
            query = f"{query}\n\n[Context: The following files are loaded:]\n{''.join(file_contents)}"

    autonomous = AutonomousRunner(session)
    use_autonomous = False

    if autonomous.should_run_autonomously(query):
        console.print()
        console.print("  [#FFC107]◈[/]  [bold white]Multi-step task detected[/]")
        console.print()

        user_choice = ask_yes_no("Work on this autonomously?", default=False)
        if user_choice:
            use_autonomous = True
        else:
            console.print()
            console.print("  [#666666]Continuing in manual mode...[/]")
            console.print()

    with show_thinking(session=session) as thinking:
        thinking.start_waiting("Thinking")

        if use_autonomous:
            result = autonomous.run(query, thinking_display=thinking)
        else:
            result = session.solve(query, thinking_display=thinking)

        thinking.stop_waiting()

    elapsed = result.get("_elapsed", 0)
    console.print()

    answer = result.get("final_answer") or result.get("best_solution") or "No answer."
    answer = _clean_answer(answer)

    route = result.get("route_decision", "FAST")
    tokens_in = result.get("tokens_in", 0)
    tokens_out = result.get("tokens_out", 0)

    was_streamed = route in (
        "FAST", "SLOW", "HYBRID", "SLOW-RETRY", "SLOW-PATH-FIX", "DIRECT", "AGENT",
    )

    if not was_streamed and answer and answer != "No answer.":
        console.print(answer, style="white")
        console.print()

    _render_status(console, route, elapsed, tokens_in, tokens_out, result=result, session=session)

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
                    console.print(f"  [#666666]└─[/] [#00FF00]Committed:[/] {commit_hash[:7]}", style="dim")
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

def repl(session: NareSession):
    console = Console()

    session._query_count = 0
    session._start_time = time.time()

    global _session_for_toolbar
    _session_for_toolbar = session
    mode_manager = get_mode_manager()
    ui.print_banner(repo_path=session.repo_path, mode=mode_manager.current_mode.value)

    history_file = os.path.join(session.repo_path, ".nare_history")

    kb = KeyBindings()

    @kb.add('tab')
    def _(event):
        mode_manager = get_mode_manager()
        mode_manager.cycle_mode()

    @kb.add('c-l')
    def _(event):
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
            raw = prompt_session.prompt(HTML('<style fg="#D77757">◆</style> ')).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            console.print("  [#D77757]◆[/] [#999999]Goodbye. Session ended.[/]")
            console.print()
            break

        if not raw:
            continue

        if raw.startswith("/"):
            action = dispatch(session, raw)
            if action == "exit":
                console.print()
                console.print("  [#D77757]◆[/] [#999999]Goodbye. Session ended.[/]")
                console.print()
                break
            continue

        try:
            run_query(session, raw)
        except KeyboardInterrupt:
            ui.print_warning("Interrupted")
        except Exception as e:
            ui.print_error(str(e))
