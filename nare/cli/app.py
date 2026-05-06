"""
NARE CLI — Entry point.

Usage:
    python -m nare.cli                     Interactive REPL
    python -m nare.cli "fix the bug"       One-shot solve
    python -m nare.cli --repo ./project    Set repo
    python -m nare.cli --bench 10          SWE-bench
"""

import sys
import os
import argparse
import logging

from nare.cli.session import NareSession
from nare.cli.repl import repl, run_query
from nare.cli.display import ui

# Detect if running in Git Bash / MinTTY on Windows
def is_mintty():
    return sys.platform == 'win32' and os.environ.get('TERM') == 'xterm-256color'

def main():

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        except Exception:
            pass

    import os
    os.environ['PYTHONUNBUFFERED'] = '1'

    parser = argparse.ArgumentParser(
        description="NARE - Neural Amortized Reasoning Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m nare.cli                   Interactive REPL\n"
            '  python -m nare.cli "fix auth bug"    One-shot solve\n'
            "  python -m nare.cli --repo ./django   Set project root\n"
            "  python -m nare.cli --bench 10        Run SWE-bench\n"
        ),
    )
    parser.add_argument("query", nargs="?", default=None, help="One-shot query")
    parser.add_argument("--repo", "-r", default=".", help="Repository path")
    parser.add_argument("--bench", "-b", type=int, default=None, help="Run SWE-bench with N tasks")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode (verbose logs and tracebacks)")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    else:

        logging.basicConfig(
            level=logging.CRITICAL,
            format="%(message)s",
        )

    if args.bench is not None:
        from nare.cli.commands import BenchCommand
        BenchCommand().execute(NareSession(args.repo), str(args.bench))
        return

    session = NareSession(repo_path=args.repo)

    # Check API key on first use
    from nare.config.api_keys import ensure_api_key
    api_key = ensure_api_key(provider="anthropic")
    if not api_key:
        ui.print_error("No API key provided. NARE cannot function without an API key.")
        sys.exit(1)

    if args.query:
        ui.print_banner()
        ui.print_status("repo", session.repo_path, "info")
        ui.print_status("query", args.query, "muted")
        ui.console.print()
        try:
            run_query(session, args.query)
        except KeyboardInterrupt:
            ui.print_warning("Interrupted")
            sys.exit(130)
        except Exception as e:
            if args.debug:

                import traceback
                ui.console.print(f"[error]✖ Error:[/] {e}")
                ui.console.print(traceback.format_exc())
            else:

                ui.print_error(str(e))
            sys.exit(1)
        return

    # Block interactive mode in Git Bash
    if is_mintty():
        print("Error: Interactive mode doesn't work in Git Bash on Windows.")
        print("Please use one of these instead:")
        print("  1. Windows Terminal (recommended)")
        print("  2. CMD.exe")
        print("  3. PowerShell")
        print("\nOr use one-shot mode: python -m nare.cli \"your query\"")
        sys.exit(1)

    try:
        repl(session)
    except KeyboardInterrupt:
        ui.console.print("\n[text_muted]Goodbye.[/]")
        sys.exit(0)
    except Exception as e:
        if args.debug:
            import traceback
            ui.console.print(f"[error]✖ Fatal error:[/] {e}")
            ui.console.print(traceback.format_exc())
        else:
            ui.print_error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
