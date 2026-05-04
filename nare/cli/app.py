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


def main():
    # Fix Windows encoding and disable buffering
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        except Exception:
            pass

    # Disable output buffering for real-time streaming
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

    # Configure logging based on debug flag
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        # Silent mode - only show critical errors
        logging.basicConfig(
            level=logging.CRITICAL,
            format="%(message)s",
        )

    # Benchmark mode
    if args.bench is not None:
        from nare.cli.commands import BenchCommand
        BenchCommand().execute(NareSession(args.repo), str(args.bench))
        return

    session = NareSession(repo_path=args.repo)

    # One-shot mode
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
                # Show full traceback in debug mode
                import traceback
                ui.console.print(f"[error]✖ Error:[/] {e}")
                ui.console.print(traceback.format_exc())
            else:
                # Clean error message in normal mode
                ui.print_error(str(e))
            sys.exit(1)
        return

    # Interactive REPL
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
