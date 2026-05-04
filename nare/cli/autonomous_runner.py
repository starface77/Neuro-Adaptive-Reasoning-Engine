"""Autonomous execution integrated into main REPL.

Component: Autonomous Runner
Purpose: Detect when to work autonomously and ask user interactively
Architecture: Integrated into session.solve(), not separate CLI

Key features:
- Agent detects multi-step tasks automatically
- Asks user via interactive dialogs (arrow keys + Enter)
- Works until task complete or user stops
- No separate /autonomous command needed
"""

import logging
import time
from typing import Optional, Dict, Any

from nare.cli.interactive import (
    ask_yes_no,
    ask_autonomous_action,
    ask_continue_after_error,
    ask_task_complete_action
)
from nare.cli.display import ui

log = logging.getLogger("nare.cli.autonomous")

class AutonomousRunner:
    """Manages autonomous execution within main REPL."""

    def __init__(self, session):
        """Initialize autonomous runner.

        Args:
            session: NareSession instance
        """
        self.session = session
        self.is_running = False
        self.max_iterations = 50
        self.max_errors = 5
        self.error_count = 0

    def should_run_autonomously(self, query: str) -> bool:
        """Detect if query needs autonomous execution.

        Indicators:
        - Multiple steps mentioned ("first", "then", "after")
        - Long-running tasks ("refactor all", "implement", "create")
        - Explicit request ("do this automatically", "work on this")

        Args:
            query: User query

        Returns:
            True if should run autonomously
        """
        query_lower = query.lower()

        multi_step_words = [
            "first", "then", "after", "next", "finally",
            "step 1", "step 2", "1.", "2.", "3."
        ]
        if any(word in query_lower for word in multi_step_words):
            return True

        long_tasks = [
            "refactor all", "implement", "create", "build",
            "fix all", "update all", "migrate", "convert all"
        ]
        if any(task in query_lower for task in long_tasks):
            return True

        autonomous_words = [
            "automatically", "autonomously", "work on this",
            "do this for me", "complete this", "finish this"
        ]
        if any(word in query_lower for word in autonomous_words):
            return True

        return False

    def run(self, initial_query: str, thinking_display=None) -> Dict[str, Any]:
        """Run autonomously until task complete.

        Flow:
        1. Execute initial query
        2. Check if more work needed
        3. Ask user what to do (continue/pause/stop)
        4. If continue - extract next task and repeat
        5. If error - ask user (retry/skip/stop)

        Args:
            initial_query: Initial user query
            thinking_display: Optional thinking display

        Returns:
            Final result dict
        """
        self.is_running = True
        self.error_count = 0
        current_query = initial_query
        iteration = 0
        last_result = None

        ui.console.print()
        ui.console.print("  [#00FFFF]◆ Autonomous mode activated[/]")
        ui.console.print(f"  [#666666]Press Ctrl+C anytime to pause[/]")
        ui.console.print()

        while self.is_running and iteration < self.max_iterations:
            iteration += 1

            try:

                result = self.session.solve(
                    query=current_query,
                    thinking_display=thinking_display
                )
                last_result = result

                self.error_count = 0

                if self._is_task_complete(result):
                    ui.console.print()
                    ui.console.print("  [#00FF00]✓ Task completed successfully[/]")
                    ui.console.print()

                    action = ask_task_complete_action()

                    if action == "done":
                        break
                    elif action == "review":

                        self.is_running = False
                        break
                    elif action == "next":

                        next_task = self._extract_next_task(result)
                        if next_task:
                            current_query = next_task
                            continue
                        else:
                            ui.console.print("  [#666666]No more tasks found[/]")
                            break

                next_task = self._extract_next_task(result)
                if next_task:
                    ui.console.print()
                    action = ask_autonomous_action(next_task)

                    if action == "continue":
                        current_query = next_task
                        continue
                    elif action == "pause":
                        ui.console.print("  [#FFA500]⊙ Paused - returning to prompt[/]")
                        self.is_running = False
                        break
                    elif action == "stop":
                        ui.console.print("  [#FF0000]⊘ Stopped by user[/]")
                        break
                else:

                    break

            except KeyboardInterrupt:
                ui.console.print()
                ui.console.print("  [#FFA500]⊙ Paused by user (Ctrl+C)[/]")

                if ask_yes_no("Resume autonomous mode?", default=False):
                    continue
                else:
                    break

            except Exception as e:
                log.error(f"[Autonomous] Error: {e}")
                self.error_count += 1

                ui.console.print()
                action = ask_continue_after_error(str(e), self.error_count)

                if action == "retry":
                    if self.error_count >= self.max_errors:
                        ui.console.print(f"  [#FF0000]Too many errors ({self.max_errors}), stopping[/]")
                        break

                    time.sleep(2)
                    continue
                elif action == "skip":

                    if last_result:
                        next_task = self._extract_next_task(last_result)
                        if next_task:
                            current_query = next_task
                            continue
                    break
                elif action == "stop":
                    break

        self.is_running = False
        ui.console.print()
        ui.console.print(f"  [#666666]Completed {iteration} iteration(s)[/]")
        ui.console.print()

        return last_result or {}

    def _is_task_complete(self, result: Dict[str, Any]) -> bool:
        """Check if task is complete.

        Args:
            result: Result from solve()

        Returns:
            True if task is done
        """
        answer = result.get("final_answer", "").lower()

        completion_phrases = [
            "task completed",
            "all done",
            "finished",
            "no more work",
            "nothing left to do",
            "successfully completed",
            "implementation complete"
        ]

        return any(phrase in answer for phrase in completion_phrases)

    def _extract_next_task(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract next task from result.

        Looks for:
        - "Next: ..." in answer
        - "TODO: ..." in answer
        - "Next step: ..." in answer

        Args:
            result: Result from solve()

        Returns:
            Next task description or None
        """
        answer = result.get("final_answer", "")

        markers = ["next:", "todo:", "next step:", "next task:"]

        for marker in markers:
            if marker in answer.lower():
                lines = answer.split("\n")
                for line in lines:
                    if marker in line.lower():

                        parts = line.lower().split(marker, 1)
                        if len(parts) > 1:
                            task = parts[1].strip()

                            task = task.lstrip("- *#").strip()
                            if task and len(task) > 10:
                                return task

        return None

    def stop(self):
        """Stop autonomous execution."""
        self.is_running = False
