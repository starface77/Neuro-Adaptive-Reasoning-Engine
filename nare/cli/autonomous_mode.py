"""Autonomous mode for NARE CLI.

Allows agent to work continuously without user interaction.
"""

import logging
import time
from typing import Optional

log = logging.getLogger("nare.cli.autonomous")

class AutonomousMode:
    """Autonomous execution mode for long-running tasks."""

    def __init__(self, session):
        self.session = session
        self.running = False
        self.current_task = None
        self.max_iterations = 50
        self.max_errors = 5
        self.error_count = 0

    def start(self, task: str, max_iterations: Optional[int] = None):
        """Start autonomous execution.

        Args:
            task: Initial task description
            max_iterations: Maximum iterations (default: 50)
        """
        if max_iterations:
            self.max_iterations = max_iterations

        self.running = True
        self.error_count = 0
        self.current_task = task

        log.info(f"[Autonomous] Starting: {task}")
        print(f"\n🤖 Autonomous mode started")
        print(f"Task: {task}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Press Ctrl+C to stop\n")

        iteration = 0

        while self.running and iteration < self.max_iterations:
            iteration += 1

            try:
                print(f"\n[Iteration {iteration}/{self.max_iterations}]")

                result = self.session.solve(
                    query=self.current_task,
                    thinking_display=None
                )

                if result.get("_stop_reason") == "success":
                    print(f"✓ Task completed successfully")
                    print(f"Result: {result.get('final_answer', '')[:200]}")

                    self.error_count = 0

                    if self._is_task_complete(result):
                        print(f"\n✓ All work completed")
                        break

                    self.current_task = self._get_next_task(result)
                    if not self.current_task:
                        print(f"\n✓ No more tasks")
                        break

                else:
                    print(f"⚠ Task incomplete: {result.get('_stop_reason')}")
                    self.error_count += 1

                    if self.error_count >= self.max_errors:
                        print(f"\n✗ Too many errors ({self.max_errors}), stopping")
                        break

                time.sleep(2)

            except KeyboardInterrupt:
                print(f"\n\n⊘ Stopped by user")
                break

            except Exception as e:
                log.error(f"[Autonomous] Error: {e}")
                print(f"✗ Error: {e}")
                self.error_count += 1

                if self.error_count >= self.max_errors:
                    print(f"\n✗ Too many errors ({self.max_errors}), stopping")
                    break

                time.sleep(5)

        self.running = False
        print(f"\n🤖 Autonomous mode stopped")
        print(f"Completed {iteration} iterations")

    def stop(self):
        """Stop autonomous execution."""
        self.running = False

    def _is_task_complete(self, result: dict) -> bool:
        """Check if task is complete.

        Returns:
            True if task is done, False if more work needed
        """
        answer = result.get("final_answer", "").lower()

        completion_phrases = [
            "task completed",
            "all done",
            "finished",
            "no more work",
            "nothing left to do",
        ]

        return any(phrase in answer for phrase in completion_phrases)

    def _get_next_task(self, result: dict) -> Optional[str]:
        """Extract next task from result.

        Returns:
            Next task description or None
        """
        answer = result.get("final_answer", "")

        if "next:" in answer.lower():
            lines = answer.split("\n")
            for line in lines:
                if "next:" in line.lower():
                    return line.split("next:", 1)[1].strip()

        if "todo:" in answer.lower():
            lines = answer.split("\n")
            for line in lines:
                if "todo:" in line.lower():
                    return line.split("todo:", 1)[1].strip()

        return None
