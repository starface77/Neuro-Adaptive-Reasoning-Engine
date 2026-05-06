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

from nare.utils.logger import get_logger
import time
from typing import Optional, Dict, Any

from nare.cli.interactive import (
    ask_yes_no,
    ask_autonomous_action,
    ask_continue_after_error,
    ask_task_complete_action
)
from nare.cli.display import ui

log = get_logger("nare.cli.autonomous")

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
        self.max_errors = 3
        self.error_count = 0

    def should_run_autonomously(self, query: str) -> bool:
        """Detect if query needs autonomous execution.

        Uses LLM to understand user intent for autonomous mode.

        Args:
            query: User query

        Returns:
            True if should run autonomously
        """
        from nare.reasoning import llm

        # Quick heuristic check first (fast path)
        query_lower = query.lower()

        # Explicit autonomous keywords
        explicit_keywords = [
            "автономн", "автоматическ", "работай сам", "сделай сам",
            "automatically", "autonomously", "work on this"
        ]
        if any(kw in query_lower for kw in explicit_keywords):
            return True

        # Multi-step indicators
        multi_step_words = [
            "first", "then", "after", "next", "finally",
            "step 1", "step 2", "1.", "2.", "3.",
            "сначала", "потом", "затем", "далее"
        ]
        if any(word in query_lower for word in multi_step_words):
            return True

        # Long tasks
        long_tasks = [
            "refactor all", "implement", "create", "build",
            "fix all", "update all", "migrate", "convert all",
            "исправь все", "реализуй", "создай", "построй"
        ]
        if any(task in query_lower for task in long_tasks):
            return True

        # LLM-based intent detection (slow path, only for ambiguous cases)
        # Skip if query is very short (likely not autonomous)
        if len(query.split()) < 5:
            return False

        try:
            prompt = f"""Analyze if this user request requires autonomous multi-step execution.

User request: "{query}"

Answer with ONLY "yes" or "no".

Answer "yes" if:
- User explicitly asks for autonomous/automatic work
- Task requires multiple sequential steps
- Task is complex and will take multiple iterations
- User wants you to work independently on something

Answer "no" if:
- Simple question or single action
- User wants to guide each step
- Conversational query

Answer:"""

            response = llm.generate_samples(prompt, n=1, temperature=0.0, mode="DIRECT")
            if response and len(response) > 0:
                answer = response[0].get('text', '').strip().lower()
                return 'yes' in answer
        except Exception as e:
            log.warning(f"[AutonomousRunner] Failed to check if more work needed: {e}")

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
        ui.console.print("  [#D77757]◆ Autonomous mode activated[/]")
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

                # Check if task is complete
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

                # Task not complete - check if there's more work
                next_task = self._extract_next_task(result)

                # If no explicit next task found, but answer suggests continuation, keep going
                if not next_task:
                    answer = result.get("final_answer", "").lower()

                    # First check if LLM wants to continue
                    continuation_phrases = [
                        "начинаю", "продолжаю", "далее", "теперь", "следующий шаг", "сейчас",
                        "starting", "continuing", "next", "now", "proceeding", "first", "firstly"
                    ]
                    wants_to_continue = any(phrase in answer for phrase in continuation_phrases)

                    # Check for hallucinations
                    hallucination_indicators = [
                        "backend/", "frontend/", "server.js", "app.jsx", "database.js",
                        "auth.js", "users.js", "config.js", "middleware/", "routes.js"
                    ]
                    is_hallucinating = any(indicator in answer for indicator in hallucination_indicators)

                    if is_hallucinating:
                        log.warning("[Autonomous] Hallucination detected - skipping iteration")
                        # Skip hallucinated response but continue if LLM wants to
                        if wants_to_continue:
                            time.sleep(1)
                            continue
                        else:
                            break

                    if wants_to_continue:
                        # Continue with same query to let LLM proceed
                        ui.console.print()
                        ui.console.print("  [#666666]Continuing work...[/]")
                        time.sleep(1)
                        continue
                    else:
                        # No continuation detected, stop
                        break

                # Found explicit next task - ask user
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

            except KeyboardInterrupt:
                ui.console.print()
                ui.console.print("  [#FFA500]⊙ Paused by user (Ctrl+C)[/]")

                if ask_yes_no("Resume autonomous mode?", default=False):
                    continue
                else:
                    break

            except Exception as e:
                log.error(f"[Autonomous] Error: {e}")
                log.error(f"[Autonomous] Error type: {type(e).__name__}")
                import traceback
                traceback_str = traceback.format_exc()
                log.error(f"[Autonomous] Traceback:\n{traceback_str}")

                # Also print to console for debugging
                print("\n=== DEBUG TRACEBACK ===")
                print(traceback_str)
                print("=== END DEBUG ===\n")

                self.error_count += 1

                if self.error_count >= self.max_errors:
                    ui.console.print()
                    ui.console.print(f"  [#FF0000]Too many errors ({self.max_errors}), stopping autonomous mode[/]")
                    ui.console.print()
                    break

                ui.console.print()
                ui.console.print(f"  [#FF5555]Error:[/] {str(e)[:100]}")
                ui.console.print(f"  [#666666]Retries so far: {self.error_count}[/]")
                ui.console.print()

                action = ask_continue_after_error(str(e), self.error_count)

                if action == "retry":
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
