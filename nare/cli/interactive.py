"""Interactive questions with arrow key selection.

Component: Interactive UI
Purpose: Ask user questions with arrow key navigation
Architecture: Uses prompt_toolkit for rich terminal UI

Features:
- Arrow keys to navigate options
- Enter to select
- Escape to cancel
- Visual highlighting of selected option
"""

from typing import List, Optional
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit.formatted_text import HTML
from rich.console import Console

def ask_choice(
    question: str,
    choices: List[tuple[str, str]],
    default: Optional[str] = None
) -> Optional[str]:
    """Ask user to choose from options using arrow keys.

    Args:
        question: Question to ask
        choices: List of (value, label) tuples
        default: Default selected value

    Returns:
        Selected value or None if cancelled

    Example:
        choice = ask_choice(
            "Continue autonomously?",
            [
                ("yes", "Yes, continue working"),
                ("no", "No, wait for my input"),
                ("stop", "Stop completely")
            ],
            default="yes"
        )
    """
    try:
        result = radiolist_dialog(
            title=question,
            text="Use arrow keys to select, Enter to confirm:",
            values=choices,
            default=default
        ).run()
        return result
    except (KeyboardInterrupt, EOFError):
        return None

def ask_yes_no(question: str, default: bool = True) -> bool:
    """Ask yes/no question with arrow keys.

    Args:
        question: Question to ask
        default: Default answer (True = Yes, False = No)

    Returns:
        True for Yes, False for No

    Example:
        if ask_yes_no("Continue working?"):
            # User said yes
    """
    choices = [
        ("yes", "Yes"),
        ("no", "No")
    ]
    default_val = "yes" if default else "no"

    result = ask_choice(question, choices, default=default_val)
    return result == "yes" if result else default

def ask_autonomous_action(task_description: str) -> str:
    """Ask what to do next in autonomous mode.

    Args:
        task_description: Description of current task

    Returns:
        Action: "continue", "pause", "stop"
    """
    console = Console()
    console.print()
    console.print(f"  [#FFA500]Task:[/] {task_description}", style="#999999")
    console.print()

    choices = [
        ("continue", "Continue working autonomously"),
        ("pause", "Pause and wait for my input"),
        ("stop", "Stop completely")
    ]

    result = ask_choice(
        "What should I do?",
        choices,
        default="continue"
    )

    return result or "pause"

def ask_multi_step_plan(steps: List[str]) -> Optional[str]:
    """Show multi-step plan and ask for confirmation.

    Args:
        steps: List of steps to execute

    Returns:
        Action: "execute", "modify", "cancel"
    """
    console = Console()
    console.print()
    console.print("  [#FFA500]Plan:[/]", style="#999999")
    for i, step in enumerate(steps, 1):
        console.print(f"    {i}. {step}", style="#999999")
    console.print()

    choices = [
        ("execute", "Execute this plan"),
        ("modify", "Let me modify it"),
        ("cancel", "Cancel")
    ]

    result = ask_choice(
        "Approve this plan?",
        choices,
        default="execute"
    )

    return result or "cancel"

def ask_continue_after_error(error: str, retry_count: int) -> str:
    """Ask what to do after an error.

    Args:
        error: Error message
        retry_count: Number of retries so far

    Returns:
        Action: "retry", "skip", "stop"
    """
    console = Console()
    console.print()
    console.print(f"  [#FF0000]Error:[/] {error}", style="#999999")
    console.print(f"  [#666666]Retries so far: {retry_count}[/]")
    console.print()

    choices = [
        ("retry", "Try again with different approach"),
        ("skip", "Skip this step and continue"),
        ("stop", "Stop completely")
    ]

    result = ask_choice(
        "What should I do?",
        choices,
        default="retry" if retry_count < 3 else "skip"
    )

    return result or "stop"

def ask_task_complete_action() -> str:
    """Ask what to do after completing a task.

    Returns:
        Action: "next", "review", "done"
    """
    console = Console()
    console.print()
    console.print("  [#00FF00]✓ Task completed[/]")
    console.print()

    choices = [
        ("next", "Continue with next task"),
        ("review", "Let me review the changes"),
        ("done", "All done, return to prompt")
    ]

    result = ask_choice(
        "What's next?",
        choices,
        default="next"
    )

    return result or "done"
