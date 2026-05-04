"""Autonomy modes for the NARE CLI.

Three levels of agent freedom: ask before every action, autonomous
read-only research, and full autopilot. The CLI cycles through these
with a hotkey and uses :func:`should_ask_permission` to decide whether
to prompt the user before each action.
"""

from enum import Enum


class AutonomyMode(Enum):
    """Autonomy levels for NARE."""

    # Manual mode - ask before every action
    MANUAL = "manual"

    # Deep Research - autonomous file reading and analysis
    DEEP_RESEARCH = "deep_research"

    # Autopilot - fully autonomous (create files, run commands, etc.)
    AUTOPILOT = "autopilot"


# Mode descriptions
MODE_DESCRIPTIONS = {
    AutonomyMode.MANUAL: "Ask before every action",
    AutonomyMode.DEEP_RESEARCH: "Autonomous reading & analysis",
    AutonomyMode.AUTOPILOT: "Fully autonomous (create, edit, run)",
}


# Mode colors
MODE_COLORS = {
    AutonomyMode.MANUAL: "warning",
    AutonomyMode.DEEP_RESEARCH: "info",
    AutonomyMode.AUTOPILOT: "success",
}


# Mode symbols
MODE_SYMBOLS = {
    AutonomyMode.MANUAL: "·",
    AutonomyMode.DEEP_RESEARCH: "◇",
    AutonomyMode.AUTOPILOT: "◆",
}


def get_next_mode(current: AutonomyMode) -> AutonomyMode:
    """Cycle to next autonomy mode."""
    modes = list(AutonomyMode)
    current_idx = modes.index(current)
    next_idx = (current_idx + 1) % len(modes)
    return modes[next_idx]


def should_ask_permission(mode: AutonomyMode, action_type: str) -> bool:
    """Check if we should ask permission for an action.

    Args:
        mode: Current autonomy mode
        action_type: Type of action (read, create, edit, run_command)

    Returns:
        True if should ask, False if can proceed
    """
    if mode == AutonomyMode.MANUAL:
        # Ask for everything
        return True

    if mode == AutonomyMode.DEEP_RESEARCH:
        # Can read freely, ask for writes and commands
        if action_type in ("read", "list"):
            return False
        return True

    if mode == AutonomyMode.AUTOPILOT:
        # Can do everything
        return False

    return True
