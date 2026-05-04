"""Autonomy modes for the NARE CLI.

Three levels of agent freedom: ask before every action, autonomous
read-only research, and full autopilot. The CLI cycles through these
with a hotkey and uses :func:`should_ask_permission` to decide whether
to prompt the user before each action.
"""

from enum import Enum

class AutonomyMode(Enum):
    """Autonomy levels for NARE."""

    MANUAL = "manual"

    DEEP_RESEARCH = "deep_research"

    AUTOPILOT = "autopilot"

MODE_DESCRIPTIONS = {
    AutonomyMode.MANUAL: "Ask before every action",
    AutonomyMode.DEEP_RESEARCH: "Autonomous reading & analysis",
    AutonomyMode.AUTOPILOT: "Fully autonomous (create, edit, run)",
}

MODE_COLORS = {
    AutonomyMode.MANUAL: "warning",
    AutonomyMode.DEEP_RESEARCH: "info",
    AutonomyMode.AUTOPILOT: "success",
}

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

        return True

    if mode == AutonomyMode.DEEP_RESEARCH:

        if action_type in ("read", "list"):
            return False
        return True

    if mode == AutonomyMode.AUTOPILOT:

        return False

    return True
