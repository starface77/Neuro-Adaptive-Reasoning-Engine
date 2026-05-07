"""Autonomy levels for NARE agent control."""

from enum import Enum


class AutonomyLevel(Enum):
    """Agent autonomy levels - how much freedom agent has to act."""

    # Level 1: Full control - agent asks permission for everything
    SUPERVISED = "supervised"

    # Level 2: Balanced - agent reads freely, asks before edits
    ASSISTED = "assisted"

    # Level 3: Full autonomy - agent acts freely (risky)
    AUTONOMOUS = "autonomous"


AUTONOMY_DESCRIPTIONS = {
    AutonomyLevel.SUPERVISED: "Agent asks permission before reading files or making changes (safest)",
    AutonomyLevel.ASSISTED: "Agent reads freely, asks permission before edits (balanced)",
    AutonomyLevel.AUTONOMOUS: "Agent acts freely without asking (fastest, risky)",
}


def should_ask_permission(level: AutonomyLevel, action_type: str) -> bool:
    """Check if agent should ask permission for this action.

    Args:
        level: Current autonomy level
        action_type: Type of action ('read', 'edit', 'bash', etc)

    Returns:
        True if should ask permission, False if can proceed
    """
    if level == AutonomyLevel.AUTONOMOUS:
        # Full autonomy - never ask
        return False

    if level == AutonomyLevel.ASSISTED:
        # Ask only for destructive actions
        return action_type in ('edit_file', 'write_file', 'bash', 'apply_hunks')

    if level == AutonomyLevel.SUPERVISED:
        # Ask for everything
        return True

    return True  # Default to safe
