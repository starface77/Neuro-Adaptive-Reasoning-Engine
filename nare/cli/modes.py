"""
CLI Modes System
Professional mode management for NARE CLI

Modes:
- Manual: User controls everything, explicit confirmation for actions
- Research: Auto-explore codebase, suggest solutions, ask before executing
- Autopilot: Full autonomy, execute actions automatically
- Focus: Minimal output, only show results
- Verbose: Maximum detail, show all reasoning steps
- Interactive: Step-by-step execution with user feedback

Architecture:
- Mode enum defines available modes
- ModeConfig holds mode-specific settings
- ModeManager handles mode switching and state
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class Mode(Enum):
    """Available CLI modes.

    Each mode defines a different level of autonomy and verbosity.
    """
    MANUAL = "Manual"
    RESEARCH = "Research"
    AUTOPILOT = "Autopilot"
    FOCUS = "Focus"
    VERBOSE = "Verbose"
    INTERACTIVE = "Interactive"

@dataclass
class ModeConfig:
    """Configuration for a specific mode.

    Attributes:
        auto_execute: Execute actions without confirmation
        show_thinking: Display reasoning process
        show_plan: Display execution plan
        show_tools: Display tool calls
        ask_before_edit: Ask before modifying files
        ask_before_create: Ask before creating files
        max_auto_iterations: Max iterations in autopilot
        auto_commit: Automatically commit successful changes
    """
    auto_execute: bool = False
    show_thinking: bool = True
    show_plan: bool = True
    show_tools: bool = True
    ask_before_edit: bool = True
    ask_before_create: bool = True
    max_auto_iterations: int = 5

    auto_commit: bool = False

MODE_CONFIGS = {
    Mode.MANUAL: ModeConfig(
        auto_execute=False,
        show_thinking=True,
        show_plan=True,
        show_tools=True,
        ask_before_edit=True,
        ask_before_create=True,
        max_auto_iterations=1,
    ),
    Mode.RESEARCH: ModeConfig(
        auto_execute=False,
        show_thinking=True,
        show_plan=True,
        show_tools=True,
        ask_before_edit=True,
        ask_before_create=True,
        max_auto_iterations=3,
    ),
    Mode.AUTOPILOT: ModeConfig(
        auto_execute=True,
        show_thinking=True,
        show_plan=False,
        show_tools=False,
        ask_before_edit=False,
        ask_before_create=False,
        max_auto_iterations=10,
    ),
    Mode.FOCUS: ModeConfig(
        auto_execute=True,
        show_thinking=False,
        show_plan=False,
        show_tools=False,
        ask_before_edit=False,
        ask_before_create=False,
        max_auto_iterations=10,
    ),
    Mode.VERBOSE: ModeConfig(
        auto_execute=False,
        show_thinking=True,
        show_plan=True,
        show_tools=True,
        ask_before_edit=True,
        ask_before_create=True,
        max_auto_iterations=5,
    ),
    Mode.INTERACTIVE: ModeConfig(
        auto_execute=False,
        show_thinking=True,
        show_plan=True,
        show_tools=True,
        ask_before_edit=True,
        ask_before_create=True,
        max_auto_iterations=1,
    ),
}

MODE_DESCRIPTIONS = {
    Mode.MANUAL: "Full control - confirm every action",
    Mode.RESEARCH: "Auto-explore, suggest solutions, ask before executing",
    Mode.AUTOPILOT: "Full autonomy - execute automatically",
    Mode.FOCUS: "Minimal output - only show results",
    Mode.VERBOSE: "Maximum detail - show all reasoning",
    Mode.INTERACTIVE: "Step-by-step with feedback",
}

MODE_SYMBOLS = {
    Mode.MANUAL: "",
    Mode.RESEARCH: "",
    Mode.AUTOPILOT: "",
    Mode.FOCUS: "",
    Mode.VERBOSE: "",
    Mode.INTERACTIVE: "",
}

class ModeManager:
    """Manages CLI mode state and transitions.

    Responsibilities:
    - Track current mode
    - Switch between modes
    - Provide mode configuration
    - Cycle through modes (for Tab key)

    Usage:
        manager = ModeManager()
        manager.set_mode(Mode.AUTOPILOT)
        config = manager.get_config()
        if config.auto_execute:
            execute_action()
    """

    def __init__(self, initial_mode: Mode = Mode.AUTOPILOT):
        """Initialize mode manager.

        Args:
            initial_mode: Starting mode (default: AUTOPILOT)
        """
        self._current_mode = initial_mode
        self._mode_history: list[Mode] = [initial_mode]

    @property
    def current_mode(self) -> Mode:
        """Get current mode."""
        return self._current_mode

    def set_mode(self, mode: Mode) -> None:
        """Set current mode.

        Args:
            mode: Mode to switch to
        """
        if mode != self._current_mode:
            self._current_mode = mode
            self._mode_history.append(mode)

    def get_config(self) -> ModeConfig:
        """Get configuration for current mode.

        Returns:
            ModeConfig for current mode
        """
        return MODE_CONFIGS[self._current_mode]

    def cycle_mode(self) -> Mode:
        """Cycle to next mode (for Tab key).

        Order: Manual → Research → Autopilot → Focus → Verbose → Interactive → Manual

        Returns:
            New current mode
        """
        modes = list(Mode)
        current_idx = modes.index(self._current_mode)
        next_idx = (current_idx + 1) % len(modes)
        self._current_mode = modes[next_idx]
        self._mode_history.append(self._current_mode)
        return self._current_mode

    def get_previous_mode(self) -> Optional[Mode]:
        """Get previous mode from history.

        Returns:
            Previous mode or None if no history
        """
        if len(self._mode_history) > 1:
            return self._mode_history[-2]
        return None

    def get_mode_display(self) -> str:
        """Get formatted mode display for UI.

        Returns:
            Formatted string: "Autopilot"
        """
        return self._current_mode.value

    def get_mode_description(self) -> str:
        """Get mode description for UI.

        Returns:
            Description string
        """
        return MODE_DESCRIPTIONS.get(self._current_mode, "")

_mode_manager: Optional[ModeManager] = None

def get_mode_manager() -> ModeManager:
    """Get global mode manager instance.

    Returns:
        Global ModeManager instance
    """
    global _mode_manager
    if _mode_manager is None:
        _mode_manager = ModeManager()
    return _mode_manager

def get_current_mode() -> Mode:
    """Get current mode.

    Returns:
        Current Mode
    """
    return get_mode_manager().current_mode

def set_mode(mode: Mode) -> None:
    """Set current mode.

    Args:
        mode: Mode to switch to
    """
    get_mode_manager().set_mode(mode)

def cycle_mode() -> Mode:
    """Cycle to next mode.

    Returns:
        New current mode
    """
    return get_mode_manager().cycle_mode()

def get_mode_config() -> ModeConfig:
    """Get current mode configuration.

    Returns:
        ModeConfig for current mode
    """
    return get_mode_manager().get_config()
