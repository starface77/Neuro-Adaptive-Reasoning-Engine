"""Shared context for coordinating components during solve."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time

@dataclass
class SolveContext:
    """Shared state across Router, VS, Memory, Critic during solve.

    Enables components to coordinate through data, not magic.
    """
    query: str
    oracle: Optional[Any] = None

    attempts: List[Dict[str, Any]] = field(default_factory=list)
    best_iou: float = 0.0
    best_solution: Optional[str] = None

    partial_solutions: List[Dict[str, Any]] = field(default_factory=list)

    start_time: float = field(default_factory=time.time)

    def add_attempt(self, solution: str, iou: float, converged: bool, error: Optional[str] = None):
        """Record an attempt with its IoU score."""
        self.attempts.append({
            'solution': solution,
            'iou': iou,
            'converged': converged,
            'error': error,
            'timestamp': time.time()
        })

        if iou > self.best_iou:
            self.best_iou = iou
            self.best_solution = solution

        if 0.95 <= iou < 1.0:
            self.partial_solutions.append({
                'solution': solution,
                'iou': iou,
                'attempt_num': len(self.attempts)
            })

    def should_extend_attempts(self, current_max: int) -> bool:
        """Decide if we should add more attempts.

        Extend if:
        - IoU is improving (>= 0.95)
        - Haven't exceeded hard limit (12 attempts)
        """
        if len(self.attempts) >= 12:
            return False

        if self.best_iou >= 0.95:

            return True

        return False

    def get_feedback_for_next_attempt(self) -> str:
        """Generate feedback prompt for next VS attempt."""
        if not self.attempts:
            return ""

        if self.best_iou >= 0.95:
            return f"\nPrevious best attempt achieved {self.best_iou:.1%} accuracy. Very close - fix remaining errors."
        elif self.best_iou >= 0.80:
            return f"\nPrevious attempts reached {self.best_iou:.1%} accuracy. Significant progress made."
        else:
            return ""

    def elapsed(self) -> float:
        """Time elapsed since solve started."""
        return time.time() - self.start_time
