"""Budget management and planning."""

from __future__ import annotations

from typing import Optional
from nare.utils.logger import get_logger

logger = get_logger(__name__)


class BudgetManager:
    """Manages token budget and planning."""
    
    def __init__(self, total_budget: int = 200000):
        self.total_budget = total_budget
        self.used_budget = 0
        self.checkpoints = []
    
    @property
    def remaining(self) -> int:
        """Get remaining budget."""
        return self.total_budget - self.used_budget
    
    @property
    def usage_percentage(self) -> float:
        """Get budget usage as percentage."""
        return (self.used_budget / self.total_budget) * 100
    
    def consume(self, amount: int, operation: Optional[str] = None) -> bool:
        """
        Consume tokens from budget.
        
        Args:
            amount: Number of tokens to consume
            operation: Optional operation name for logging
        
        Returns:
            True if consumption successful, False if insufficient budget
        """
        if self.remaining >= amount:
            self.used_budget += amount
            
            if operation:
                logger.debug(
                    f"Budget consumed: {amount} tokens for '{operation}' "
                    f"({self.usage_percentage:.1f}% used)"
                )
            
            return True
        
        logger.warning(
            f"Insufficient budget: requested {amount}, "
            f"remaining {self.remaining}"
        )
        return False
    
    def create_checkpoint(self, name: str):
        """Create a budget checkpoint."""
        checkpoint = {
            "name": name,
            "used": self.used_budget,
            "remaining": self.remaining
        }
        self.checkpoints.append(checkpoint)
        logger.debug(f"Checkpoint '{name}': {self.used_budget} tokens used")
    
    def get_checkpoint_delta(self, checkpoint_name: str) -> Optional[int]:
        """Get token usage since a checkpoint."""
        for checkpoint in self.checkpoints:
            if checkpoint["name"] == checkpoint_name:
                return self.used_budget - checkpoint["used"]
        return None
    
    def reset(self):
        """Reset budget usage."""
        self.used_budget = 0
        self.checkpoints.clear()
        logger.info("Budget reset")
    
    def should_warn(self, threshold: float = 0.8) -> bool:
        """Check if budget usage exceeds warning threshold."""
        return self.usage_percentage >= (threshold * 100)