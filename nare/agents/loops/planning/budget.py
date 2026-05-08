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


def estimate_token_usage(operation: str, context_size: int = 0) -> int:
    """
    Estimate token usage for an operation.

    Args:
        operation: Operation type (e.g., 'llm_call', 'tool_execution')
        context_size: Size of context in tokens

    Returns:
        Estimated token count
    """
    estimates = {
        'llm_call': 1000 + context_size,
        'tool_execution': 100,
        'planning': 500,
        'synthesis': 2000 + context_size,
        'verification': 300,
    }
    return estimates.get(operation, 500)


def check_budget_before_call(budget_manager: BudgetManager, operation: str, context_size: int = 0) -> bool:
    """
    Check if budget is sufficient before making a call.

    Args:
        budget_manager: BudgetManager instance
        operation: Operation type
        context_size: Size of context in tokens

    Returns:
        True if sufficient budget, False otherwise
    """
    estimated = estimate_token_usage(operation, context_size)
    return budget_manager.remaining >= estimated


def update_budget_after_call(budget_manager: BudgetManager, actual_tokens: int, operation: str):
    """
    Update budget after a call completes.

    Args:
        budget_manager: BudgetManager instance
        actual_tokens: Actual tokens consumed
        operation: Operation name for logging
    """
    budget_manager.consume(actual_tokens, operation)