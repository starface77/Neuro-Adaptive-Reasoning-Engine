"""Route usage metrics.

Tracks which routing paths are used and how often compiled skills are executed.
Helps understand system determinism: high COMPILED_SKILL usage = predictable system.
"""

from collections import defaultdict
from typing import Dict, Optional

class RouteMetrics:
    """Track routing decisions and skill usage."""

    def __init__(self):
        self.route_counts: Dict[str, int] = defaultdict(int)
        self.skill_usage: Dict[str, int] = defaultdict(int)

    def record_route(self, route: str, pattern: Optional[str] = None):
        """Record a routing decision.

        Args:
            route: Route name (COMPILED_SKILL, FAST, HYBRID, SLOW, etc.)
            pattern: Skill pattern if route is COMPILED_SKILL
        """
        self.route_counts[route] += 1
        if pattern:
            self.skill_usage[pattern] += 1

    def get_stats(self) -> Dict:
        """Get routing statistics.

        Returns:
            Dict with total_queries, route_distribution, and top_skills
        """
        total = sum(self.route_counts.values())
        return {
            "total_queries": total,
            "route_distribution": {
                route: count / total if total > 0 else 0
                for route, count in self.route_counts.items()
            },
            "top_skills": sorted(
                self.skill_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def reset(self):
        """Reset all metrics."""
        self.route_counts.clear()
        self.skill_usage.clear()
