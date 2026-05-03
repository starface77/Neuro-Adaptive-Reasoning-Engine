"""MetricsTracker for VARE.

Three composite metrics (per MemoryBench):
  1. Quality (Accuracy)  — fraction of verified solutions
  2. Latency            — average response time
  3. Tokens             — LLM token consumption per query

Also tracks: routing distribution, cost trends, amortization dynamics.
"""

import time
import json
import os
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self, persist_dir: str = "memory_store"):
        self.persist_dir = persist_dir
        self.history: List[Dict[str, Any]] = []
        self._load()

    def record(self, query: str, route: str, elapsed: float,
               tokens_used: int, similarity: float,
               answer: str, score: float):
        entry = {
            "timestamp": time.time(),
            "query": query[:200],
            "route": route,
            "elapsed_seconds": round(elapsed, 4),
            "tokens_used": tokens_used,
            "similarity": round(similarity, 4),
            "answer_hash": hash(answer[:100]),
            "answer_length": len(answer),
            "score": round(score, 4),
        }
        self.history.append(entry)
        self._save()

    # ------------------------------------------------------------------
    # Metric 1: Quality (Accuracy)
    # ------------------------------------------------------------------
    def quality(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Fraction of queries with score >= 0.5 (verified)."""
        entries = self.history[-last_n:] if last_n else self.history
        if not entries:
            return {"accuracy": 0.0, "total": 0}
        correct = sum(1 for e in entries if e["score"] >= 0.5)
        return {
            "accuracy": round(correct / len(entries), 4),
            "correct": correct,
            "total": len(entries),
        }

    # ------------------------------------------------------------------
    # Metric 2: Latency
    # ------------------------------------------------------------------
    def latency(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Average latency and breakdown by route."""
        entries = self.history[-last_n:] if last_n else self.history
        if not entries:
            return {"avg_latency": 0.0}
        times = [e["elapsed_seconds"] for e in entries]
        by_route: Dict[str, List[float]] = {}
        for e in entries:
            by_route.setdefault(e["route"], []).append(e["elapsed_seconds"])
        return {
            "avg_latency": round(np.mean(times), 4),
            "median_latency": round(float(np.median(times)), 4),
            "by_route": {
                r: round(np.mean(t), 4) for r, t in by_route.items()
            },
        }

    # ------------------------------------------------------------------
    # Metric 3: Token Efficiency
    # ------------------------------------------------------------------
    def tokens(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Average token usage per query."""
        entries = self.history[-last_n:] if last_n else self.history
        if not entries:
            return {"avg_tokens": 0.0}
        toks = [e["tokens_used"] for e in entries]
        return {
            "avg_tokens": round(np.mean(toks), 1),
            "total_tokens": sum(toks),
        }

    # ------------------------------------------------------------------
    # Routing Distribution
    # ------------------------------------------------------------------
    def routing_stats(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        entries = self.history[-last_n:] if last_n else self.history
        if not entries:
            return {}
        counts: Dict[str, int] = {}
        for e in entries:
            r = e["route"]
            counts[r] = counts.get(r, 0) + 1
        total = len(entries)
        amortized = counts.get("FAST", 0)
        return {
            "total_queries": total,
            "route_counts": counts,
            "route_pct": {r: round(c / total * 100, 1) for r, c in counts.items()},
            "amortization_ratio": round(amortized / total, 4),
        }

    # ------------------------------------------------------------------
    # Cost Trend
    # ------------------------------------------------------------------
    def cost_trend(self, window: int = 10) -> Dict[str, Any]:
        if len(self.history) < 2:
            return {"insufficient_data": True}
        first_half = self.history[:len(self.history) // 2]
        second_half = self.history[len(self.history) // 2:]
        avg_time_first = np.mean([e["elapsed_seconds"] for e in first_half])
        avg_time_second = np.mean([e["elapsed_seconds"] for e in second_half])
        avg_tok_first = np.mean([e["tokens_used"] for e in first_half])
        avg_tok_second = np.mean([e["tokens_used"] for e in second_half])
        return {
            "first_half_avg_latency": round(avg_time_first, 4),
            "second_half_avg_latency": round(avg_time_second, 4),
            "latency_reduction_pct": round(
                (1 - avg_time_second / max(avg_time_first, 1e-9)) * 100, 1
            ),
            "first_half_avg_tokens": round(avg_tok_first, 1),
            "second_half_avg_tokens": round(avg_tok_second, 1),
            "token_reduction_pct": round(
                (1 - avg_tok_second / max(avg_tok_first, 1e-9)) * 100, 1
            ),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self, memory_size: int = 0, config: "Any" = None) -> Dict[str, Any]:
        return {
            "quality": self.quality(),
            "latency": self.latency(),
            "tokens": self.tokens(),
            "routing": self.routing_stats(),
            "cost_trend": self.cost_trend(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self):
        path = os.path.join(self.persist_dir, "metrics.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")

    def _load(self):
        path = os.path.join(self.persist_dir, "metrics.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []
