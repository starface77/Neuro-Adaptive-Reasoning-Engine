"""
MetricsTracker: Continuous learning metrics per NARE theory.

Tracks 4 key metrics:
1. Recall & Precision of routing (amortization efficiency)
2. Computational cost reduction over time
3. Convergence of inferences (variance reduction)
4. Stability-plasticity balance (no catastrophic interference)
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
    # Metric 1: Routing Recall & Precision
    # ------------------------------------------------------------------
    def routing_stats(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Distribution of routing decisions."""
        entries = self.history[-last_n:] if last_n else self.history
        if not entries:
            return {}
        counts = {}
        for e in entries:
            r = e["route"]
            counts[r] = counts.get(r, 0) + 1
        total = len(entries)
        amortized = sum(counts.get(r, 0) for r in ("FAST", "REFLEX", "REFLEX_PROVISIONAL"))
        return {
            "total_queries": total,
            "route_counts": counts,
            "route_pct": {r: round(c / total * 100, 1) for r, c in counts.items()},
            "amortization_ratio": round(amortized / total, 4),
            "alpha_mean": round(np.mean([e["similarity"] for e in entries]), 4),
        }

    # ------------------------------------------------------------------
    # Metric 2: Computational Cost Reduction
    # ------------------------------------------------------------------
    def cost_trend(self, window: int = 10) -> Dict[str, Any]:
        """Rolling average of latency and token usage."""
        if len(self.history) < 2:
            return {"insufficient_data": True}
        first_half = self.history[: len(self.history) // 2]
        second_half = self.history[len(self.history) // 2:]
        avg_time_first = np.mean([e["elapsed_seconds"] for e in first_half])
        avg_time_second = np.mean([e["elapsed_seconds"] for e in second_half])
        avg_tok_first = np.mean([e["tokens_used"] for e in first_half])
        avg_tok_second = np.mean([e["tokens_used"] for e in second_half])
        return {
            "first_half_avg_latency": round(avg_time_first, 4),
            "second_half_avg_latency": round(avg_time_second, 4),
            "latency_reduction_pct": round((1 - avg_time_second / max(avg_time_first, 1e-9)) * 100, 1),
            "first_half_avg_tokens": round(avg_tok_first, 1),
            "second_half_avg_tokens": round(avg_tok_second, 1),
            "token_reduction_pct": round((1 - avg_tok_second / max(avg_tok_first, 1e-9)) * 100, 1),
        }

    # ------------------------------------------------------------------
    # Metric 3: Convergence (variance reduction for similar queries)
    # ------------------------------------------------------------------
    def convergence(self) -> Dict[str, Any]:
        """Measure answer determinism over time."""
        if len(self.history) < 3:
            return {"insufficient_data": True}
        first_half = self.history[: len(self.history) // 2]
        second_half = self.history[len(self.history) // 2:]
        var_first = np.var([e["answer_length"] for e in first_half])
        var_second = np.var([e["answer_length"] for e in second_half])
        latency_var_first = np.var([e["elapsed_seconds"] for e in first_half])
        latency_var_second = np.var([e["elapsed_seconds"] for e in second_half])
        return {
            "answer_length_variance_first": round(var_first, 2),
            "answer_length_variance_second": round(var_second, 2),
            "latency_variance_first": round(latency_var_first, 4),
            "latency_variance_second": round(latency_var_second, 4),
            "converging": var_second < var_first,
        }

    # ------------------------------------------------------------------
    # Metric 4: Stability-Plasticity Balance
    # ------------------------------------------------------------------
    def stability_plasticity(self) -> Dict[str, Any]:
        """Check that new learning doesn't destroy old skills."""
        if len(self.history) < 5:
            return {"insufficient_data": True}
        scores = [e["score"] for e in self.history]
        window = min(5, len(scores) // 2)
        early_avg = np.mean(scores[:window])
        late_avg = np.mean(scores[-window:])
        return {
            "early_avg_score": round(early_avg, 4),
            "late_avg_score": round(late_avg, 4),
            "score_degradation": round(early_avg - late_avg, 4),
            "stable": late_avg >= early_avg * 0.9,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "routing": self.routing_stats(),
            "cost": self.cost_trend(),
            "convergence": self.convergence(),
            "stability": self.stability_plasticity(),
        }

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
