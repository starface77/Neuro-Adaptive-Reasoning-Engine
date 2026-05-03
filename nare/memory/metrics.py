"""VARE MetricsTracker — Quality, Latency, Tokens.

Three composite metrics per MemoryBench:
  - Quality: fraction of verified solutions (score >= 0.5)
  - Latency: average response time, by-route breakdown
  - Tokens: average LLM consumption per query
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
        self.baseline_results: List[Dict[str, Any]] = []
        self._load()

    def record(self, query: str, route: str, elapsed: float,
               tokens_used: int, similarity: float,
               answer: str, score: float, correct: Optional[bool] = None):
        entry = {
            "timestamp": time.time(),
            "query": query[:200],
            "route": route,
            "elapsed_seconds": round(elapsed, 4),
            "tokens_used": tokens_used,
            "similarity": round(similarity, 4),
            "answer_length": len(answer),
            "score": round(score, 4),
            "correct": correct,
        }
        self.history.append(entry)
        self._save()

    def record_baseline(self, query: str, answer: str, elapsed: float, tokens: int, correct: bool):
        self.baseline_results.append({
            'query': query[:200],
            'answer': answer[:200],
            'elapsed': elapsed,
            'tokens': tokens,
            'correct': correct,
            'timestamp': time.time()
        })
        self._save()

    def quality(self) -> Dict[str, Any]:
        """Fraction of queries with score >= 0.5 (verified)."""
        if not self.history:
            return {"accuracy": 0.0, "correct": 0, "total": 0}
        verified = sum(1 for e in self.history if e.get("score", 0) >= 0.5)
        return {
            "accuracy": round(verified / len(self.history), 4),
            "correct": verified,
            "total": len(self.history),
        }

    def latency(self) -> Dict[str, Any]:
        """Average response time, by-route breakdown."""
        if not self.history:
            return {"avg_latency": 0.0, "median_latency": 0.0, "by_route": {}}
        times = [e["elapsed_seconds"] for e in self.history]
        by_route: Dict[str, List[float]] = {}
        for e in self.history:
            r = e["route"]
            by_route.setdefault(r, []).append(e["elapsed_seconds"])
        return {
            "avg_latency": round(float(np.mean(times)), 4),
            "median_latency": round(float(np.median(times)), 4),
            "by_route": {
                r: round(float(np.mean(ts)), 4)
                for r, ts in by_route.items()
            },
        }

    def tokens(self) -> Dict[str, Any]:
        """Average token usage per query."""
        if not self.history:
            return {"avg_tokens": 0.0, "total_tokens": 0}
        toks = [e["tokens_used"] for e in self.history]
        return {
            "avg_tokens": round(float(np.mean(toks)), 2),
            "total_tokens": sum(toks),
        }

    def routing_stats(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Route distribution."""
        entries = self.history[-last_n:] if last_n else self.history
        if not entries:
            return {"total_queries": 0, "route_counts": {}, "amortization_ratio": 0.0}
        counts: Dict[str, int] = {}
        for e in entries:
            r = e["route"]
            counts[r] = counts.get(r, 0) + 1
        total = len(entries)
        amortized = counts.get("FAST", 0) + counts.get("COMPILED_SKILL", 0)
        return {
            "total_queries": total,
            "route_counts": counts,
            "amortization_ratio": round(amortized / total, 4) if total else 0.0,
        }

    def compute_deltas(self) -> Dict[str, float]:
        """Compute deltas between VARE and baseline."""
        if not self.baseline_results or not self.history:
            return {}
        vare_correct = [r for r in self.history if r.get('correct') is not None]
        vare_accuracy = (sum(1 for r in vare_correct if r['correct']) / len(vare_correct)) if vare_correct else 0.0
        baseline_accuracy = sum(1 for r in self.baseline_results if r['correct']) / len(self.baseline_results)
        vare_latency = float(np.mean([r['elapsed_seconds'] for r in self.history]))
        baseline_latency = float(np.mean([r['elapsed'] for r in self.baseline_results]))
        vare_tokens = float(np.mean([r['tokens_used'] for r in self.history]))
        baseline_tokens = float(np.mean([r['tokens'] for r in self.baseline_results]))
        return {
            'delta_quality': round(vare_accuracy - baseline_accuracy, 4),
            'delta_latency_pct': round(((vare_latency - baseline_latency) / baseline_latency) * 100, 2) if baseline_latency > 0 else 0.0,
            'delta_tokens_pct': round(((vare_tokens - baseline_tokens) / baseline_tokens) * 100, 2) if baseline_tokens > 0 else 0.0,
            'vare_accuracy': round(vare_accuracy, 4),
            'baseline_accuracy': round(baseline_accuracy, 4),
        }

    def compute_amortization_rate(self) -> float:
        if not self.history:
            return 0.0
        fast = sum(1 for r in self.history if r.get('route') in ('FAST', 'COMPILED_SKILL'))
        return round(fast / len(self.history), 4)

    def cost_trend(self, window: int = 10) -> Dict[str, Any]:
        if len(self.history) < 2:
            return {"insufficient_data": True}
        mid = len(self.history) // 2
        first = self.history[:mid]
        second = self.history[mid:]
        at1 = float(np.mean([e["elapsed_seconds"] for e in first]))
        at2 = float(np.mean([e["elapsed_seconds"] for e in second]))
        tt1 = float(np.mean([e["tokens_used"] for e in first]))
        tt2 = float(np.mean([e["tokens_used"] for e in second]))
        return {
            "first_half_avg_latency": round(at1, 4),
            "second_half_avg_latency": round(at2, 4),
            "latency_reduction_pct": round((1 - at2 / max(at1, 1e-9)) * 100, 1),
            "first_half_avg_tokens": round(tt1, 1),
            "second_half_avg_tokens": round(tt2, 1),
            "token_reduction_pct": round((1 - tt2 / max(tt1, 1e-9)) * 100, 1),
        }

    def convergence(self) -> Dict[str, Any]:
        if len(self.history) < 3:
            return {"insufficient_data": True}
        mid = len(self.history) // 2
        v1 = float(np.var([e["answer_length"] for e in self.history[:mid]]))
        v2 = float(np.var([e["answer_length"] for e in self.history[mid:]]))
        return {"answer_length_variance_first": round(v1, 2), "answer_length_variance_second": round(v2, 2), "converging": v2 < v1}

    def stability_plasticity(self) -> Dict[str, Any]:
        if len(self.history) < 5:
            return {"insufficient_data": True}
        scores = [e["score"] for e in self.history]
        w = min(5, len(scores) // 2)
        early = float(np.mean(scores[:w]))
        late = float(np.mean(scores[-w:]))
        return {"early_avg_score": round(early, 4), "late_avg_score": round(late, 4), "stable": late >= early * 0.9}

    def amortization_dynamics(self, memory_size: int, kappa: float = 0.05, c_llm: float = 2000.0, c_mem: float = 0.0) -> Dict[str, Any]:
        alpha = self.compute_amortization_rate()
        c_t = (1.0 - alpha) * c_llm + alpha * c_mem
        return {
            "memory_size": memory_size,
            "alpha_t_empirical": round(float(alpha), 6),
            "cost_t_theoretical": round(float(c_t), 2),
        }

    def summary(self, memory_size: int = 0, config: Any = None) -> Dict[str, Any]:
        return {
            "quality": self.quality(),
            "latency": self.latency(),
            "tokens": self.tokens(),
            "routing": self.routing_stats(),
            "cost": self.cost_trend(),
            "convergence": self.convergence(),
            "stability": self.stability_plasticity(),
        }

    def _save(self):
        path = os.path.join(self.persist_dir, "metrics.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({'history': self.history, 'baseline_results': self.baseline_results}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")

    def _load(self):
        path = os.path.join(self.persist_dir, "metrics.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.history = data.get('history', [])
                        self.baseline_results = data.get('baseline_results', [])
                    else:
                        self.history = data
                        self.baseline_results = []
            except Exception:
                self.history = []
                self.baseline_results = []
