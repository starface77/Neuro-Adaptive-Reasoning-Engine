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
            "answer_hash": hash(answer[:100]),
            "answer_length": len(answer),
            "score": round(score, 4),
            "correct": correct,
        }
        self.history.append(entry)
        self._save()

    def record_baseline(self, query: str, answer: str, elapsed: float, tokens: int, correct: bool):
        """Record baseline LLM performance (no memory/routing)."""
        self.baseline_results.append({
            'query': query[:200],
            'answer': answer[:200],
            'elapsed': elapsed,
            'tokens': tokens,
            'correct': correct,
            'timestamp': time.time()
        })
        self._save()

    def compute_deltas(self) -> Dict[str, float]:
        """Compute Δ between NARE and baseline (MemoryBench metrics)."""
        if not self.baseline_results or not self.history:
            return {}

        vare_correct = [r for r in self.history if r.get('correct') is not None]
        if vare_correct:
            vare_accuracy = sum(1 for r in vare_correct if r['correct']) / len(vare_correct)
        else:
            vare_accuracy = 0.0

        baseline_accuracy = sum(1 for r in self.baseline_results if r['correct']) / len(self.baseline_results)
        delta_quality = vare_accuracy - baseline_accuracy

        vare_latency = np.mean([r['elapsed_seconds'] for r in self.history])
        baseline_latency = np.mean([r['elapsed'] for r in self.baseline_results])
        delta_latency = (vare_latency - baseline_latency) / baseline_latency if baseline_latency > 0 else 0.0

        vare_tokens = np.mean([r['tokens_used'] for r in self.history])
        baseline_tokens = np.mean([r['tokens'] for r in self.baseline_results])
        delta_tokens = (vare_tokens - baseline_tokens) / baseline_tokens if baseline_tokens > 0 else 0.0

        return {
            'delta_quality': round(delta_quality, 4),
            'delta_latency_pct': round(delta_latency * 100, 2),
            'delta_tokens_pct': round(delta_tokens * 100, 2),
            'vare_accuracy': round(vare_accuracy, 4),
            'baseline_accuracy': round(baseline_accuracy, 4),
            'vare_latency': round(vare_latency, 4),
            'baseline_latency': round(baseline_latency, 4),
            'vare_tokens': round(vare_tokens, 2),
            'baseline_tokens': round(baseline_tokens, 2),
        }

    def compute_amortization_rate(self) -> float:
        """Compute % of queries served by FAST/REFLEX/COMPILED_SKILL (O(1) paths)."""
        if not self.history:
            return 0.0
        fast_routes = sum(1 for r in self.history if r.get('route') in ('FAST', 'REFLEX', 'COMPILED_SKILL'))
        return round(fast_routes / len(self.history), 4)

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
        amortized = sum(counts.get(r, 0) for r in ("FAST", "REFLEX", "REFLEX_PROVISIONAL", "COMPILED_SKILL"))
        return {
            "total_queries": total,
            "route_counts": counts,
            "route_pct": {r: round(c / total * 100, 1) for r, c in counts.items()},
            "amortization_ratio": round(amortized / total, 4),
            "alpha_mean": round(np.mean([e["similarity"] for e in entries]), 4),
        }

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

    def amortization_dynamics(
        self,
        memory_size: int,
        kappa: float = 0.05,
        c_llm: float = 2000.0,
        c_mem: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute formal amortization metrics.

        alpha_t now represents empirical amortization ratio,
        not a function of global memory size.

        Old formula (incorrect): α_t = 1 - exp(-κ·|M_t|)
        New formula (honest): α_t = empirical ratio of FAST/REFLEX queries

        C_t = (1-α_t)·C_LLM + α_t·C_mem  — blended cost
        """

        if self.history:
            amortized_count = sum(
                1 for e in self.history
                if e["route"] in ("FAST", "REFLEX", "REFLEX_PROVISIONAL", "COMPILED_SKILL")
            )
            alpha_empirical = amortized_count / len(self.history)
        else:
            alpha_empirical = 0.0

        alpha_t = alpha_empirical
        c_t = (1.0 - alpha_t) * c_llm + alpha_t * c_mem
        dc_dm = -kappa * np.exp(-kappa * memory_size) * (c_llm - c_mem)

        if self.history:
            c_empirical = np.mean([e["tokens_used"] for e in self.history])
        else:
            c_empirical = c_llm

        return {
            "memory_size": memory_size,
            "alpha_t_theoretical": round(float(alpha_t), 6),
            "alpha_t_empirical": round(float(alpha_empirical), 6),
            "cost_t_theoretical": round(float(c_t), 2),
            "cost_t_empirical": round(float(c_empirical), 2),
            "dCost_dMemory": round(float(dc_dm), 4),
            "kappa": kappa,
        }

    def summary(self, memory_size: int = 0, config: "Any" = None) -> Dict[str, Any]:
        kappa = 0.05
        c_llm = 2000.0
        c_mem = 0.0
        if config is not None:
            kappa = config.amortization.kappa
            c_llm = config.amortization.c_llm
            c_mem = config.amortization.c_mem
        return {
            "routing": self.routing_stats(),
            "cost": self.cost_trend(),
            "convergence": self.convergence(),
            "stability": self.stability_plasticity(),
            "amortization": self.amortization_dynamics(
                memory_size=memory_size,
                kappa=kappa,
                c_llm=c_llm,
                c_mem=c_mem,
            ),
        }

    def _save(self):
        path = os.path.join(self.persist_dir, "metrics.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    'history': self.history,
                    'baseline_results': self.baseline_results
                }, f, ensure_ascii=False, indent=2)
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
