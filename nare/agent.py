import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple

from .config import DEFAULT_CONFIG, NareConfig
from .memory import MemorySystem
from .metrics import MetricsTracker
from .critic import HybridCritic
from .core.router import ReasoningRouter
from .core.evolution import EvolutionEngine
from . import llm

class NAREProductionAgent:
    """The production-grade NARE Agent.
    
    Acts as a facade coordinating:
    - MemorySystem: Persistence and retrieval
    - ReasoningRouter: Decision making and execution
    - EvolutionEngine: Background learning and maintenance
    """

    def __init__(
        self,
        config: NareConfig = DEFAULT_CONFIG,
        persist_dir: Optional[str] = None,
    ):
        self.config = config
        self.memory = MemorySystem(
            config=config,
            persist_dir=persist_dir,
        )
        self.critic = HybridCritic(config=config)
        self.metrics = MetricsTracker(persist_dir=self.memory.persist_dir)
        
        # Core components
        self.router = ReasoningRouter(
            memory=self.memory,
            critic=self.critic,
            config=self.config,
            metrics=self.metrics
        )
        self.evolution = EvolutionEngine(
            memory=self.memory,
            config=self.config
        )

    def set_memory(self, memory: MemorySystem):
        """Replace the memory store and re-wire all sub-components.

        This is needed when the benchmark or caller swaps in a different
        persist_dir after construction. Without this, the router and
        evolution engine keep a stale reference to the old memory and
        the FAST cache path silently never fires.
        """
        self.memory = memory
        self.router.memory = memory
        self.evolution.memory = memory
        self.metrics = MetricsTracker(persist_dir=memory.persist_dir)
        self.router.metrics = self.metrics

        if config.bootstrap.load_seeds_on_init:
            self._bootstrap_load_seeds()

    def solve(
        self,
        query: str,
        oracle: Optional[Callable] = None,
        expected_hint: Optional[str] = None,
        oracle_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Solve a query by delegating to the Router."""
        
        # Build functional oracle from spec if needed
        if oracle is None and oracle_spec is not None:
            from .oracle import build_oracle_from_spec
            oracle = build_oracle_from_spec(oracle_spec)

        # Route query through the 4-tier pipeline
        result = self.router.route(query, oracle, expected_hint)
        
        # Post-solve actions
        self._after_solve(query, result)
        
        return result

    def _after_solve(self, query: str, result: Dict[str, Any]):
        """Perform background updates after a solve call."""
        route = result.get("route_decision")
        final_answer = result.get("final_answer")
        
        # 1. Save new episodes for SLOW/HYBRID paths
        if route in ("SLOW", "HYBRID") and result.get("generated_candidates"):
            best = result["generated_candidates"][0]
            if best.get("final_score", 0) > 0.5:
                self._save_episode(query, best)

        # 2. Trigger sleep cycle if needed
        if self.evolution.check_sleep_trigger():
            self.evolution.run_sleep_cycle()

    def _save_episode(self, query: str, best_cand: Dict):
        """Persist a successful reasoning trace to episodic memory."""
        query_emb = llm.get_embedding(query)
        episode_data = {
            "query": query,
            "solution": best_cand['solution'],
            "reasoning_trace": best_cand.get('reasoning', 'N/A'),
            "score": best_cand.get('final_score', 0.8),
            "embedding": query_emb,
            "timestamp": time.time()
        }
        self.memory.add_episode(episode_data, np.array([query_emb], dtype=np.float32))
        logging.info(f"[Memory] Saved new {episode_data['score']:.2f} episode.")

    def _bootstrap_load_seeds(self):
        """Placeholder for seed loading logic."""
        pass
    
    def wait_for_sleep(self, timeout=300):
        """Wait for background evolution tasks to finish."""
        start = time.time()
        while self.evolution._is_sleeping:
            if time.time() - start > timeout: break
            time.sleep(1)
