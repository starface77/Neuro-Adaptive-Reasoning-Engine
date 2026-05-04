import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple

from ..config import DEFAULT_CONFIG, NareConfig
from ..memory.memory import MemorySystem
from ..memory.metrics import MetricsTracker
from ..reasoning.critic import Critic
from .router import ReasoningRouter
from .evolution import EvolutionEngine
from ..reasoning import llm

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
        embedding_dim: int = 3072,
    ):
        self.config = config
        self.memory = MemorySystem(
            config=config,
            persist_dir=persist_dir,
            embedding_dim=embedding_dim,
        )

        # CRITICAL: Load existing memory from disk
        self.memory.load()
        logging.info(f"[Agent] Loaded memory: {len(self.memory.episodes)} episodes, {len(self.memory.compiled_skills)} skills")

        self.critic = Critic()
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

        if self.config.bootstrap.load_seeds_on_init:
            self._bootstrap_load_seeds()

    def solve(
        self,
        query: str,
        oracle: Optional[Callable] = None,
        expected_hint: Optional[str] = None,
        oracle_spec: Optional[Dict[str, Any]] = None,
        file_provider: Optional[Callable] = None,
        thinking_display=None,
        working_dir: str = ".",
        chat_history: Optional[str] = None,
        repo_map: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Solve a query by delegating to the Router.

        Parameters
        ----------
        file_provider
            Optional ``(path) -> Optional[str]`` callback. When the LLM
            responds with ``CANNOT_FIX: need file <path>``, the Verified
            Synthesis loop calls this to fetch the file and inject it
            into the next attempt's context.  This is the minimal
            agentic capability: the system requests missing context
            without a full agent loop.
        working_dir
            Working directory for file operations (default: current directory)
        chat_history
            String of previous conversation turns to provide context
        repo_map
            String representation of repository directory structure
        """

        # Build functional oracle from spec if needed
        if oracle is None and oracle_spec is not None:
            from .oracle import build_oracle_from_spec
            oracle = build_oracle_from_spec(oracle_spec)

        # Route query through the 4-tier pipeline
        result = self.router.route(
            query=query,
            oracle=oracle,
            expected_hint=expected_hint,
            file_provider=file_provider,
            thinking_display=thinking_display,
            working_dir=working_dir,
            chat_history=chat_history,
            repo_map=repo_map,
            intent=intent
        )
        
        # Post-solve actions (best-effort — never crash the user's answer)
        try:
            self._after_solve(query, result)
        except Exception as e:
            logging.warning(f"[Agent] Post-solve error (non-fatal): {e}")

        return result

    def _after_solve(self, query: str, result: Dict[str, Any]):
        route = result.get("route_decision")
        final_answer = result.get("final_answer")

        if route in ("SLOW", "HYBRID") and result.get("generated_candidates"):
            best = result["generated_candidates"][0]
            solve_ctx = best.get("solve_context")

            # Save high-quality solutions (80%+) to main memory
            if best.get("final_score", 0) >= 0.80:
                self._save_episode(query, best)

                # Check if this pattern should be compiled as a skill
                query_emb = llm.get_embedding(query)
                similar_eps = self.memory.retrieve_episodes(np.array([query_emb], dtype=np.float32), k=5)

                # Count episodes with high similarity (>0.85)
                high_sim_count = sum(1 for ep in similar_eps if ep.get('similarity', 0) > 0.85)

                if high_sim_count >= 3:
                    # Compile as skill
                    self._compile_skill(query, best['solution'], query_emb)

            # Save partial solutions (95%+ IoU but didn't fully converge)
            elif solve_ctx is not None and solve_ctx.partial_solutions:
                self._save_partial_solutions(query, solve_ctx)

        # Prune memory periodically
        if len(self.memory.episodes) > 100:
            self.memory.prune_memory()

        if self.config.sleep.enabled and self.evolution.check_compilation_trigger():
            self.evolution.run_compilation_cycle()

    def _save_episode(self, query: str, best_cand: Dict):
        query_emb = llm.get_embedding(query)
        episode_data = {
            "query": query,
            "solution": best_cand['solution'],
            "reasoning_trace": best_cand.get('reasoning', 'N/A'),
            "score": best_cand.get('final_score', 0.5),  # Conservative fallback
            "embedding": query_emb,
            "timestamp": time.time()
        }
        self.memory.add_episode(episode_data, np.array([query_emb], dtype=np.float32))
        logging.info(f"[Memory] Saved new {episode_data['score']:.2f} episode.")

    def _bootstrap_load_seeds(self):
        """Placeholder for seed loading logic."""
        pass

    def _compile_skill(self, query: str, solution: str, query_emb: list):
        """Compile repeated pattern as reusable skill."""
        from .sandbox import extract_python_block

        code_block = extract_python_block(solution)
        if not code_block:
            return  # Can't compile non-code solutions

        # Extract pattern (first line of query as trigger)
        pattern = query.split('\n')[0][:100]

        # Store as compiled skill
        self.memory.add_compiled_skill(
            pattern=pattern,
            code=code_block,
            trigger_emb=np.array(query_emb, dtype=np.float32)
        )

        logging.info(f"[LibraryLearning] Compiled skill from pattern: {pattern}")

    def _save_partial_solutions(self, query: str, context):
        """Save partial solutions (IoU >= 0.95) for future reference."""
        for partial in context.partial_solutions:
            query_emb = llm.get_embedding(query)
            episode_data = {
                "query": query,
                "solution": partial['solution'],
                "reasoning_trace": f"Partial solution (IoU: {partial['iou']:.2%}, attempt: {partial['attempt_num']})",
                "score": partial['iou'],
                "embedding": query_emb,
                "timestamp": time.time(),
                "partial": True  # Mark as partial for future retrieval
            }
            self.memory.add_episode(episode_data, np.array([query_emb], dtype=np.float32))
            logging.info(f"[Memory] Saved partial solution (IoU: {partial['iou']:.2%})")

    def wait_for_compilation(self, timeout=300):
        start = time.time()
        while self.evolution._is_compiling:
            if time.time() - start > timeout: break
            time.sleep(1)
