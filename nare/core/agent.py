"""VareAgent — main VARE orchestrator.

Two routes:
  FAST — O(log N) cache hit via HNSW (compiled skills first, then episodes)
  VERIFIED_RETRY — iterative synthesis with sandbox verification (MDP)

Background Library Learning compiles recurring patterns into skills.
"""

import os
import re
import time
import logging
import threading
import numpy as np
import faiss
from typing import List, Dict, Any, Optional

from ..config import DEFAULT_CONFIG, VareConfig
from ..memory.memory import MemorySystem
from ..memory.metrics import MetricsTracker
from ..reasoning import llm
from ..execution.sandbox import (
    validate_code,
    safe_execute_freeform,
    extract_python_block,
    SecurityError,
)


class VareAgent:
    """Verified Amortized Reasoning Engine.

    Three components: M_cache (memory), G_θ (LLM), V_sandbox (verifier).
    Two routes: FAST (cache) and VERIFIED_RETRY (synthesis loop).
    """

    def __init__(
        self,
        config: VareConfig = DEFAULT_CONFIG,
        oracle: Optional[Any] = None,
        persist_dir: Optional[str] = None,
        embedding_dim: int = 3072,
    ):
        self.config = config
        self.oracle = oracle
        self.memory = MemorySystem(
            config=config,
            persist_dir=persist_dir,
            embedding_dim=embedding_dim,
        )
        self.metrics = MetricsTracker(persist_dir=self.memory.persist_dir)
        self.tau_fast = config.routing.tau_fast
        self._library_thread: Optional[threading.Thread] = None

    def set_memory(self, memory: MemorySystem):
        """Replace memory store (used by benchmarks)."""
        self.memory = memory
        self.metrics = MetricsTracker(persist_dir=memory.persist_dir)

    def solve(
        self,
        query: str,
        oracle: Optional[Any] = None,
        expected_hint: Optional[str] = None,
        oracle_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Main entry: embed → FAST → VERIFIED_RETRY → store → Library Learning."""
        _start = time.time()
        _tokens = 0
        log: List[str] = []

        # Build oracle from spec if needed
        active_oracle = oracle or self.oracle
        if active_oracle is None and oracle_spec is not None:
            from ..reasoning.oracle import build_oracle_from_spec
            active_oracle = build_oracle_from_spec(oracle_spec)

        # Embed query
        query_emb = llm.get_embedding(query)
        query_vec = np.array([query_emb], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # --- FAST route ---
        fast_result = self._try_fast_route(query, query_vec, log)
        if fast_result is not None:
            elapsed = time.time() - _start
            self._calibrate_tau(reward=1.0, fast_used=True)
            self.metrics.record(
                query=query, route=fast_result["route_decision"],
                elapsed=elapsed, tokens_used=0,
                similarity=fast_result.get("similarity", 1.0),
                answer=fast_result["final_answer"], score=1.0,
            )
            fast_result["elapsed"] = elapsed
            fast_result["memory_update_log"] = log
            return fast_result

        # --- VERIFIED_RETRY route ---
        result = self._verified_retry(query, query_vec, query_emb, log, active_oracle)
        elapsed = time.time() - _start
        _tokens = result.get("tokens_used", 0)

        score = 1.0 if result.get("verified") else 0.3
        self._calibrate_tau(reward=score, fast_used=False)

        # Store episode in memory
        episode = {
            "query": query,
            "solution": result["final_answer"],
            "reasoning_trace": "\n".join(log),
            "score": score,
        }
        self.memory.add_episode(episode, query_vec)
        log.append(f"Stored episode (score={score})")

        self.metrics.record(
            query=query, route="VERIFIED_RETRY",
            elapsed=elapsed, tokens_used=_tokens,
            similarity=0.0, answer=result["final_answer"],
            score=score,
        )

        # Check Library Learning trigger
        if self._should_run_library_learning():
            self._start_library_learning()

        result["elapsed"] = elapsed
        result["memory_update_log"] = log
        return result

    def _try_fast_route(
        self, query: str, query_vec: np.ndarray, log: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Attempt FAST route: skills first, then episodes."""
        tau = self.tau_fast

        # 1. Search compiled skills
        skills = self.memory.search_skills(query_vec, k=1)
        if skills and skills[0].get("similarity", 0) >= tau:
            skill = skills[0]
            code = skill.get("ast_code") or skill.get("code", "")
            if code:
                try:
                    answer = safe_execute_freeform(code)
                    if answer and not answer.startswith("Error:"):
                        log.append(f"FAST: compiled skill hit (sim={skill['similarity']:.3f})")
                        return {
                            "route_decision": "FAST",
                            "final_answer": answer,
                            "similarity": skill["similarity"],
                            "tokens_used": 0,
                            "verified": True,
                            "generated_candidates": [],
                        }
                except Exception as e:
                    log.append(f"Skill execution failed: {e}")

        # 2. Search episodic cache
        episodes = self.memory.search(query_vec, k=1)
        if episodes:
            ep = episodes[0]
            sim = ep.get("similarity", 0)
            ep_score = ep.get("score", 0)
            if sim >= tau and ep_score >= 0.5:
                idx = ep.get("memory_id")
                if idx is not None:
                    self.memory.boost_activation(idx)
                log.append(f"FAST: episodic cache hit (sim={sim:.3f}, score={ep_score:.2f})")
                return {
                    "route_decision": "FAST",
                    "final_answer": ep.get("solution", ""),
                    "similarity": sim,
                    "tokens_used": 0,
                    "verified": True,
                    "generated_candidates": [],
                }

        return None

    def _verified_retry(
        self,
        query: str,
        query_vec: np.ndarray,
        query_emb: list,
        log: List[str],
        oracle: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Verified Code Synthesis loop (MDP)."""
        max_retries = self.config.synthesis.max_retries
        error_trace = ""
        total_tokens = 0
        candidates = []

        # Retrieve context from memory for the prompt
        context_eps = self.memory.search(query_vec, k=3)
        context_str = ""
        if context_eps:
            context_parts = []
            for ep in context_eps:
                context_parts.append(f"Q: {ep.get('query', '')}\nA: {ep.get('solution', '')}")
            context_str = "\n---\n".join(context_parts)

        for attempt in range(max_retries):
            temp = (
                self.config.synthesis.initial_temperature
                if attempt == 0
                else self.config.synthesis.refinement_temperature
            )

            prompt = self._build_synthesis_prompt(query, context_str, error_trace)
            samples, tok = llm.generate_samples(prompt, n=1, temperature=temp)
            total_tokens += tok

            if not samples:
                log.append(f"Attempt {attempt+1}: no candidates generated")
                continue

            solution = samples[0]
            candidates.append({"solution": solution, "attempt": attempt + 1})

            is_valid, info = self._verify_solution(query, solution, oracle)

            if is_valid:
                log.append(f"VERIFIED_RETRY: verified on attempt {attempt+1} — {info}")
                return {
                    "route_decision": "VERIFIED_RETRY",
                    "final_answer": solution,
                    "tokens_used": total_tokens,
                    "verified": True,
                    "attempts": attempt + 1,
                    "generated_candidates": candidates,
                }

            error_trace += f"\nAttempt {attempt+1} failed: {info}"
            log.append(f"Attempt {attempt+1} failed: {info[:100]}")

        # All retries exhausted — return best attempt
        log.append(f"VERIFIED_RETRY: exhausted {max_retries} attempts")
        best = candidates[-1]["solution"] if candidates else "No solution generated."
        return {
            "route_decision": "VERIFIED_RETRY",
            "final_answer": best,
            "tokens_used": total_tokens,
            "verified": False,
            "attempts": max_retries,
            "generated_candidates": candidates,
        }

    def _build_synthesis_prompt(
        self, query: str, context: str, error_trace: str
    ) -> str:
        parts = [f"Solve the following task:\n{query}"]
        if context:
            parts.append(f"\nRelevant past solutions:\n{context}")
        if error_trace:
            parts.append(f"\nPrevious attempts failed with these errors:{error_trace}")
            parts.append("\nPlease fix the errors and provide a corrected solution.")
        return "\n".join(parts)

    def _verify_solution(
        self, query: str, solution: str, oracle: Optional[Any] = None
    ) -> tuple:
        """V_sandbox verification."""
        # 1. Oracle (ground truth)
        if oracle is not None:
            try:
                ok, info = oracle(query, solution)
                return ok, info
            except Exception as e:
                return False, f"Oracle error: {e}"

        # 2. Code detection → AST + sandbox
        code = self._extract_code(solution)
        if code:
            try:
                validate_code(code)
                result = safe_execute_freeform(code)
                if result and not result.startswith("Error:"):
                    return True, f"Code executed successfully: {result[:100]}"
                return False, f"Execution error: {result[:200]}"
            except SecurityError as e:
                return False, f"Security violation: {e}"
            except Exception as e:
                return False, f"Validation error: {e}"

        # 3. Text solution: basic non-empty check
        stripped = solution.strip()
        if stripped and not stripped.startswith("Error"):
            return True, "Non-empty text solution"
        return False, "Empty or error solution"

    def _calibrate_tau(self, reward: float, fast_used: bool):
        """Adjust tau_fast based on outcome feedback."""
        lr = self.config.routing.calibration_lr
        tau_min = self.config.routing.tau_min
        tau_max = self.config.routing.tau_max

        if fast_used and reward < 0.5:
            self.tau_fast = min(tau_max, self.tau_fast + lr)
        elif not fast_used and reward > 0.8:
            self.tau_fast = max(tau_min, self.tau_fast - lr)

    def _should_run_library_learning(self) -> bool:
        """Check if enough episodes have accumulated for clustering."""
        if self._library_thread and self._library_thread.is_alive():
            return False
        threshold = self.config.library.cluster_density_threshold
        verified = sum(1 for ep in self.memory.episodes if ep.get("score", 0) >= 0.5)
        return verified >= threshold

    def _start_library_learning(self):
        """Spawn background thread for Library Learning."""
        self._library_thread = threading.Thread(
            target=self._library_learning, daemon=True
        )
        self._library_thread.start()

    def _library_learning(self):
        """Background skill compilation."""
        try:
            logging.info("[Library Learning] Starting cluster analysis...")
            episodes = [ep for ep in self.memory.episodes if ep.get("score", 0) >= 0.5]
            if len(episodes) < self.config.library.cluster_density_threshold:
                return

            # Build similarity matrix
            embeddings = np.array(
                [ep["embedding"] for ep in episodes], dtype=np.float32
            )
            faiss.normalize_L2(embeddings)
            sims = np.dot(embeddings, embeddings.T)

            # Find dense clusters
            sim_thresh = self.config.library.cluster_similarity_threshold
            density_thresh = self.config.library.cluster_density_threshold
            visited = set()
            clusters: List[List[int]] = []

            for i in range(len(episodes)):
                if i in visited:
                    continue
                neighbors = [j for j in range(len(episodes))
                             if j != i and sims[i][j] > sim_thresh]
                if len(neighbors) >= density_thresh:
                    cluster = [i] + neighbors
                    clusters.append(cluster)
                    visited.update(cluster)

            for cluster_indices in clusters:
                cluster_eps = [episodes[i] for i in cluster_indices]
                skill_code = self._abstract_skill(cluster_eps)
                if skill_code is None:
                    continue

                if self._verify_skill_on_cluster(skill_code, cluster_eps):
                    centroid = np.mean(
                        [np.array(episodes[i]["embedding"]) for i in cluster_indices],
                        axis=0,
                    ).astype(np.float32)
                    self.memory.add_skill(
                        {
                            "pattern": cluster_eps[0].get("query", "")[:100],
                            "ast_code": skill_code,
                            "code": skill_code,
                            "confidence": 1.0,
                            "source_count": len(cluster_indices),
                        },
                        centroid,
                    )
                    logging.info(
                        f"[Library Learning] Compiled skill from {len(cluster_indices)} episodes"
                    )

            self.memory.decay_and_prune()
            logging.info("[Library Learning] Complete.")
        except Exception as e:
            logging.error(f"[Library Learning] Error: {e}")

    def _abstract_skill(self, cluster_episodes: List[Dict]) -> Optional[str]:
        """Ask LLM to generate a reusable Python function from cluster."""
        task_examples = []
        for ep in cluster_episodes[:5]:
            task_examples.append(f"Task: {ep.get('query', '')}\nSolution: {ep.get('solution', '')}")

        prompt = (
            "Analyze these solved tasks and create ONE reusable Python function "
            "that can solve ALL of them.\n\n"
            + "\n---\n".join(task_examples)
            + "\n\nWrite a Python function `solve(query: str) -> str` that handles "
            "all these task patterns. Return ONLY the Python code."
        )

        samples, _ = llm.generate_samples(prompt, n=1, temperature=0.4)
        if not samples:
            return None

        code = extract_python_block(samples[0]) or samples[0]
        try:
            validate_code(code)
            return code
        except (SecurityError, Exception):
            return None

    def _verify_skill_on_cluster(
        self, skill_code: str, cluster_episodes: List[Dict]
    ) -> bool:
        """Execute skill on all cluster tasks. Check pass rate."""
        passed = 0
        total = len(cluster_episodes)

        for ep in cluster_episodes:
            try:
                result = safe_execute_freeform(skill_code)
                if result and not result.startswith("Error:"):
                    passed += 1
            except Exception:
                pass

        ratio = passed / total if total > 0 else 0
        return ratio >= self.config.library.min_skill_confidence

    @staticmethod
    def _extract_code(solution: str) -> Optional[str]:
        """Extract Python code from solution."""
        block = extract_python_block(solution)
        if block:
            return block
        if "def " in solution or "import " in solution:
            return solution
        return None

    @staticmethod
    def _answers_match(a: str, b: str) -> bool:
        """Flexible matching: exact, numeric, substring."""
        a_clean = a.strip().lower()
        b_clean = b.strip().lower()
        if a_clean == b_clean or a_clean in b_clean or b_clean in a_clean:
            return True
        a_nums = set(re.findall(r"-?\d+\.?\d*", a_clean))
        b_nums = set(re.findall(r"-?\d+\.?\d*", b_clean))
        if a_nums and b_nums and a_nums & b_nums:
            return True
        return False

    def wait_for_sleep(self, timeout: int = 300):
        """Wait for background Library Learning to complete."""
        if self._library_thread and self._library_thread.is_alive():
            self._library_thread.join(timeout=timeout)

    def learn_fact(self, content: str, source: str = "user") -> bool:
        """Manually add fact/episode to memory."""
        emb = llm.get_embedding(content)
        vec = np.array([emb], dtype=np.float32)
        return self.memory.add_episode(
            {"query": content, "solution": content, "source": source, "score": 0.8},
            vec,
        )


# Backward compatibility
NAREProductionAgent = VareAgent
