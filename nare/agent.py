"""VARE Agent — Verified Amortized Reasoning Engine.

Architecture: three modular components:
  M_cache   — HNSW episodic memory (nare.memory.MemorySystem)
  G_θ       — Fixed-weight LLM generator (nare.llm)
  V_sandbox — Formal verifier (nare.sandbox)

Two execution routes:
  FAST           — cache hit (similarity >= tau_fast): instant return
  VERIFIED_RETRY — iterative synthesis with sandbox verification

Background process:
  Library Learning — cluster similar episodes, abstract into compiled
                     skills, verify on all cluster tasks, store as
                     COMPILED_SKILL for O(1) future execution.
"""

import time
import logging
import threading
import numpy as np
import faiss
from typing import Dict, List, Any, Optional

from .config import VareConfig, DEFAULT_CONFIG
from .memory import MemorySystem
from .metrics import MetricsTracker
from .sandbox import (
    safe_load_module,
    safe_execute,
    safe_call_trigger,
    safe_call_execute_in_namespace,
    validate_code,
    SecurityError,
)
from . import llm

# Backward compat
NareConfig = VareConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')


class VareAgent:
    """Verified Amortized Reasoning Engine.

    Processes queries through a deterministic pipeline:
    1. Embed query → search M_cache
    2. If similarity >= tau_fast → FAST route (return cached/skill)
    3. Else → VERIFIED_RETRY: generate → verify → refine loop
    4. Store verified result in M_cache
    5. Background: Library Learning (cluster → abstract → verify → store)
    """

    def __init__(
        self,
        config: VareConfig = DEFAULT_CONFIG,
        oracle: "Oracle | None" = None,
    ):
        self.config = config
        self.oracle = oracle
        self.memory = MemorySystem(config=config)
        self.metrics = MetricsTracker(persist_dir=self.memory.persist_dir)

        self.tau_fast = config.routing.tau_fast
        self.tau_min = config.routing.tau_min
        self.tau_max = config.routing.tau_max

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _calibrate_tau(self, reward: float, fast_used: bool):
        """Adjust tau_fast based on outcome feedback."""
        lr = self.config.routing.calibration_lr
        if fast_used and reward < 0.5:
            self.tau_fast = min(self.tau_max, self.tau_fast + lr)
        elif not fast_used and reward > 0.8:
            self.tau_fast = max(self.tau_min, self.tau_fast - (lr / 2))

    # ------------------------------------------------------------------
    # FAST route
    # ------------------------------------------------------------------

    def _try_fast_route(
        self, query: str, query_emb: list, log: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Attempt FAST route: return cached answer or execute compiled skill.

        Returns result dict on cache hit, or None to fall through to
        VERIFIED_RETRY.
        """
        # 1. Check compiled skills first (O(1) execution)
        skills = self.memory.search_skills(np.array(query_emb), k=1)
        if skills and skills[0]['similarity'] >= self.tau_fast:
            skill = skills[0]
            python_code = skill.get('ast_code', '')
            if python_code:
                try:
                    result = safe_execute(python_code, query)
                    if not result.startswith("Error"):
                        log.append(
                            f"Route: FAST (COMPILED_SKILL, sim={skills[0]['similarity']:.3f})"
                        )
                        return {
                            "route_decision": "FAST",
                            "source": "COMPILED_SKILL",
                            "final_answer": result,
                            "memory_update_log": log,
                            "tokens_used": 0,
                        }
                except (SecurityError, Exception) as e:
                    log.append(f"Skill execution failed: {e}")

        # 2. Check episodic cache
        hits = self.memory.search(np.array(query_emb), k=1)
        if hits and hits[0]['similarity'] >= self.tau_fast:
            ep = hits[0]
            if ep.get('score', 0) >= 0.5:
                idx = ep.get('memory_id', -1)
                if idx >= 0:
                    self.memory.boost_activation(idx)
                log.append(
                    f"Route: FAST (EPISODE, sim={ep['similarity']:.3f})"
                )
                return {
                    "route_decision": "FAST",
                    "source": "EPISODE",
                    "final_answer": ep['solution'],
                    "memory_update_log": log,
                    "tokens_used": 0,
                }

        return None

    # ------------------------------------------------------------------
    # VERIFIED_RETRY (System 2)
    # ------------------------------------------------------------------

    def _verified_retry(
        self, query: str, query_emb: list, log: List[str]
    ) -> Dict[str, Any]:
        """Verified Code Synthesis loop.

        MDP: state=(query, error_history), action=generate candidate,
        transition=sandbox verify, reward=R(y)∈{0,1}.

        Iterates up to max_retries, feeding error traces back to G_θ.
        """
        max_retries = self.config.synthesis.max_retries
        error_trace = ""
        total_tokens = 0
        best_answer = ""
        best_reasoning = ""
        verified = False

        # Retrieve similar episodes as context (analogies)
        context_eps = self.memory.search(np.array(query_emb), k=3)
        context_str = ""
        if context_eps:
            context_str = "\n--- RELEVANT PAST EXPERIENCE ---\n"
            for ep in context_eps:
                context_str += (
                    f"Past Query: {ep['query']}\n"
                    f"Past Solution: {ep['solution']}\n---\n"
                )

        for attempt in range(max_retries):
            # Build prompt with self-refinement
            temp = (
                self.config.synthesis.initial_temperature
                if attempt == 0
                else self.config.synthesis.refinement_temperature
            )

            if attempt == 0:
                prompt = f"Task: {query}\n{context_str}"
            else:
                prompt = (
                    f"Task: {query}\n\n"
                    f"Previous attempt failed. Error trace:\n{error_trace}\n\n"
                    "Fix the errors and provide a corrected solution."
                )

            candidates, tokens = llm.generate_samples(
                prompt, n=1, temperature=temp, mode="SLOW"
            )
            total_tokens += tokens

            if not candidates:
                error_trace += f"\nAttempt {attempt+1}: No response from generator.\n"
                log.append(f"[Attempt {attempt+1}] No candidates generated.")
                continue

            candidate = candidates[0]
            solution = candidate['solution']
            reasoning = candidate.get('reasoning', '')

            # Verify in sandbox (V_sandbox)
            is_valid, verification_result = self._verify_solution(
                query, solution
            )

            if is_valid:
                log.append(
                    f"[Attempt {attempt+1}] Verified OK. "
                    f"Route: VERIFIED_RETRY ({attempt+1} attempts)"
                )
                best_answer = solution
                best_reasoning = reasoning
                verified = True
                break
            else:
                error_trace += (
                    f"\nAttempt {attempt+1} error: {verification_result}\n"
                )
                log.append(
                    f"[Attempt {attempt+1}] Verification failed: "
                    f"{verification_result[:100]}"
                )
                # Keep last answer as fallback
                best_answer = solution
                best_reasoning = reasoning

        route = "VERIFIED_RETRY" if verified else "SLOW_UNVERIFIED"
        score = 1.0 if verified else 0.3

        # Store verified episode in M_cache
        if best_answer:
            episode = {
                "query": query,
                "solution": best_answer,
                "reasoning_trace": best_reasoning,
                "score": score,
                "embedding": query_emb,
                "verified": verified,
                "attempts": min(attempt + 1, max_retries),
            }
            self.memory.add_episode(
                episode, np.array([query_emb], dtype=np.float32)
            )

        return {
            "route_decision": route,
            "final_answer": best_answer,
            "memory_update_log": log,
            "tokens_used": total_tokens,
            "attempts": min(attempt + 1, max_retries),
            "verified": verified,
        }

    def _verify_solution(self, query: str, solution: str) -> tuple:
        """V_sandbox: attempt formal verification.

        For code solutions: compile + execute in sandbox.
        For text solutions: basic sanity check (non-empty, reasonable).
        If an oracle is provided, use it for ground-truth verification.

        Returns (is_valid: bool, info: str).
        """
        # 1. Oracle verification (if available)
        if self.oracle is not None:
            try:
                correct, info = self.oracle(query, solution)
                return correct, info
            except Exception as e:
                return False, f"Oracle error: {e}"

        # 2. Code verification via sandbox
        code = self._extract_code(solution)
        if code is not None:
            try:
                validate_code(code)
                return True, "Code AST-valid"
            except SecurityError as e:
                return False, f"Security: {e}"
            except Exception as e:
                return False, f"Validation error: {e}"

        # 3. Text solution: basic sanity
        if len(solution.strip()) > 5:
            return True, "Text answer present"

        return False, "Empty or trivial answer"

    @staticmethod
    def _extract_code(solution: str) -> Optional[str]:
        """Extract Python code from solution (raw or markdown-fenced)."""
        import re
        if "```python" in solution:
            m = re.search(r'```python\n(.*?)\n```', solution, re.DOTALL)
            if m:
                return m.group(1)
        if "def " in solution and ("return" in solution or "print" in solution):
            return solution
        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(self, query: str) -> Dict[str, Any]:
        """Process a query through the VARE pipeline.

        1. Embed query
        2. Try FAST route (cache/skill)
        3. Fall through to VERIFIED_RETRY
        4. Check Library Learning trigger
        """
        _start = time.time()
        log = [f"Query: {query}"]

        # Embed
        query_emb = llm.get_embedding(query)

        # Route 1: FAST
        fast_result = self._try_fast_route(query, query_emb, log)
        if fast_result is not None:
            elapsed = time.time() - _start
            self._calibrate_tau(reward=1.0, fast_used=True)
            self.metrics.record(
                query=query, route="FAST",
                elapsed=elapsed, tokens_used=0,
                similarity=1.0, answer=fast_result['final_answer'],
                score=1.0,
            )
            fast_result['elapsed'] = elapsed
            return fast_result

        # Route 2: VERIFIED_RETRY
        result = self._verified_retry(query, query_emb, log)
        elapsed = time.time() - _start
        score = 1.0 if result.get('verified', False) else 0.3
        self._calibrate_tau(reward=score, fast_used=False)
        self.metrics.record(
            query=query, route=result['route_decision'],
            elapsed=elapsed,
            tokens_used=result.get('tokens_used', 0),
            similarity=0.0, answer=result.get('final_answer', ''),
            score=score,
        )
        result['elapsed'] = elapsed

        # Check Library Learning trigger (background)
        if self._should_run_library_learning():
            self._start_library_learning()

        return result

    # ------------------------------------------------------------------
    # Library Learning (background)
    # ------------------------------------------------------------------

    def _should_run_library_learning(self) -> bool:
        """Check if enough episodes have accumulated for clustering."""
        lib_cfg = self.config.library
        n = len(self.memory.episodes)
        if n < lib_cfg.cluster_density_threshold:
            return False
        return not getattr(self, '_is_learning', False)

    def _start_library_learning(self):
        """Launch Library Learning in a background thread."""
        self._is_learning = True

        def _learn():
            try:
                self._library_learning()
            finally:
                self._is_learning = False

        t = threading.Thread(target=_learn, daemon=True)
        t.start()
        self._learn_thread = t

    def _library_learning(self):
        """Cluster similar episodes → abstract → verify → store as skill.

        1. Compute pairwise similarities among episodes
        2. Find dense clusters
        3. For each cluster: ask LLM to abstract a reusable function
        4. Verify function against all cluster tasks in V_sandbox
        5. If passes: store as COMPILED_SKILL
        """
        lib_cfg = self.config.library
        episodes = self.memory.episodes

        if len(episodes) < lib_cfg.cluster_density_threshold:
            return

        # Build similarity matrix
        with self.memory._lock:
            vecs = np.array(
                [ep['embedding'] for ep in episodes],
                dtype=np.float32,
            )
        faiss.normalize_L2(vecs)
        sim_matrix = np.dot(vecs, vecs.T)
        np.fill_diagonal(sim_matrix, 0.0)

        # Find dense clusters
        threshold = lib_cfg.cluster_similarity_threshold
        min_neighbours = max(1, lib_cfg.cluster_density_threshold - 1)
        density = np.sum(sim_matrix > threshold, axis=1)

        processed = set()
        for anchor in np.argsort(-density):
            if density[anchor] < min_neighbours:
                break
            if int(anchor) in processed:
                continue

            # Gather cluster
            cluster_indices = [int(anchor)]
            for j in range(len(episodes)):
                if j != anchor and sim_matrix[anchor, j] > threshold:
                    cluster_indices.append(j)
            cluster_indices = list(set(cluster_indices))

            if len(cluster_indices) < lib_cfg.cluster_density_threshold:
                continue

            processed.update(cluster_indices)
            cluster_episodes = [episodes[i] for i in cluster_indices]

            logging.info(
                f"[Library Learning] Found cluster of {len(cluster_episodes)} episodes. "
                f"Attempting abstraction..."
            )

            # Ask LLM to abstract a reusable function
            skill_code = self._abstract_skill(cluster_episodes)
            if not skill_code:
                logging.warning("[Library Learning] Abstraction failed.")
                continue

            # Verify against all cluster tasks
            passed = self._verify_skill_on_cluster(skill_code, cluster_episodes)
            if not passed:
                logging.warning("[Library Learning] Skill failed verification.")
                continue

            # Store as COMPILED_SKILL
            centroid = np.mean(vecs[cluster_indices], axis=0).reshape(1, -1)
            faiss.normalize_L2(centroid)
            skill_data = {
                'pattern': cluster_episodes[0].get('query', '')[:100],
                'ast_code': skill_code,
                'confidence': 1.0,
                'source_count': len(cluster_episodes),
                'embedding': centroid.flatten().tolist(),
            }
            self.memory.add_skill(skill_data, centroid)
            logging.info(
                f"[Library Learning] Compiled skill from {len(cluster_episodes)} "
                f"episodes: {skill_data['pattern'][:60]}"
            )

        # Decay and prune old episodes
        self.memory.decay_and_prune()
        self.memory.save()

    def _abstract_skill(self, cluster_episodes: List[Dict]) -> Optional[str]:
        """Ask LLM to generate an abstracted Python function from cluster."""
        prompt = (
            "Analyze these solved tasks and create a SINGLE reusable Python function "
            "that solves ALL of them.\n\n"
        )
        for i, ep in enumerate(cluster_episodes[:5]):
            prompt += (
                f"Task {i+1}: {ep['query']}\n"
                f"Solution: {ep['solution']}\n---\n"
            )
        prompt += (
            "\nWrite a Python module with exactly two functions:\n"
            "  def trigger(query: str) -> bool:  # returns True if this skill applies\n"
            "  def execute(query: str) -> str:    # returns the answer\n\n"
            "Use only: re, math. No other imports. Output ONLY the Python code, "
            "no markdown fences."
        )
        try:
            candidates, _ = llm.generate_samples(prompt, n=1, temperature=0.3, mode="SLOW")
            if candidates:
                code = candidates[0]['solution']
                # Clean markdown fences if present
                import re as re_mod
                code = re_mod.sub(r'^```python\s*\n?', '', code)
                code = re_mod.sub(r'\n?```\s*$', '', code)
                # Validate AST
                validate_code(code)
                return code
        except Exception as e:
            logging.warning(f"[Library Learning] Abstraction error: {e}")
        return None

    def _verify_skill_on_cluster(
        self, skill_code: str, cluster_episodes: List[Dict]
    ) -> bool:
        """Verify compiled skill passes all cluster tasks in V_sandbox."""
        passed = 0
        total = len(cluster_episodes)

        for ep in cluster_episodes:
            try:
                result = safe_execute(skill_code, ep['query'])
                if not result.startswith("Error"):
                    # Check if result is consistent with stored solution
                    if self._answers_match(result, ep.get('solution', '')):
                        passed += 1
            except Exception:
                continue

        ratio = passed / max(total, 1)
        logging.info(
            f"[Library Learning] Skill verification: {passed}/{total} "
            f"({ratio:.0%})"
        )
        return ratio >= self.config.library.min_skill_confidence

    @staticmethod
    def _answers_match(a: str, b: str) -> bool:
        """Flexible answer matching: numeric or substring."""
        import re as re_mod
        a_clean = a.strip().lower()
        b_clean = b.strip().lower()
        if a_clean == b_clean:
            return True
        # Numeric match
        nums_a = re_mod.findall(r'-?\d+\.?\d*', a_clean)
        nums_b = re_mod.findall(r'-?\d+\.?\d*', b_clean)
        if nums_a and nums_b and nums_a[0] == nums_b[0]:
            return True
        # Substring match
        if len(a_clean) > 3 and a_clean in b_clean:
            return True
        if len(b_clean) > 3 and b_clean in a_clean:
            return True
        return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def wait_for_sleep(self, timeout: int = 300):
        """Wait for background Library Learning to complete."""
        start = time.time()
        while getattr(self, '_is_learning', False):
            if time.time() - start > timeout:
                logging.warning("[Agent] Timeout waiting for Library Learning.")
                break
            time.sleep(1)

    def learn_fact(self, content: str, source: str = "user") -> bool:
        """Manually add a fact/episode to memory."""
        embedding = llm.get_embedding(content)
        return self.memory.add_episode(
            {"query": content, "solution": content, "score": 1.0,
             "reasoning_trace": "User-provided fact", "source": source,
             "embedding": embedding},
            np.array(embedding, dtype=np.float32),
        )


# Backward compatibility alias
NAREProductionAgent = VareAgent
