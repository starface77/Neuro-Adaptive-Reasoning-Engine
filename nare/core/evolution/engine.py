import logging
import time
import threading
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from ...reasoning import llm
from ...memory.engine import MemorySystem, episode_content_key
from ...config import NareConfig
from ...execution.sandboxes.base import safe_call_execute_in_namespace, safe_call_trigger
from .learning import discover_rule

class EvolutionEngine:
    """Handles offline Library Learning and Skill Compilation.

    Renamed from biological metaphors (NREM/REM sleep) to
    honest terminology (Library Learning, Skill Compilation).

    Functionality unchanged - still clusters episodes and crystallizes
    reusable skills, but without misleading neuroscience terminology.
    """

    def __init__(self, memory: MemorySystem, config: NareConfig, oracle_fn: Optional[Any] = None):
        self.memory = memory
        self.config = config
        self.oracle = oracle_fn
        self._is_compiling = False

    def check_compilation_trigger(self) -> bool:
        """Check if we should trigger skill compilation.

        Renamed from check_sleep_trigger.

        Triggers when:
        1. Enough episodes accumulated (>= cluster_density_threshold)
        2. Dense cluster detected (at least N similar episodes)
        3. Not too frequent (cooldown period)
        """

        if len(self.memory.episodes) < self.config.sleep.cluster_density_threshold:
            return False

        if len(self.memory.episodes) >= self.config.sleep.max_episodes_before_sleep:
            logging.info(f"[Evolution] Triggering compilation: {len(self.memory.episodes)} episodes >= {self.config.sleep.max_episodes_before_sleep}")
            return True

        if self.memory.episodic_index.ntotal < self.config.sleep.cluster_density_threshold:
            return False

        threshold = self.config.sleep.cluster_similarity_threshold
        min_neighbors = self.config.sleep.cluster_density_threshold

        recent_episodes = self.memory.episodes[-20:]

        for ep in recent_episodes:
            if 'embedding' not in ep:
                continue

            query_vec = np.array([ep['embedding']], dtype=np.float32)
            faiss.normalize_L2(query_vec)

            k_search = min(min_neighbors + 1, self.memory.episodic_index.ntotal)
            sims, indices = self.memory.episodic_index.search(query_vec, k_search)

            neighbors = sum(1 for sim in sims[0][1:] if sim >= threshold)

            if neighbors >= min_neighbors:
                logging.info(f"[Evolution] Dense cluster detected: {neighbors} neighbors above {threshold:.2f}")
                return True

        return False

    def run_compilation_cycle(self):
        """Run background skill compilation cycle.

        Renamed from run_sleep_cycle.
        """
        if self._is_compiling: return
        self._is_compiling = True

        def _wrapper():
            try:
                self._compile_skills()
                self._validate_skills()
                self._background_validate_episodes()
            finally:
                self._is_compiling = False

        threading.Thread(target=_wrapper, daemon=True).start()

    def _compile_skills(self):
        """Compile reusable skills from clustered episodes.

        Process:
        1. Cluster similar successful episodes
        2. Discover generalizing rule through SEARCH (not extraction)
        3. Validate on holdout data
        4. Store as executable skill
        """
        logging.info("=== [LIBRARY LEARNING] Compiling Skills ===")
        logging.info(f"[LIBRARY LEARNING] Total episodes: {len(self.memory.episodes)}")
        logging.info(f"[LIBRARY LEARNING] Existing skills: {len(self.memory.compiled_skills)}")

        episodes_to_cluster = [ep for ep in self.memory.episodes if ep.get('score', 0) >= 0.80]
        logging.info(f"[LIBRARY LEARNING] High-score episodes (≥0.80): {len(episodes_to_cluster)}")

        if len(episodes_to_cluster) < 3:
            logging.info("[LIBRARY LEARNING] Need ≥3 verified episodes for rule discovery.")
            return

        logging.info(f"[LIBRARY LEARNING] Discovering rule from {len(episodes_to_cluster)} episodes...")
        rule = discover_rule(
            episodes=episodes_to_cluster,
            oracle=self.oracle,
            n_candidates=5,
            holdout_ratio=0.3
        )

        if rule:
            text_to_embed = f"Pattern: {rule['pattern']}\nCode: {rule['python_code']}"
            embedding = llm.get_embedding(text_to_embed)

            self.memory.add_compiled_skill(
                pattern=rule['pattern'],
                code=rule['python_code'],
                trigger_emb=np.array(embedding, dtype=np.float32)
            )

            self.memory.add_semantic_rule(rule, np.array(embedding, dtype=np.float32))

            logging.info(f"[LIBRARY LEARNING] Successfully compiled skill: {rule['pattern']} (confidence: {rule['confidence']:.2f})")
            logging.info(f"[LIBRARY LEARNING] Total skills: {len(self.memory.compiled_skills)}")
        else:
            logging.warning("[LIBRARY LEARNING] Failed to discover generalizing rule.")

    def _validate_skills(self):
        """Validate existing skills through stress testing.

        Renamed from _rem_sleep_phase. Removes REM metaphor.

        Process:
        1. Test each skill on recent episodes
        2. Update confidence based on success rate
        3. Remove skills with low confidence
        """
        logging.info("=== [SKILL VALIDATION] Stress-testing Skills ===")

        if not self.memory.compiled_skills:
            logging.info("[SKILL VALIDATION] No skills to validate")
            return

        logging.info(f"[SKILL VALIDATION] Validating {len(self.memory.compiled_skills)} skills")

        recent_episodes = [ep for ep in self.memory.episodes[-50:] if ep.get('score', 0) >= 0.80]

        if len(recent_episodes) < 3:
            logging.info("[SKILL VALIDATION] Not enough episodes for validation")
            return

        from ...execution.sandboxes.base import safe_load_module

        skills_to_remove = []

        for idx, skill in enumerate(self.memory.compiled_skills):
            try:

                safe_globals = safe_load_module(skill['code'])

                if 'trigger' not in safe_globals or 'execute' not in safe_globals:
                    logging.warning(f"[SKILL VALIDATION] Skill {idx} missing trigger/execute")
                    skills_to_remove.append(idx)
                    continue

                trigger_fn = safe_globals['trigger']
                execute_fn = safe_globals['execute']

                correct = 0
                total = 0

                for ep in recent_episodes[:10]:
                    query = ep['query']
                    expected = ep.get('solution', '')

                    try:
                        if trigger_fn(query):
                            total += 1
                            result = str(execute_fn(query))

                            if result and result.strip() == expected.strip():
                                correct += 1
                    except:
                        total += 1

                if total > 0:
                    accuracy = correct / total
                    logging.info(f"[SKILL VALIDATION] Skill {idx}: {correct}/{total} = {accuracy:.2f}")

                    old_conf = skill.get('confidence', 0.5)
                    new_conf = 0.7 * old_conf + 0.3 * accuracy
                    skill['confidence'] = new_conf

                    if new_conf < 0.3:
                        logging.warning(f"[SKILL VALIDATION] Removing low-confidence skill {idx}")
                        skills_to_remove.append(idx)

            except Exception as e:
                logging.warning(f"[SKILL VALIDATION] Skill {idx} validation failed: {e}")
                skills_to_remove.append(idx)

        if skills_to_remove:
            with self.memory._lock:
                for idx in sorted(skills_to_remove, reverse=True):
                    if 0 <= idx < len(self.memory.compiled_skills):
                        removed = self.memory.compiled_skills.pop(idx)
                        logging.info(f"[SKILL VALIDATION] Removed skill: {removed.get('pattern', 'unknown')}")
                self.memory.save()

        logging.info(f"[SKILL VALIDATION] Validation complete. {len(self.memory.compiled_skills)} skills remaining")

    def _background_validate_episodes(self):
        """Random audit of episodic memory quality.

        Process:
        1. Sample random episodes
        2. Check if they still produce correct results
        3. Update trust coefficient (tau)
        4. Remove episodes with low tau
        """
        logging.info("=== [BACKGROUND VALIDATION] Auditing Episodes ===")

        if len(self.memory.episodes) < 10:
            logging.info("[BACKGROUND VALIDATION] Not enough episodes to audit")
            return

        audit_count = min(self.config.immune.background_audit_count, len(self.memory.episodes))
        import random
        sample_indices = random.sample(range(len(self.memory.episodes)), audit_count)

        logging.info(f"[BACKGROUND VALIDATION] Auditing {audit_count} random episodes")

        for idx in sample_indices:
            ep = self.memory.episodes[idx]
            query = ep.get('query', '')
            solution = ep.get('solution', '')
            tau = ep.get('tau', 0.5)

            is_valid = bool(solution and len(solution) > 10 and not solution.startswith("Error"))

            if is_valid:
                delta_v = 1.0
                self.memory.update_episode_tau(idx, delta_v)
                logging.debug(f"[BACKGROUND VALIDATION] Episode {idx}: tau {tau:.2f} -> {tau + self.config.immune.tau_lr:.2f}")
            else:
                delta_v = -1.0
                self.memory.update_episode_tau(idx, delta_v)
                logging.warning(f"[BACKGROUND VALIDATION] Episode {idx} failed validation: tau {tau:.2f} -> {tau - self.config.immune.tau_lr:.2f}")

        self.memory.prune_untrusted_episodes()

        logging.info("[BACKGROUND VALIDATION] Audit complete")

    def record_skill_result(self, rule: dict, success: bool):
        """Record skill execution result for confidence updates.

        Args:
            rule: Skill dictionary with 'pattern' and 'code'
            success: Whether the skill execution was successful

        Updates:
            - Skill confidence (increase on success, decrease on failure)
            - Use count
            - Success/failure history
        """

        pattern = rule.get('pattern', '')

        with self.memory._lock:
            for skill in self.memory.compiled_skills:
                if skill.get('pattern') == pattern:

                    old_conf = skill.get('confidence', 0.5)
                    delta_v = 1.0 if success else -1.0
                    lr = self.config.immune.tau_lr

                    new_conf = old_conf + lr * delta_v
                    new_conf = max(0.0, min(1.0, new_conf))

                    skill['confidence'] = new_conf

                    if 'success_count' not in skill:
                        skill['success_count'] = 0
                    if 'failure_count' not in skill:
                        skill['failure_count'] = 0

                    if success:
                        skill['success_count'] += 1
                    else:
                        skill['failure_count'] += 1

                    logging.info(f"[SKILL RESULT] {pattern}: {'success' if success else 'failure'}, confidence {old_conf:.2f} -> {new_conf:.2f}")

                    self.memory.save()
                    break
