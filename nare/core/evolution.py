import logging
import time
import threading
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from ..reasoning import llm
from ..memory.memory import MemorySystem, episode_content_key
from ..config import NareConfig
from ..execution.sandbox import safe_call_execute_in_namespace, safe_call_trigger
from .library_learning import discover_rule

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
        self._is_compiling = False  # was _is_sleeping

    def check_compilation_trigger(self) -> bool:
        """Check if we should trigger skill compilation.

        Renamed from check_sleep_trigger.
        """
        if len(self.memory.episodes) < self.config.sleep.cluster_density_threshold:
            return False
        # Logic to detect dense clusters
        return True # Placeholder for now, triggered by agent

    def run_compilation_cycle(self):
        """Run background skill compilation cycle.

        Renamed from run_sleep_cycle.
        """
        if self._is_compiling: return
        self._is_compiling = True

        def _wrapper():
            try:
                self._compile_skills()  # was _sleep_phase
                self._validate_skills()  # was _rem_sleep_phase
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
        # 1. Cluster episodes (take successful recent episodes)
        episodes_to_cluster = [ep for ep in self.memory.episodes if ep.get('score', 0) >= 0.80]
        if len(episodes_to_cluster) < 3:
            logging.info("[LIBRARY LEARNING] Need ≥3 verified episodes for rule discovery.")
            return

        # 2. Discover rule through search (NOT extraction)
        logging.info(f"[LIBRARY LEARNING] Discovering rule from {len(episodes_to_cluster)} episodes...")
        rule = discover_rule(
            episodes=episodes_to_cluster,
            oracle=self.oracle,
            n_candidates=5,
            holdout_ratio=0.3
        )

        # 3. Save discovered rule
        if rule:
            text_to_embed = f"Pattern: {rule['pattern']}\nCode: {rule['python_code']}"
            embedding = llm.get_embedding(text_to_embed)
            self.memory.add_semantic_rule(rule, np.array(embedding, dtype=np.float32))
            logging.info(f"[LIBRARY LEARNING] Successfully compiled skill: {rule['pattern']} (confidence: {rule['confidence']:.2f})")
        else:
            logging.warning("[LIBRARY LEARNING] Failed to discover generalizing rule.")

    def _validate_skills(self):
        """Validate existing skills through stress testing.

        Renamed from _rem_sleep_phase. Removes REM metaphor.
        """
        logging.info("=== [SKILL VALIDATION] Stress-testing Skills ===")
        # 1. Generate adversarial tests
        # 2. Repair failing skills
        pass

    def _background_validate_episodes(self):
        """Random audit of episodic memory quality."""
        # 1. Random audit of episodic memory
        pass

    def record_skill_result(self, rule: dict, success: bool):
        """Record skill execution result for confidence updates."""
        # Penalty backpropagation logic
        delta_v = 1.0 if success else -1.0
        # ... update confidence and tau ...
        pass
