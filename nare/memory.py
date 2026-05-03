"""M_cache — HNSW-backed episodic memory with activation-based forgetting.

Each cell stores:
  - embedding: query vector (3072-dim Gemini)
  - query: original query text
  - solution: verified answer
  - reasoning_trace: LLM reasoning chain
  - score: verification result (0 or 1)
  - activation: strength/retention metric (Ebbinghaus decay)
  - timestamp: creation time
  - last_used: last access time
  - type: "EPISODE" or "COMPILED_SKILL"
  - ast_code: (skills only) Python source code

Thread-safe via RLock on all reads/writes.
"""

import time
import faiss
import numpy as np
import json
import os
import logging
import threading
from typing import List, Dict, Any, Optional

from .config import DEFAULT_CONFIG, VareConfig

NareConfig = VareConfig  # backward compat


def _make_hnsw_index(dim: int) -> faiss.Index:
    """Create an HNSW index for O(log N) approximate nearest neighbour."""
    M = 32
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 64
    index.hnsw.efSearch = 32
    return index


class MemorySystem:
    """Unified episodic + compiled-skill memory (M_cache).

    Two storage types in a single HNSW index:
      - EPISODE: raw (query, solution) pairs from verified synthesis
      - COMPILED_SKILL: abstracted Python functions from Library Learning

    Activation-based forgetting removes stale entries.
    """

    def __init__(
        self,
        embedding_dim: int = 3072,
        persist_dir: str = "memory_store",
        config: VareConfig = DEFAULT_CONFIG,
    ):
        self.embedding_dim = embedding_dim
        self.persist_dir = persist_dir
        self.config = config
        self._lock = threading.RLock()

        self.episodic_index = _make_hnsw_index(embedding_dim)
        self.episodes: List[Dict[str, Any]] = []

        # Compiled skills have their own brute-force index (small N).
        self.skill_index = faiss.IndexFlatIP(embedding_dim)
        self.skills: List[Dict[str, Any]] = []

        os.makedirs(self.persist_dir, exist_ok=True)
        self.load()

    # ------------------------------------------------------------------
    # Episode CRUD
    # ------------------------------------------------------------------

    def add_episode(self, episode_data: Dict[str, Any], embedding: np.ndarray) -> bool:
        """Store a verified episode. Deduplicates by cosine threshold."""
        vector = np.array(embedding, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)
        threshold = self.config.memory.dedup_threshold

        with self._lock:
            if self.episodic_index.ntotal > 0:
                sims, _ = self.episodic_index.search(vector, 1)
                if sims[0][0] > threshold:
                    logging.info("[Memory] Dedup: identical episode exists.")
                    return False

            episode_data.setdefault('timestamp', time.time())
            episode_data.setdefault('last_used', time.time())
            episode_data.setdefault('activation', 1.0)
            episode_data.setdefault('type', 'EPISODE')
            self.episodic_index.add(vector)
            self.episodes.append(episode_data)
            self.save()
            return True

    def search(self, query_embedding: np.ndarray, k: int = 1) -> List[Dict[str, Any]]:
        """Search M_cache for nearest episodes. Returns list with 'similarity'."""
        with self._lock:
            if self.episodic_index.ntotal == 0:
                return []

            vector = np.array(query_embedding, dtype=np.float32)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            faiss.normalize_L2(vector)
            k_search = min(k + 5, self.episodic_index.ntotal)
            sims, indices = self.episodic_index.search(vector, k_search)

            results = []
            for sim, idx in zip(sims[0], indices[0]):
                if idx != -1 and idx < len(self.episodes):
                    res = self.episodes[idx].copy()
                    res['similarity'] = float(sim)
                    res['memory_id'] = int(idx)
                    results.append(res)
                    if len(results) >= k:
                        break
            return results

    def boost_activation(self, idx: int):
        """Boost activation on use (Ebbinghaus reinforcement)."""
        with self._lock:
            if 0 <= idx < len(self.episodes):
                self.episodes[idx]['last_used'] = time.time()
                self.episodes[idx]['activation'] = (
                    self.episodes[idx].get('activation', 1.0)
                    + self.config.memory.strength_boost_on_use
                )

    # ------------------------------------------------------------------
    # Compiled Skills
    # ------------------------------------------------------------------

    def add_skill(self, skill_data: Dict[str, Any], embedding: np.ndarray) -> bool:
        """Store a compiled skill (COMPILED_SKILL type)."""
        vector = np.array(embedding, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)

        with self._lock:
            if self.skill_index.ntotal > 0:
                sims, _ = self.skill_index.search(vector, 1)
                if sims[0][0] > self.config.library.skill_dedup_threshold:
                    # Update existing skill if new one is better
                    idx = int(_[0][0])
                    if idx < len(self.skills):
                        old_conf = self.skills[idx].get('confidence', 0)
                        new_conf = skill_data.get('confidence', 0)
                        if new_conf > old_conf:
                            self.skills[idx] = skill_data
                            logging.info("[Memory] Upgraded existing skill.")
                            self.save()
                    return False

            skill_data.setdefault('type', 'COMPILED_SKILL')
            skill_data.setdefault('timestamp', time.time())
            self.skill_index.add(vector)
            self.skills.append(skill_data)
            self.save()
            return True

    def search_skills(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Search compiled skills by embedding similarity."""
        with self._lock:
            if self.skill_index.ntotal == 0:
                return []
            vector = np.array(query_embedding, dtype=np.float32)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            faiss.normalize_L2(vector)
            k_search = min(k, self.skill_index.ntotal)
            sims, indices = self.skill_index.search(vector, k_search)
            results = []
            for sim, idx in zip(sims[0], indices[0]):
                if idx != -1 and idx < len(self.skills):
                    res = self.skills[idx].copy()
                    res['similarity'] = float(sim)
                    results.append(res)
            return results

    # ------------------------------------------------------------------
    # Activation decay & pruning (Ebbinghaus)
    # ------------------------------------------------------------------

    def decay_and_prune(self):
        """Apply exponential activation decay and prune weak entries."""
        S = self.config.memory.strength_decay_constant
        tau_prune = self.config.memory.tau_prune
        now = time.time()

        with self._lock:
            surviving = []
            for ep in self.episodes:
                dt = now - ep.get('last_used', ep.get('timestamp', now))
                ep['activation'] = ep.get('activation', 1.0) * np.exp(-dt / S)
                if ep['activation'] >= tau_prune:
                    surviving.append(ep)

            removed = len(self.episodes) - len(surviving)
            if removed > 0:
                logging.info(f"[Memory] Pruned {removed} faded episodes.")
                self.episodes = surviving
                self._rebuild_episodic_index()
                self.save()

        # Also enforce max size
        with self._lock:
            max_eps = self.config.memory.max_episodes
            if len(self.episodes) > max_eps:
                # Sort by activation, keep top max_eps
                self.episodes.sort(key=lambda x: x.get('activation', 0), reverse=True)
                self.episodes = self.episodes[:max_eps]
                self._rebuild_episodic_index()
                self.save()

    def _rebuild_episodic_index(self):
        """Rebuild HNSW index from current episodes (caller must hold lock)."""
        self.episodic_index = _make_hnsw_index(self.embedding_dim)
        if self.episodes:
            vecs = np.array(
                [ep['embedding'] for ep in self.episodes],
                dtype=np.float32,
            )
            faiss.normalize_L2(vecs)
            self.episodic_index.add(vecs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        with self._lock:
            path = os.path.join(self.persist_dir, "episodes.json")
            safe = []
            for ep in self.episodes:
                entry = {k: v for k, v in ep.items() if k != 'embedding'}
                entry['embedding'] = (
                    ep['embedding'].tolist()
                    if isinstance(ep.get('embedding'), np.ndarray)
                    else ep.get('embedding', [])
                )
                safe.append(entry)
            with open(path, 'w') as f:
                json.dump(safe, f, default=str)

            skill_path = os.path.join(self.persist_dir, "skills.json")
            skill_safe = []
            for sk in self.skills:
                entry = {k: v for k, v in sk.items() if k != 'embedding'}
                entry['embedding'] = (
                    sk['embedding'].tolist()
                    if isinstance(sk.get('embedding'), np.ndarray)
                    else sk.get('embedding', [])
                )
                skill_safe.append(entry)
            with open(skill_path, 'w') as f:
                json.dump(skill_safe, f, default=str)

    def load(self):
        with self._lock:
            # Load episodes
            path = os.path.join(self.persist_dir, "episodes.json")
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    self.episodes = data
                    if self.episodes:
                        vecs = np.array(
                            [ep['embedding'] for ep in self.episodes],
                            dtype=np.float32,
                        )
                        faiss.normalize_L2(vecs)
                        self.episodic_index.add(vecs)
                    logging.info(f"[Memory] Loaded {len(self.episodes)} episodes.")
                except Exception as e:
                    logging.warning(f"[Memory] Failed to load episodes: {e}")

            # Load skills
            skill_path = os.path.join(self.persist_dir, "skills.json")
            if os.path.exists(skill_path):
                try:
                    with open(skill_path, 'r') as f:
                        data = json.load(f)
                    self.skills = data
                    if self.skills:
                        vecs = np.array(
                            [sk['embedding'] for sk in self.skills],
                            dtype=np.float32,
                        )
                        faiss.normalize_L2(vecs)
                        self.skill_index.add(vecs)
                    logging.info(f"[Memory] Loaded {len(self.skills)} skills.")
                except Exception as e:
                    logging.warning(f"[Memory] Failed to load skills: {e}")
