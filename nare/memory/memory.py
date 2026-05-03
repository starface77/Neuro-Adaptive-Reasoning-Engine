"""M_cache — HNSW-backed episodic memory with activation-based forgetting.

Unified index for EPISODE and COMPILED_SKILL types.
O(log N) approximate nearest neighbour via HNSW.
"""

import time
import faiss
import numpy as np
import json
import os
import logging
import threading
from typing import List, Dict, Any, Optional

from ..config import DEFAULT_CONFIG, VareConfig


def _make_hnsw_index(dim: int) -> faiss.Index:
    """O(log N) approximate nearest neighbor search."""
    M = 32
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 64
    index.hnsw.efSearch = 32
    return index


class MemorySystem:
    """Episodic memory + compiled skills with HNSW index.

    Thread-safe via RLock. Supports activation decay (Ebbinghaus),
    deduplication, and capacity-based pruning.
    """

    def __init__(
        self,
        embedding_dim: int = 3072,
        persist_dir: Optional[str] = None,
        config: VareConfig = DEFAULT_CONFIG,
    ):
        self.embedding_dim = embedding_dim
        self.persist_dir = (
            persist_dir
            or os.environ.get("NARE_MEMORY_DIR")
            or "memory_store"
        )
        self.config = config
        self._lock = threading.RLock()

        # Episodic Memory — HNSW index
        self.episodic_index = _make_hnsw_index(embedding_dim)
        self.episodes: List[Dict[str, Any]] = []

        # Compiled Skills — FlatIP index (small N)
        self.skill_index = faiss.IndexFlatIP(embedding_dim)
        self.skills: List[Dict[str, Any]] = []

        os.makedirs(self.persist_dir, exist_ok=True)
        self.load()

    def add_episode(self, episode_data: Dict[str, Any], embedding: np.ndarray) -> bool:
        """Store verified episode. Dedup by cosine > threshold. Return True if added."""
        vector = np.array(embedding, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)

        threshold = self.config.memory.dedup_threshold

        with self._lock:
            if self.episodic_index.ntotal > 0:
                sims, _indices = self.episodic_index.search(vector, 1)
                if sims[0][0] > threshold:
                    logging.info("[Memory] Deduplication triggered.")
                    return False

            episode_data.setdefault('timestamp', time.time())
            episode_data.setdefault('last_used', time.time())
            episode_data.setdefault('activation', 1.0)
            episode_data.setdefault('score', 0.5)
            episode_data.setdefault('type', 'EPISODE')
            episode_data['embedding'] = vector.flatten().tolist()

            self.episodic_index.add(vector)
            self.episodes.append(episode_data)
            self.save()
            return True

    def search(self, query_embedding: np.ndarray, k: int = 1) -> List[Dict[str, Any]]:
        """Search HNSW for nearest episodes."""
        with self._lock:
            if self.episodic_index.ntotal == 0:
                return []

            vector = np.array(query_embedding, dtype=np.float32)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            faiss.normalize_L2(vector)

            k_search = min(k, self.episodic_index.ntotal)
            sims, indices = self.episodic_index.search(vector, k_search)

            results = []
            for sim, idx in zip(sims[0], indices[0]):
                if idx != -1 and 0 <= idx < len(self.episodes):
                    ep = self.episodes[idx].copy()
                    ep['similarity'] = float(sim)
                    ep['memory_id'] = int(idx)
                    results.append(ep)
            return results

    # Backward compat alias
    def retrieve_episodes(self, query_emb: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        return self.search(query_emb, k)

    def boost_activation(self, idx: int):
        """Reinforce on cache hit."""
        with self._lock:
            if 0 <= idx < len(self.episodes):
                self.episodes[idx]['last_used'] = time.time()
                boost = self.config.memory.strength_boost_on_use
                self.episodes[idx]['activation'] = self.episodes[idx].get('activation', 1.0) + boost

    def add_skill(self, skill_data: Dict[str, Any], embedding: np.ndarray) -> bool:
        """Store compiled skill. Check dedup threshold. Return True if added."""
        vector = np.array(embedding, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)

        threshold = self.config.library.skill_dedup_threshold

        with self._lock:
            if self.skill_index.ntotal > 0:
                sims, _indices = self.skill_index.search(vector, 1)
                if sims[0][0] > threshold:
                    logging.info("[Memory] Skill dedup triggered.")
                    return False

            skill_data['embedding'] = vector.flatten().tolist()
            skill_data.setdefault('type', 'COMPILED_SKILL')
            skill_data.setdefault('confidence', 0.0)
            skill_data.setdefault('source_count', 0)

            self.skill_index.add(vector)
            self.skills.append(skill_data)
            self.save()
            return True

    # Backward compat alias
    def add_compiled_skill(self, pattern: str, code: str, trigger_emb: np.ndarray):
        self.add_skill(
            {'pattern': pattern, 'ast_code': code, 'code': code},
            trigger_emb,
        )

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
                if idx != -1 and 0 <= idx < len(self.skills):
                    skill = self.skills[idx].copy()
                    skill['similarity'] = float(sim)
                    skill['skill_id'] = int(idx)
                    results.append(skill)
            return results

    # Backward compat alias
    def retrieve_skills(self, query_emb, k: int = 3) -> List[Dict]:
        return self.search_skills(np.array(query_emb, dtype=np.float32), k)

    def decay_and_prune(self):
        """Apply exponential activation decay. Remove if activation < tau_prune."""
        now = time.time()
        S = self.config.memory.strength_decay_constant
        tau_prune = self.config.memory.tau_prune
        max_eps = self.config.memory.max_episodes

        with self._lock:
            for ep in self.episodes:
                dt = now - ep.get('last_used', ep.get('timestamp', now))
                ep['activation'] = ep.get('activation', 1.0) * np.exp(-dt / S)

            before = len(self.episodes)
            self.episodes = [ep for ep in self.episodes if ep.get('activation', 0) >= tau_prune]
            removed = before - len(self.episodes)

            if len(self.episodes) > max_eps:
                self.episodes.sort(key=lambda e: e.get('activation', 0), reverse=True)
                self.episodes = self.episodes[:max_eps]
                removed += before - len(self.episodes) - removed

            if removed > 0:
                logging.info(f"[Memory] Pruned {removed} episodes (decay/capacity)")
                self._rebuild_episodic_index()

            self.save()

    # Backward compat alias
    def prune_memory(self, **kwargs):
        self.decay_and_prune()

    def _rebuild_episodic_index(self):
        """Rebuild HNSW from current episodes."""
        self.episodic_index = _make_hnsw_index(self.embedding_dim)
        if self.episodes:
            vecs = np.array(
                [ep['embedding'] for ep in self.episodes],
                dtype=np.float32,
            )
            faiss.normalize_L2(vecs)
            self.episodic_index.add(vecs)

    def save(self):
        with self._lock:
            os.makedirs(self.persist_dir, exist_ok=True)
            with open(os.path.join(self.persist_dir, "episodes.json"), "w", encoding="utf-8") as f:
                json.dump(self.episodes, f, ensure_ascii=False, indent=2)
            with open(os.path.join(self.persist_dir, "skills.json"), "w", encoding="utf-8") as f:
                json.dump(self.skills, f, ensure_ascii=False, indent=2)

    def load(self):
        ep_path = os.path.join(self.persist_dir, "episodes.json")
        if os.path.exists(ep_path):
            with open(ep_path, "r", encoding="utf-8") as f:
                self.episodes = json.load(f)
            if self.episodes:
                self._rebuild_episodic_index()

        sk_path = os.path.join(self.persist_dir, "skills.json")
        if os.path.exists(sk_path):
            with open(sk_path, "r", encoding="utf-8") as f:
                self.skills = json.load(f)
            if self.skills:
                for skill in self.skills:
                    if 'embedding' in skill:
                        vec = np.array(skill['embedding'], dtype=np.float32).reshape(1, -1)
                        faiss.normalize_L2(vec)
                        self.skill_index.add(vec)
