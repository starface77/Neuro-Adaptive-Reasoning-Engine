import time
import faiss
import numpy as np
import json
import os
import logging
import threading
from typing import List, Dict, Tuple, Any, Optional

from .config import DEFAULT_CONFIG, NareConfig


class MemorySystem:
    """Unified episodic + semantic + factual memory.

    Thread-safety: a single ``RLock`` protects the FAISS indices and the
    parallel Python lists. Both reads and writes acquire it. The agent
    runs sleep/REM phases on a background thread, and ``solve()`` reads
    these structures concurrently — without this lock there were race
    conditions on index rebuilds.
    """

    def __init__(
        self,
        embedding_dim: int = 3072,
        persist_dir: str = "memory_store",
        config: NareConfig = DEFAULT_CONFIG,
    ):
        self.embedding_dim = embedding_dim
        self.persist_dir = persist_dir
        self.config = config
        # Reentrant: save() may be invoked from within a write path that
        # already holds the lock.
        self._lock = threading.RLock()
        
        # Episodic Memory (IndexFlatIP for Cosine Similarity with L2 normalized vectors)
        self.episodic_index = faiss.IndexFlatIP(embedding_dim)
        self.episodes: List[Dict[str, Any]] = []
        
        # Semantic Memory (Rules)
        self.semantic_index = faiss.IndexFlatIP(embedding_dim)
        self.semantic_rules: List[Dict[str, Any]] = []
        
        # Factual Memory (RAG knowledge base)
        self.factual_index = faiss.IndexFlatIP(embedding_dim)
        self.facts: List[Dict[str, Any]] = []
        
        os.makedirs(self.persist_dir, exist_ok=True)
        self.load()

    def add_episode(self, episode_data: Dict[str, Any], embedding: np.ndarray):
        """Episode Schema: {query, context, solution, reasoning_trace, score, timestamp}.

        Deduplicates if similarity > config.sleep.episode_dedup_threshold.
        """
        vector = np.array(embedding, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)
        threshold = self.config.sleep.episode_dedup_threshold

        with self._lock:
            if self.episodic_index.ntotal > 0:
                sims, _indices = self.episodic_index.search(vector, 1)
                if sims[0][0] > threshold:
                    logging.info(
                        "[Memory] Deduplication triggered. Identical episode exists."
                    )
                    return False

            episode_data['timestamp'] = time.time()
            episode_data['last_used'] = time.time()
            episode_data['strength'] = 1.0
            self.episodic_index.add(vector)
            self.episodes.append(episode_data)
            self.save()
            return True

    def prune_fading_memories(self, threshold: Optional[float] = None):
        """Ebbinghaus Forgetting: R = exp(-t / (s * 24h)).

        Episodes with retention below ``threshold`` are dropped at the
        next sleep cycle. If ``threshold`` is None, falls back to
        ``config.sleep.fading_retention_threshold``.
        """
        if threshold is None:
            threshold = self.config.sleep.fading_retention_threshold

        now = time.time()
        with self._lock:
            kept_episodes = []
            for ep in self.episodes:
                t = (now - ep.get('last_used', ep['timestamp'])) / 3600
                s = ep.get('strength', 1.0)
                retention = np.exp(-t / (s * 24))
                if retention >= threshold:
                    kept_episodes.append(ep)

            if len(kept_episodes) < len(self.episodes):
                logging.info(
                    f"[Memory] Forgetting: "
                    f"{len(self.episodes) - len(kept_episodes)} episodes faded."
                )
                self.episodes = kept_episodes
                self.episodic_index = faiss.IndexFlatIP(self.embedding_dim)
                if self.episodes:
                    vecs = np.array(
                        [ep['embedding'] for ep in self.episodes],
                        dtype=np.float32,
                    )
                    faiss.normalize_L2(vecs)
                    self.episodic_index.add(vecs)
                self.save()

    def retrieve_episodes(self, query_emb: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        with self._lock:
            if self.episodic_index.ntotal == 0:
                return []

            vector = np.array(query_emb, dtype=np.float32)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            faiss.normalize_L2(vector)
            k_search = min(k, self.episodic_index.ntotal)
            sims, indices = self.episodic_index.search(vector, k_search)

            results = []
            for sim, idx in zip(sims[0], indices[0]):
                if idx != -1 and idx < len(self.episodes):
                    res = self.episodes[idx].copy()
                    res['similarity'] = float(sim)
                    res['memory_id'] = int(idx)
                    results.append(res)
            return results

    def add_semantic_rule(self, rule_data: Dict[str, Any], embedding: np.ndarray):
        """Rule Schema: {pattern, python_code, confidence, success_count}.

        Deduplicates if similarity > config.sleep.semantic_dedup_threshold.
        """
        vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)
        threshold = self.config.sleep.semantic_dedup_threshold

        # Deduplication check (under lock to keep index + list aligned).
        with self._lock:
            if self.semantic_index.ntotal > 0:
                sims, indices = self.semantic_index.search(vector, 1)
                if sims[0][0] > threshold:
                    idx = int(indices[0][0])
                    existing_rule = self.semantic_rules[idx]
                    logging.info(
                        f"[Memory] Deduplication triggered: Merging new rule "
                        f"'{rule_data.get('pattern')}' into "
                        f"'{existing_rule.get('pattern')}' (sim: {sims[0][0]:.2f})"
                    )

                    new_conf = rule_data.get('confidence', 0.5)
                    old_conf = existing_rule.get('confidence', 0.5)

                    if new_conf > old_conf:
                        rule_data['sleep_cycles'] = existing_rule.get('sleep_cycles', 0)
                        rule_data['score_history'] = existing_rule.get('score_history', [])
                        self.update_semantic_rule(idx, rule_data, new_embedding=embedding)
                    else:
                        existing_rule['sleep_cycles'] = existing_rule.get('sleep_cycles', 0) + 1
                        self.update_semantic_rule(idx, existing_rule)
                    return True

            rule_data['embedding'] = embedding.tolist()
            rule_data['confidence'] = rule_data.get('confidence', 0.5)
            rule_data['success_count'] = rule_data.get('success_count', 0)

            self.semantic_index.add(vector)
            self.semantic_rules.append(rule_data)
            self.save()
            return False

    def update_semantic_rule(self, idx: int, rule_data: Dict[str, Any], new_embedding: np.ndarray = None):
        """Update an existing rule and rebuild the semantic index."""
        with self._lock:
            if new_embedding is not None:
                rule_data['embedding'] = new_embedding.tolist()
            else:
                rule_data['embedding'] = self.semantic_rules[idx].get('embedding')

            self.semantic_rules[idx] = rule_data

            # Rebuild index
            self.semantic_index = faiss.IndexFlatIP(self.embedding_dim)
            for rule in self.semantic_rules:
                if 'embedding' in rule:
                    v = np.array(rule['embedding'], dtype=np.float32).flatten()
                    v = v.reshape(1, -1)
                    faiss.normalize_L2(v)
                    self.semantic_index.add(v)
            self.save()

    def retrieve_semantics(self, query_emb: np.ndarray, k: int = 2) -> List[Dict[str, Any]]:
        with self._lock:
            if self.semantic_index.ntotal == 0:
                return []

            vector = np.array(query_emb, dtype=np.float32)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            faiss.normalize_L2(vector)
            k_search = min(k, self.semantic_index.ntotal)
            sims, indices = self.semantic_index.search(vector, k_search)

            results = []
            for sim, idx in zip(sims[0], indices[0]):
                if idx != -1 and idx < len(self.semantic_rules):
                    res = self.semantic_rules[idx].copy()
                    res['similarity'] = float(sim)
                    res['memory_id'] = int(idx)
                    results.append(res)
            return results

    def add_fact(self, fact_data: Dict[str, Any], embedding: np.ndarray) -> bool:
        """Add a factual knowledge entry (RAG layer).

        Schema: {content, source, category, timestamp}.
        Deduplicates if similarity > config.sleep.fact_dedup_threshold.
        """
        vector = np.array(embedding, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)
        threshold = self.config.sleep.fact_dedup_threshold

        with self._lock:
            if self.factual_index.ntotal > 0:
                sims, _indices = self.factual_index.search(vector, 1)
                if sims[0][0] > threshold:
                    logging.info("[Memory] Fact deduplication triggered.")
                    return False

            fact_data["timestamp"] = time.time()
            fact_data["embedding"] = (
                embedding.tolist()
                if hasattr(embedding, "tolist")
                else list(embedding.flat)
            )
            self.factual_index.add(vector)
            self.facts.append(fact_data)
            self.save()
            return True

    def retrieve_facts(self, query_emb: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant facts via cosine similarity (RAG)."""
        with self._lock:
            if self.factual_index.ntotal == 0:
                return []
            vector = np.array(query_emb, dtype=np.float32)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            faiss.normalize_L2(vector)
            k_search = min(k, self.factual_index.ntotal)
            sims, indices = self.factual_index.search(vector, k_search)
            results = []
            for sim, idx in zip(sims[0], indices[0]):
                if idx != -1 and idx < len(self.facts) and float(sim) > 0.5:
                    res = self.facts[idx].copy()
                    res["similarity"] = float(sim)
                    results.append(res)
            return results

    def save(self):
        with self._lock:
            faiss.write_index(self.episodic_index, os.path.join(self.persist_dir, "episodic.faiss"))
            faiss.write_index(self.semantic_index, os.path.join(self.persist_dir, "semantic.faiss"))
            faiss.write_index(self.factual_index, os.path.join(self.persist_dir, "factual.faiss"))
            with open(os.path.join(self.persist_dir, "episodes.json"), "w", encoding="utf-8") as f:
                json.dump(self.episodes, f, ensure_ascii=False, indent=2)
            with open(os.path.join(self.persist_dir, "rules.json"), "w", encoding="utf-8") as f:
                json.dump(self.semantic_rules, f, ensure_ascii=False, indent=2)
            with open(os.path.join(self.persist_dir, "facts.json"), "w", encoding="utf-8") as f:
                json.dump(self.facts, f, ensure_ascii=False, indent=2)

    def load(self):
        ep_index = os.path.join(self.persist_dir, "episodic.faiss")
        ep_data = os.path.join(self.persist_dir, "episodes.json")
        if os.path.exists(ep_index) and os.path.exists(ep_data):
            self.episodic_index = faiss.read_index(ep_index)
            with open(ep_data, "r", encoding="utf-8") as f:
                self.episodes = json.load(f)

        sem_index = os.path.join(self.persist_dir, "semantic.faiss")
        sem_data = os.path.join(self.persist_dir, "rules.json")
        if os.path.exists(sem_index) and os.path.exists(sem_data):
            self.semantic_index = faiss.read_index(sem_index)
            with open(sem_data, "r", encoding="utf-8") as f:
                self.semantic_rules = json.load(f)

        fact_index = os.path.join(self.persist_dir, "factual.faiss")
        fact_data = os.path.join(self.persist_dir, "facts.json")
        if os.path.exists(fact_index) and os.path.exists(fact_data):
            self.factual_index = faiss.read_index(fact_index)
            with open(fact_data, "r", encoding="utf-8") as f:
                self.facts = json.load(f)
