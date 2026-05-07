import hashlib
import time
import faiss
import numpy as np
import json
import os
import logging
import threading
from typing import List, Dict, Tuple, Any, Optional

from ..config import DEFAULT_CONFIG, NareConfig

def episode_content_key(query: str, solution: str = "") -> str:
    """Stable content-derived identifier for an episode.

    We deliberately avoid using the positional index in
    ``MemorySystem.episodes`` as an identifier across operations that
    may shrink/reorder the list (sleep crystallisation deletes source
    episodes, immune pruning drops untrusted ones, etc). A SHA1 over
    the (normalised) query + first 256 chars of the solution survives
    those mutations and uniquely keys the episode by content.
    """
    q = (query or "").strip().lower()
    s = (solution or "").strip()[:256]
    h = hashlib.sha1(f"{q}\n--\n{s}".encode("utf-8")).hexdigest()
    return h[:16]

from .vector import make_hnsw_index, make_quantized_index, normalize_vector

class MemorySystem:
    """Unified episodic + semantic + factual memory.

    Thread-safety: a single ``RLock`` protects the FAISS indices and the
    parallel Python lists. Both reads and writes acquire it. The agent
    runs sleep/REM phases on a background thread, and ``solve()`` reads
    these structures concurrently — without this lock there were race
    conditions on index rebuilds.

    Changes vs previous revision:
      * Episodic index uses HNSW (O(log N)) instead of IndexFlatIP (O(N))
        Semantic/factual remain brute-force (small N).
      * Each episode now carries a trust coefficient ``tau`` ∈ [0,1]
        (immune system).
      * Suppression dictionary blocks known-bad (query, answer) pairs
        (suppression rules).
    """

    def __init__(
        self,
        embedding_dim: int = 3072,
        persist_dir: Optional[str] = None,
        config: NareConfig = DEFAULT_CONFIG,
        episode_ttl: int = 2592000,
        enable_quantization: bool = True,
    ):
        self.embedding_dim = embedding_dim

        self.persist_dir = (
            persist_dir
            or os.environ.get("NARE_MEMORY_DIR")
            or "memory_store"
        )
        self.config = config
        self._lock = threading.RLock()
        self.episode_ttl = episode_ttl
        self.enable_quantization = enable_quantization

        self._query_epoch = 0

        # Deferred persistence
        self._dirty = False
        self._flush_timer = None

        # Disable quantization - IndexIVFPQ requires training which complicates cold start
        # Use HNSW for all cases (fast enough for <100k episodes)
        self.episodic_index = make_hnsw_index(embedding_dim)
        self.episodes: List[Dict[str, Any]] = []
        self.episode_embeddings = np.array([], dtype=np.float32).reshape(0, embedding_dim)

        self.semantic_index = faiss.IndexFlatIP(embedding_dim)
        self.semantic_rules: List[Dict[str, Any]] = []

        self.factual_index = faiss.IndexFlatIP(embedding_dim)
        self.facts: List[Dict[str, Any]] = []

        self.suppression_rules: List[Dict[str, Any]] = []

        self.compiled_skills: List[Dict[str, Any]] = []

        os.makedirs(self.persist_dir, exist_ok=True)
        self.load()

        removed = self.cleanup_expired_episodes()
        if removed > 0:
            logging.info(f"[Memory] Removed {removed} expired episodes on startup")

        if self.episodic_index.ntotal > 0:
            stored_dim = self.episodic_index.d
            if stored_dim != embedding_dim:
                logging.warning(f"[Memory] Dimension mismatch: stored={stored_dim}, requested={embedding_dim}. Rebuilding index.")
                self._rebuild_index()

        # Filter compiled skills with wrong embedding dimension
        if self.compiled_skills:
            valid_skills = []
            for skill in self.compiled_skills:
                emb = skill.get('trigger_embedding', [])
                if len(emb) == embedding_dim:
                    valid_skills.append(skill)
                else:
                    logging.warning(f"[Memory] Removing skill with wrong embedding dim: {len(emb)} != {embedding_dim}")
            self.compiled_skills = valid_skills
            if len(valid_skills) < len(self.compiled_skills):
                self._mark_dirty()

        # Filter semantic rules with wrong embedding dimension
        if self.semantic_rules:
            valid_rules = []
            for rule in self.semantic_rules:
                emb = rule.get('embedding', [])
                if len(emb) == embedding_dim:
                    valid_rules.append(rule)
                else:
                    logging.warning(f"[Memory] Removing semantic rule with wrong embedding dim: {len(emb)} != {embedding_dim}")
            if len(valid_rules) != len(self.semantic_rules):
                self.semantic_rules = valid_rules
                # Rebuild semantic index
                self.semantic_index = faiss.IndexFlatIP(embedding_dim)
                if valid_rules:
                    rule_embs = np.array([r['embedding'] for r in valid_rules], dtype=np.float32)
                    faiss.normalize_L2(rule_embs)
                    self.semantic_index.add(rule_embs)
                self._mark_dirty()

    def add_episode(self, episode_data: Dict[str, Any], embedding: np.ndarray):
        """Episode Schema: {query, context, solution, reasoning_trace, score, timestamp}.

        Deduplicates if similarity > config.sleep.episode_dedup_threshold.
        Initialises immune-system trust coefficient τ .
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
            episode_data['activation_score'] = 1.0
            episode_data['access_count'] = 0
            episode_data['strength'] = 1.0

            episode_data['created_epoch'] = self._query_epoch
            episode_data['last_used_epoch'] = self._query_epoch

            episode_data.setdefault('tau', self.config.immune.initial_tau)

            episode_data.setdefault(
                'episode_key',
                episode_content_key(
                    episode_data.get('query', ''),
                    episode_data.get('solution', ''),
                ),
            )

            episode_data['embedding'] = vector.flatten().tolist()

            self.episodic_index.add(vector)
            self.episodes.append(episode_data)
            self._mark_dirty()
            return True

    def find_episode_indices_by_keys(self, keys: List[str]) -> List[int]:
        """Return current positional indices of episodes whose
        ``episode_key`` is in ``keys``.

        Backwards compatibility: episodes saved before content keys
        existed will be matched by recomputing the key on the fly so
        the lookup degrades gracefully on mixed stores.
        """
        if not keys:
            return []
        wanted = set(keys)
        out: List[int] = []
        with self._lock:
            for idx, ep in enumerate(self.episodes):
                key = ep.get('episode_key') or episode_content_key(
                    ep.get('query', ''), ep.get('solution', ''),
                )
                if key in wanted:
                    out.append(idx)
        return out

    def update_episode_tau(self, idx: int, delta_v: float):
        """Update trust coefficient: τ_i ← τ_i + γ·ΔV, clamped to [0, 1]."""
        gamma = self.config.immune.tau_lr
        with self._lock:
            if 0 <= idx < len(self.episodes):
                old_tau = self.episodes[idx].get('tau', self.config.immune.initial_tau)
                new_tau = max(0.0, min(1.0, old_tau + gamma * delta_v))
                self.episodes[idx]['tau'] = new_tau

    def update_episode_validation(self, idx: int, success: bool):
        """Thread-safe update of episode validation state.

        Args:
            idx: Episode index
            success: True if validation passed, False if failed
        """
        with self._lock:
            if 0 <= idx < len(self.episodes):
                if success:
                    self.episodes[idx]['access_count'] = self.episodes[idx].get('access_count', 0) + 1
                else:
                    self.episodes[idx]['score'] = max(0.3, self.episodes[idx].get('score', 0.8) - 0.3)
                    self.episodes[idx]['validation_failures'] = self.episodes[idx].get('validation_failures', 0) + 1

    def increment_skill_usage(self, skill_id: int):
        """Thread-safe increment of skill use_count.

        Args:
            skill_id: Index in compiled_skills list
        """
        with self._lock:
            if 0 <= skill_id < len(self.compiled_skills):
                self.compiled_skills[skill_id]['use_count'] = self.compiled_skills[skill_id].get('use_count', 0) + 1

    def prune_untrusted_episodes(self):
        """Remove episodes whose τ fell below θ_immune ."""
        theta = self.config.immune.theta_immune
        with self._lock:
            before = len(self.episodes)
            self.episodes = [ep for ep in self.episodes if ep.get('tau', 1.0) >= theta]
            removed = before - len(self.episodes)
            if removed:
                logging.info(f"[Immune] Removed {removed} untrusted episodes (τ < {theta})")
                self._rebuild_episodic_index()
                self._mark_dirty()

    @staticmethod
    def _suppression_hash(text: str) -> str:
        """Deterministic hash for suppression keys.

        ``hash()`` is randomised per-process via PYTHONHASHSEED (CPython
        3.3+), so suppression rules saved with ``hash()`` keys silently
        stop matching after a process restart — every rule becomes a
        dead entry. Use SHA1 (truncated to 16 hex chars, same convention
        as ``episode_content_key``) so saved rules survive load.
        """
        return hashlib.sha1((text or "").strip().lower().encode("utf-8")).hexdigest()[:16]

    def add_suppression_rule(self, query: str, answer: str, embedding: np.ndarray):
        """Block a specific (query, answer) pair from future retrieval ."""
        rule = {
            'query_hash': self._suppression_hash(query),
            'answer_hash': self._suppression_hash(answer),
            'query_snippet': query[:100],
            'answer_snippet': answer[:100],
            'timestamp': time.time(),
        }
        max_rules = self.config.immune.max_suppression_rules
        with self._lock:
            self.suppression_rules.append(rule)
            if len(self.suppression_rules) > max_rules:
                self.suppression_rules = self.suppression_rules[-max_rules:]
            self._mark_dirty()
        logging.info(f"[Immune] Suppression rule added for query: {query[:60]}")

    def is_suppressed(self, query: str, answer: str) -> bool:
        """Check if a (query, answer) pair is suppressed.

        Tolerates legacy rules that stored Python ``hash()`` ints by
        also recomputing ``hash()`` in the lookup — those legacy rules
        are dead after the process that wrote them, but won't crash.
        """
        qh = self._suppression_hash(query)
        ah = self._suppression_hash(answer)
        legacy_qh = hash((query or "").strip().lower())
        legacy_ah = hash((answer or "").strip().lower())
        for rule in self.suppression_rules:
            rqh = rule.get('query_hash')
            rah = rule.get('answer_hash')
            if rqh == qh and rah == ah:
                return True

            if rqh == legacy_qh and rah == legacy_ah:
                return True
        return False

    def prune_fading_memories(self, threshold: Optional[float] = None):
        """Ebbinghaus Forgetting based on query epochs, not wall-clock time.

        Uses transaction-based epochs instead of 24-hour half-life.
        This ensures forgetting actually activates during short benchmarks.

        Old formula: R = exp(-hours / (s * 24))
        New formula: R = exp(-epochs / (s * 100))

        Episodes with retention below ``threshold`` are dropped.
        """
        if threshold is None:
            threshold = self.config.sleep.fading_retention_threshold

        now_epoch = self._query_epoch
        with self._lock:
            kept_episodes = []
            for ep in self.episodes:
                last_used_epoch = ep.get('last_used_epoch', ep.get('created_epoch', 0))
                delta_epochs = now_epoch - last_used_epoch
                s = ep.get('strength', 1.0)

                retention = np.exp(-delta_epochs / (s * 100))
                if retention >= threshold:
                    kept_episodes.append(ep)

            if len(kept_episodes) < len(self.episodes):
                logging.info(
                    f"[Memory] Forgetting: "
                    f"{len(self.episodes) - len(kept_episodes)} episodes faded."
                )
                self.episodes = kept_episodes
                self._rebuild_episodic_index()
                self._mark_dirty()

    def _rebuild_episodic_index(self):
        """Rebuild the HNSW episodic index from current episode list."""
        self.episodic_index = make_hnsw_index(self.embedding_dim)
        if self.episodes:
            # Filter episodes with correct embedding dimension
            valid_episodes = []
            for ep in self.episodes:
                emb = ep.get('embedding', [])
                if len(emb) == self.embedding_dim:
                    valid_episodes.append(ep)
                else:
                    logging.warning(f"[Memory] Skipping episode with wrong embedding dim: {len(emb)} != {self.embedding_dim}")

            if valid_episodes:
                self.episodes = valid_episodes
                vecs = np.array(
                    [ep['embedding'] for ep in valid_episodes],
                    dtype=np.float32,
                )
                # Embeddings already normalized in add_episode, no need to normalize again
                self.episodic_index.add(vecs)
            else:
                logging.warning("[Memory] No valid episodes after dimension filter, starting fresh")
                self.episodes = []

    def retrieve_episodes(self, query_emb: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k episodes, filtering suppressed pairs.

        Increments query epoch for transaction-based forgetting.

        Results are ranked by *trust-weighted similarity*:
            effective_sim = raw_similarity × τ_i
        so that high-trust episodes rank above low-trust ones even if
        their raw embedding distance is slightly worse.
        """
        with self._lock:

            self._query_epoch += 1

            if self.episodic_index.ntotal == 0:
                return []

            vector = np.array(query_emb, dtype=np.float32)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            faiss.normalize_L2(vector)

            k_search = min(k + 10, self.episodic_index.ntotal)
            sims, indices = self.episodic_index.search(vector, k_search)

            pool = []
            for sim, idx in zip(sims[0], indices[0]):
                if idx != -1 and idx < len(self.episodes):
                    ep = self.episodes[idx]

                    if self.is_suppressed(ep.get('query', ''), ep.get('solution', '')):
                        continue
                    tau = ep.get('tau', self.config.immune.initial_tau)
                    res = ep.copy()
                    res['similarity'] = float(sim)
                    res['tau'] = tau
                    res['effective_similarity'] = float(sim) * tau
                    res['memory_id'] = int(idx)

                    self.episodes[idx]['last_used_epoch'] = self._query_epoch
                    self.episodes[idx]['activation_score'] = self.episodes[idx].get('activation_score', 1.0) + 0.1
                    self.episodes[idx]['access_count'] = self.episodes[idx].get('access_count', 0) + 1
                    self.episodes[idx]['last_used'] = time.time()
                    pool.append(res)

            pool.sort(key=lambda x: x['effective_similarity'], reverse=True)
            return pool[:k]

    def add_semantic_rule(self, rule_data: Dict[str, Any], embedding: np.ndarray):
        """Rule Schema: {pattern, python_code, confidence, success_count}.

        Deduplicates if similarity > config.sleep.semantic_dedup_threshold.
        """
        vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)
        threshold = self.config.sleep.semantic_dedup_threshold

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

                    merged_keys = list(
                        set(existing_rule.get('source_episode_keys', []))
                        | set(rule_data.get('source_episode_keys', []))
                    )
                    merged_sources = list(
                        set(existing_rule.get('source_episode_ids', []))
                        | set(rule_data.get('source_episode_ids', []))
                    )

                    if new_conf > old_conf:
                        rule_data['sleep_cycles'] = existing_rule.get('sleep_cycles', 0)
                        rule_data['score_history'] = existing_rule.get('score_history', [])
                        rule_data['source_episode_keys'] = merged_keys
                        rule_data['source_episode_ids'] = merged_sources
                        self.update_semantic_rule(idx, rule_data, new_embedding=embedding)
                    else:
                        existing_rule['sleep_cycles'] = existing_rule.get('sleep_cycles', 0) + 1
                        existing_rule['source_episode_keys'] = merged_keys
                        existing_rule['source_episode_ids'] = merged_sources
                        self.update_semantic_rule(idx, existing_rule)
                    return True

            rule_data['embedding'] = embedding.tolist()
            rule_data['confidence'] = rule_data.get('confidence', 0.5)
            rule_data['success_count'] = rule_data.get('success_count', 0)

            self.semantic_index.add(vector)
            self.semantic_rules.append(rule_data)
            self._mark_dirty()
            return False

    def update_semantic_rule(self, idx: int, rule_data: Dict[str, Any], new_embedding: np.ndarray = None):
        """Update an existing rule and rebuild the semantic index."""
        with self._lock:
            if new_embedding is not None:
                rule_data['embedding'] = new_embedding.tolist()
            else:
                rule_data['embedding'] = self.semantic_rules[idx].get('embedding')

            self.semantic_rules[idx] = rule_data

            self.semantic_index = faiss.IndexFlatIP(self.embedding_dim)
            for rule in self.semantic_rules:
                if 'embedding' in rule:
                    v = np.array(rule['embedding'], dtype=np.float32).flatten()
                    v = v.reshape(1, -1)
                    faiss.normalize_L2(v)
                    self.semantic_index.add(v)
            self._mark_dirty()

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
            self._mark_dirty()
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

            os.makedirs(self.persist_dir, exist_ok=True)

        # 1. Under lock - make deep copies
        with self._lock:
            episodes_copy = copy.deepcopy(self.episodes)
            rules_copy = copy.deepcopy(self.semantic_rules)
            facts_copy = copy.deepcopy(self.facts)
            suppression_copy = copy.deepcopy(self.suppression_rules)
            skills_copy = copy.deepcopy(self.compiled_skills)

            # Clone FAISS indices
            episodic_clone = faiss.clone_index(self.episodic_index)
            semantic_clone = faiss.clone_index(self.semantic_index)
            factual_clone = faiss.clone_index(self.factual_index)

        # 2. Without lock - write to disk
        os.makedirs(self.persist_dir, exist_ok=True)

        faiss.write_index(episodic_clone, os.path.join(self.persist_dir, "episodic.faiss"))
        faiss.write_index(semantic_clone, os.path.join(self.persist_dir, "semantic.faiss"))
        faiss.write_index(factual_clone, os.path.join(self.persist_dir, "factual.faiss"))

        with open(os.path.join(self.persist_dir, "episodes.json"), "w", encoding="utf-8") as f:
            json.dump(episodes_copy, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.persist_dir, "rules.json"), "w", encoding="utf-8") as f:
            json.dump(rules_copy, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.persist_dir, "facts.json"), "w", encoding="utf-8") as f:
            json.dump(facts_copy, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.persist_dir, "suppression.json"), "w", encoding="utf-8") as f:
            json.dump(suppression_copy, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.persist_dir, "compiled_skills.json"), "w", encoding="utf-8") as f:
            json.dump(skills_copy, f, ensure_ascii=False, indent=2)

    def _mark_dirty(self):
        """Mark memory as dirty and schedule flush."""
        self._dirty = True
        if self._flush_timer is None:
            self._flush_timer = threading.Timer(5.0, self._flush)
            self._flush_timer.daemon = True
            self._flush_timer.start()

    def _flush(self):
        """Background flush of dirty memory."""
        with self._lock:
            if self._dirty:
                self.save()
                self._dirty = False
            self._flush_timer = None

    def force_save(self):
        """Force immediate save (for shutdown)."""
        if self._flush_timer:
            self._flush_timer.cancel()
            self._flush_timer = None
        self.save()

    def _mark_dirty(self):
        """Mark memory as dirty and schedule flush."""
        self._dirty = True
        if self._flush_timer is None:
            self._flush_timer = threading.Timer(5.0, self._flush)
            self._flush_timer.daemon = True
            self._flush_timer.start()

    def _flush(self):
        """Background flush of dirty memory."""
        with self._lock:
            if self._dirty:
                self._mark_dirty()
                self._dirty = False
            self._flush_timer = None

    def force_save(self):
        """Force immediate save (for shutdown)."""
        if self._flush_timer:
            self._flush_timer.cancel()
            self._flush_timer = None
        self.save()

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

        sup_data = os.path.join(self.persist_dir, "suppression.json")
        if os.path.exists(sup_data):
            with open(sup_data, "r", encoding="utf-8") as f:
                self.suppression_rules = json.load(f)

        skills_data = os.path.join(self.persist_dir, "compiled_skills.json")
        if os.path.exists(skills_data):
            with open(skills_data, "r", encoding="utf-8") as f:
                self.compiled_skills = json.load(f)

    def cleanup_expired_episodes(self) -> int:
        """Remove episodes older than TTL.

        Returns:
            Number of episodes removed
        """
        if not self.episode_ttl:
            return 0

        current_time = time.time()
        removed = 0

        with self._lock:

            valid_indices = []
            for i, episode in enumerate(self.episodes):
                timestamp = episode.get('timestamp', current_time)
                age = current_time - timestamp

                if age < self.episode_ttl:
                    valid_indices.append(i)
                else:
                    removed += 1
                    logging.info(f"[Memory] Removing expired episode (age: {age/86400:.1f} days)")

            if removed > 0:

                self.episodes = [self.episodes[i] for i in valid_indices]

                if len(self.episodes) > 0:
                    embeddings = []
                    for i in valid_indices:
                        if i < len(self.episode_embeddings):
                            embeddings.append(self.episode_embeddings[i])

                    if embeddings:
                        self.episode_embeddings = np.array(embeddings, dtype=np.float32)
                        self._rebuild_episodic_index()
                else:

                    self.episode_embeddings = np.array([], dtype=np.float32).reshape(0, self.embedding_dim)
                    if self.enable_quantization and self.embedding_dim >= 512:
                        self.episodic_index = make_quantized_index(self.embedding_dim)
                    else:
                        self.episodic_index = make_hnsw_index(self.embedding_dim)

                self._mark_dirty()

        return removed

    def defragment_index(self) -> None:
        """Defragment FAISS index for better performance.

        Rebuilds the index from scratch to optimize memory layout.
        """
        with self._lock:
            if len(self.episodes) == 0:
                return

            logging.info(f"[Memory] Defragmenting index ({len(self.episodes)} episodes)")

            self._rebuild_episodic_index()

            if len(self.semantic_rules) > 0:
                self.semantic_index = faiss.IndexFlatIP(self.embedding_dim)
                for rule in self.semantic_rules:
                    if 'embedding' in rule:
                        emb = np.array(rule['embedding'], dtype=np.float32).reshape(1, -1)
                        faiss.normalize_L2(emb)
                        self.semantic_index.add(emb)

            if len(self.facts) > 0:
                self.factual_index = faiss.IndexFlatIP(self.embedding_dim)
                for fact in self.facts:
                    if 'embedding' in fact:
                        emb = np.array(fact['embedding'], dtype=np.float32).reshape(1, -1)
                        faiss.normalize_L2(emb)
                        self.factual_index.add(emb)

            self._mark_dirty()
            logging.info("[Memory] Defragmentation complete")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Dict with memory stats
        """
        with self._lock:
            total_episodes = len(self.episodes)
            total_skills = len(self.compiled_skills)
            total_rules = len(self.semantic_rules)

            if total_episodes > 0:
                current_time = time.time()
                ages = [(current_time - ep.get('timestamp', current_time)) / 86400 for ep in self.episodes]
                avg_age = sum(ages) / len(ages)
                max_age = max(ages)
            else:
                avg_age = 0
                max_age = 0

            index_size_mb = 0
            if hasattr(self.episodic_index, 'ntotal'):

                index_size_mb = (4 * self.embedding_dim * self.episodic_index.ntotal) / (1024 * 1024)

            return {
                'episodes': total_episodes,
                'skills': total_skills,
                'rules': total_rules,
                'avg_age_days': avg_age,
                'max_age_days': max_age,
                'index_size_mb': index_size_mb,
                'embedding_dim': self.embedding_dim,
                'quantization_enabled': self.enable_quantization,
                'ttl_days': self.episode_ttl / 86400 if self.episode_ttl else None,
            }

    def prune_memory(self, tau_prune: float = 0.1, decay_rate: float = 0.01):
        """Remove episodes with low activation scores (Ebbinghaus forgetting).

        Uses exponential decay: s_i * exp(-Δt / S)
        Episodes with activation_score < tau_prune are removed.
        Also removes episodes that failed validation (false positives).
        """
        current_time = time.time()
        with self._lock:
            kept_episodes = []

            for ep in self.episodes:

                validation_failures = ep.get('validation_failures', 0)
                if validation_failures >= 2:

                    logging.info(f"[Memory] Removing episode with {validation_failures} validation failures")
                    continue

                if validation_failures > 0:
                    ep['score'] = max(0.5, ep.get('score', 0.8) - (0.2 * validation_failures))

                last_access = ep.get('last_used', ep.get('timestamp', current_time))
                delta_t = current_time - last_access
                decay = np.exp(-delta_t * decay_rate)

                ep['activation_score'] = ep.get('activation_score', 1.0) * decay

                if ep['activation_score'] >= tau_prune:
                    kept_episodes.append(ep)
                elif ep.get('access_count', 0) > 0 or ep.get('score', 0) >= 0.80:

                    if validation_failures == 0:
                        kept_episodes.append(ep)

            if len(kept_episodes) < len(self.episodes):
                removed = len(self.episodes) - len(kept_episodes)
                logging.info(f"[Memory] Pruned {removed} low-quality/stale/activation episodes")
                self.episodes = kept_episodes
                self._rebuild_episodic_index()
                self._mark_dirty()

    def _rebuild_index(self):
        """Rebuild FAISS index with current embedding dimension."""
        self._rebuild_episodic_index()

    def add_compiled_skill(self, pattern: str, code: str, trigger_emb: np.ndarray):
        """Add a compiled skill (reusable function)."""
        with self._lock:
            self.compiled_skills.append({
                'pattern': pattern,
                'code': code,
                'trigger_embedding': trigger_emb.tolist() if hasattr(trigger_emb, 'tolist') else list(trigger_emb),
                'use_count': 0,
                'created_at': time.time()
            })
            logging.info(f"[LibraryLearning] Compiled skill: {pattern}")
            self._mark_dirty()

    def retrieve_skills(self, query_emb: np.ndarray, k: int = 3) -> List[Dict]:
        """Retrieve top-k compiled skills by similarity."""
        with self._lock:
            if not self.compiled_skills:
                return []

            # Filter skills with correct embedding dimension
            valid_skills = [s for s in self.compiled_skills if len(s.get('trigger_embedding', [])) == self.embedding_dim]
            if not valid_skills:
                logging.warning("[Memory] No valid skills with correct embedding dimension")
                return []

            skill_embs = np.array([s['trigger_embedding'] for s in valid_skills], dtype=np.float32)
            faiss.normalize_L2(skill_embs)

            query_vec = np.array(query_emb, dtype=np.float32)
            if query_vec.ndim == 1:
                query_vec = query_vec.reshape(1, -1)
            faiss.normalize_L2(query_vec)

            temp_index = faiss.IndexFlatIP(skill_embs.shape[1])
            temp_index.add(skill_embs)

            k_search = min(k, len(valid_skills))
            sims, indices = temp_index.search(query_vec, k_search)

            results = []
            for sim, idx in zip(sims[0], indices[0]):
                if idx >= 0 and idx < len(valid_skills):
                    skill = valid_skills[idx].copy()
                    skill['similarity'] = float(sim)
                    skill['skill_id'] = int(idx)
                    results.append(skill)
            return results
