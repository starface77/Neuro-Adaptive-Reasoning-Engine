import hashlib
import time
import faiss
import numpy as np
import json
import os
import logging
import threading
from typing import List, Dict, Tuple, Any, Optional

from .config import DEFAULT_CONFIG, NareConfig


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


def _make_hnsw_index(dim: int, max_elements: int = 1000) -> faiss.Index:
    """Create an HNSW index for O(log N) approximate nearest neighbour.

    FAST cache uses HNSW/FAISS for O(log N) retrieval,
    not brute-force O(N).  IndexHNSWFlat wraps a flat storage with an
    HNSW graph that provides logarithmic search complexity.
    """
    M = 32  # HNSW connectivity parameter
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 64
    index.hnsw.efSearch = 32
    return index


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
    ):
        self.embedding_dim = embedding_dim
        # Resolution order:
        #   1. explicit ``persist_dir`` argument (highest priority)
        #   2. ``NARE_MEMORY_DIR`` environment variable
        #   3. default ``"memory_store"`` (CWD-relative)
        # Benchmarks set the env var to isolate runs from the global
        # store; without env-var support that contract was silently
        # broken (Devin Review on commit ac8ab52).
        self.persist_dir = (
            persist_dir
            or os.environ.get("NARE_MEMORY_DIR")
            or "memory_store"
        )
        self.config = config
        self._lock = threading.RLock()

        # Episodic Memory — HNSW index for O(log N) retrieval 
        self.episodic_index = _make_hnsw_index(embedding_dim)
        self.episodes: List[Dict[str, Any]] = []

        # Semantic Memory (Rules) — brute-force (small N)
        self.semantic_index = faiss.IndexFlatIP(embedding_dim)
        self.semantic_rules: List[Dict[str, Any]] = []

        # Factual Memory (RAG knowledge base)
        self.factual_index = faiss.IndexFlatIP(embedding_dim)
        self.facts: List[Dict[str, Any]] = []

        # Suppression dictionary : list of {query_hash, answer_hash,
        # embedding} entries.  When a retrieval hit matches a suppressed
        # pair, it is filtered out.
        self.suppression_rules: List[Dict[str, Any]] = []

        os.makedirs(self.persist_dir, exist_ok=True)
        self.load()

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
            episode_data['strength'] = 1.0
            # Immune system : initial trust coefficient
            episode_data.setdefault('tau', self.config.immune.initial_tau)
            # Stable content-derived key — survives episode deletion /
            # reordering, so semantic rules can reference source episodes
            # without risk of stale positional indices.
            episode_data.setdefault(
                'episode_key',
                episode_content_key(
                    episode_data.get('query', ''),
                    episode_data.get('solution', ''),
                ),
            )
            self.episodic_index.add(vector)
            self.episodes.append(episode_data)
            self.save()
            return True

    # ------------------------------------------------------------------
    # Stable-key episode lookup
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Immune system helpers 
    # ------------------------------------------------------------------

    def update_episode_tau(self, idx: int, delta_v: float):
        """Update trust coefficient: τ_i ← τ_i + γ·ΔV, clamped to [0, 1]."""
        gamma = self.config.immune.tau_lr
        with self._lock:
            if 0 <= idx < len(self.episodes):
                old_tau = self.episodes[idx].get('tau', self.config.immune.initial_tau)
                new_tau = max(0.0, min(1.0, old_tau + gamma * delta_v))
                self.episodes[idx]['tau'] = new_tau

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
                self.save()

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
            self.save()
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
            # Best-effort check for any in-process legacy rules.
            if rqh == legacy_qh and rah == legacy_ah:
                return True
        return False

    # ------------------------------------------------------------------
    # Ebbinghaus forgetting
    # ------------------------------------------------------------------

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
                self._rebuild_episodic_index()
                self.save()

    def _rebuild_episodic_index(self):
        """Rebuild the HNSW episodic index from current episode list."""
        self.episodic_index = _make_hnsw_index(self.embedding_dim)
        if self.episodes:
            vecs = np.array(
                [ep['embedding'] for ep in self.episodes],
                dtype=np.float32,
            )
            faiss.normalize_L2(vecs)
            self.episodic_index.add(vecs)

    def retrieve_episodes(self, query_emb: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k episodes, filtering suppressed pairs.

        Results are ranked by *trust-weighted similarity*:
            effective_sim = raw_similarity × τ_i
        so that high-trust episodes rank above low-trust ones even if
        their raw embedding distance is slightly worse.
        """
        with self._lock:
            if self.episodic_index.ntotal == 0:
                return []

            vector = np.array(query_emb, dtype=np.float32)
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            faiss.normalize_L2(vector)
            # Fetch extra to account for suppression filtering + τ re-ranking
            k_search = min(k + 10, self.episodic_index.ntotal)
            sims, indices = self.episodic_index.search(vector, k_search)

            pool = []
            for sim, idx in zip(sims[0], indices[0]):
                if idx != -1 and idx < len(self.episodes):
                    ep = self.episodes[idx]
                    # Suppression check
                    if self.is_suppressed(ep.get('query', ''), ep.get('solution', '')):
                        continue
                    tau = ep.get('tau', self.config.immune.initial_tau)
                    res = ep.copy()
                    res['similarity'] = float(sim)
                    res['tau'] = tau
                    res['effective_similarity'] = float(sim) * tau
                    res['memory_id'] = int(idx)
                    pool.append(res)

            # Re-rank by trust-weighted similarity
            pool.sort(key=lambda x: x['effective_similarity'], reverse=True)
            return pool[:k]

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

                    # Merge source episode references for penalty
                    # backpropagation. Stable content keys (sha1) are
                    # the canonical identifier; the legacy positional
                    # ids field is kept in lockstep purely for
                    # backwards-compat with serialised rules.
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
            with open(os.path.join(self.persist_dir, "suppression.json"), "w", encoding="utf-8") as f:
                json.dump(self.suppression_rules, f, ensure_ascii=False, indent=2)

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
