import time
import faiss
import numpy as np
import json
import os
import logging
from typing import List, Dict, Tuple, Any

class MemorySystem:
    """
    Unified Memory System combining Episodic Memory (Raw Traces) and Semantic Memory (Rules/Heuristics).
    Supports deduplication, top-K retrieval via Cosine Similarity.
    """
    def __init__(self, embedding_dim: int = 3072, persist_dir: str = "memory_store"):
        self.embedding_dim = embedding_dim
        self.persist_dir = persist_dir
        
        # Episodic Memory (IndexFlatIP for Cosine Similarity with L2 normalized vectors)
        self.episodic_index = faiss.IndexFlatIP(embedding_dim)
        self.episodes: List[Dict[str, Any]] = []
        
        # Semantic Memory (Rules)
        self.semantic_index = faiss.IndexFlatIP(embedding_dim)
        self.semantic_rules: List[Dict[str, Any]] = []
        
        os.makedirs(self.persist_dir, exist_ok=True)
        self.load()

    def add_episode(self, episode_data: Dict[str, Any], embedding: np.ndarray):
        """
        Episode Schema: {query, context, solution, reasoning_trace, score, timestamp}
        Deduplicates if similarity > 0.95
        """
        vector = np.array(embedding, dtype=np.float32)
        faiss.normalize_L2(vector)
        
        # Deduplication check
        if self.episodic_index.ntotal > 0:
            sims, indices = self.episodic_index.search(vector, 1)
            if sims[0][0] > 0.95:
                logging.info("[Memory] Deduplication triggered. Identical episode exists.")
                return False
                
        episode_data['timestamp'] = time.time()
        self.episodic_index.add(vector)
        self.episodes.append(episode_data)
        self.save()
        return True

    def retrieve_episodes(self, query_emb: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        if self.episodic_index.ntotal == 0:
            return []
            
        vector = np.array(query_emb, dtype=np.float32)
        faiss.normalize_L2(vector)
        k_search = min(k, self.episodic_index.ntotal)
        
        sims, indices = self.episodic_index.search(vector, k_search)
        
        results = []
        for sim, idx in zip(sims[0], indices[0]):
            if idx != -1:
                res = self.episodes[idx].copy()
                res['similarity'] = float(sim)
                res['memory_id'] = int(idx)
                results.append(res)
        return results

    def add_semantic_rule(self, rule_data: Dict[str, Any], embedding: np.ndarray):
        """Rule Schema: {pattern, python_code, confidence, success_count}"""
        vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)
        
        # Deduplication check
        if self.semantic_index.ntotal > 0:
            sims, indices = self.semantic_index.search(vector, 1)
            if sims[0][0] > 0.90:
                idx = int(indices[0][0])
                existing_rule = self.semantic_rules[idx]
                logging.info(f"[Memory] Deduplication triggered: Merging new rule '{rule_data.get('pattern')}' into '{existing_rule.get('pattern')}' (sim: {sims[0][0]:.2f})")
                
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
        if self.semantic_index.ntotal == 0:
            return []
        
        vector = np.array(query_emb, dtype=np.float32)
        faiss.normalize_L2(vector)
        k_search = min(k, self.semantic_index.ntotal)
        
        sims, indices = self.semantic_index.search(vector, k_search)
        results = []
        for sim, idx in zip(sims[0], indices[0]):
            if idx != -1:
                res = self.semantic_rules[idx].copy()
                res['similarity'] = float(sim)
                res['memory_id'] = int(idx)
                results.append(res)
        return results

    def save(self):
        faiss.write_index(self.episodic_index, os.path.join(self.persist_dir, "episodic.faiss"))
        faiss.write_index(self.semantic_index, os.path.join(self.persist_dir, "semantic.faiss"))
        with open(os.path.join(self.persist_dir, "episodes.json"), "w", encoding="utf-8") as f:
            json.dump(self.episodes, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.persist_dir, "rules.json"), "w", encoding="utf-8") as f:
            json.dump(self.semantic_rules, f, ensure_ascii=False, indent=2)

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
