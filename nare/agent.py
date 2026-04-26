import time
import numpy as np
import faiss
import logging
from typing import List, Dict, Any
from . import llm
from .memory import MemorySystem

logging.basicConfig(level=logging.INFO, format='%(message)s')

class HybridCritic:
    def __init__(self, w1=0.6, w2=0.4, w3=0.0):
        self.weights = (w1, w2, w3)

    def _rule_based_check(self, solution: str) -> float:
        score = 0.5
        if "```" in solution or "def " in solution:
            score += 0.3
        if "Error" in solution or "Exception" in solution:
            score -= 0.4
        return max(0.0, min(1.0, score))

    def _anti_gaming_penalty(self, candidates: List[Dict]) -> float:
        lengths = [len(c['solution']) for c in candidates]
        if not lengths: return 0.0
        variance = np.var(lengths)
        mean_len = np.mean(lengths)
        cv = np.sqrt(variance) / (mean_len + 1e-9)
        return min(0.3, cv * 0.2)

    def _self_consistency_bonus(self, candidates: List[Dict]) -> Dict[int, float]:
        """
        Self-Consistency: if multiple candidates converge to the same core answer,
        boost their scores. Uses simple length-bucket heuristic as proxy.
        """
        bonuses = {}
        n = len(candidates)
        if n < 2:
            return {0: 0.0} if n == 1 else {}
        
        # Compare each pair for textual overlap (first 200 chars of solution)
        snippets = [c['solution'][:200].lower().strip() for c in candidates]
        for i in range(n):
            agreement_count = 0
            for j in range(n):
                if i == j: continue
                # Simple overlap ratio
                words_i = set(snippets[i].split())
                words_j = set(snippets[j].split())
                if not words_i: continue
                overlap = len(words_i & words_j) / len(words_i | words_j)
                if overlap > 0.3:
                    agreement_count += 1
            bonuses[i] = 0.1 * (agreement_count / max(1, n - 1))
        return bonuses

    def evaluate(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates: return []
        if len(candidates) == 1:
            candidates[0]['llm_score'] = 0.5
            candidates[0]['rule_score'] = self._rule_based_check(candidates[0]['solution'])
            candidates[0]['final_score'] = 0.5
            return candidates
        
        # 1. Elo Tournament
        for c in candidates: c['elo'] = 1200.0
        
        for i in range(len(candidates) - 1):
            c1, c2 = candidates[i], candidates[i+1]
            try:
                winner = llm.llm_pairwise_judge(query, c1['solution'], c2['solution'])
            except Exception as e:
                logging.error(f"Pairwise judge failed: {e}")
                winner = 1
                
            k_factor = 32
            ea = 1.0 / (1.0 + 10.0 ** ((c2['elo'] - c1['elo']) / 400.0))
            if winner == 1:
                c1['elo'] += k_factor * (1.0 - ea)
                c2['elo'] -= k_factor * (1.0 - ea)
            else:
                c1['elo'] -= k_factor * ea
                c2['elo'] += k_factor * ea

        elos = [c['elo'] for c in candidates]
        e_min, e_max = min(elos), max(elos)
        
        penalty = self._anti_gaming_penalty(candidates)
        sc_bonuses = self._self_consistency_bonus(candidates)
        
        for idx, c in enumerate(candidates):
            if e_max == e_min:
                c['llm_score'] = 0.5
            else:
                c['llm_score'] = (c['elo'] - e_min) / (e_max - e_min)
            
            c['rule_score'] = self._rule_based_check(c['solution'])
            
            base_score = self.weights[0] * c['llm_score'] + self.weights[1] * c['rule_score']
            sc_bonus = sc_bonuses.get(idx, 0.0)
            c['final_score'] = max(0.0, base_score - penalty + sc_bonus)

        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates


class NAREProductionAgent:
    def __init__(self):
        self.memory = MemorySystem()
        self.critic = HybridCritic()
        
        self.tau_fast = 0.85
        self.tau_hybrid = 0.65
        self.tau_min = 0.80  # ARCH-3 fix: never go below 0.80
        
    def _calibrate_tau(self, reward: float, fast_path_used: bool):
        """Dynamic τ adjustment. Clamped to [0.80, 0.95]."""
        lr = 0.02
        if fast_path_used and reward < 0.5:
            self.tau_fast = min(0.95, self.tau_fast + lr)
            logging.info(f"[Calibration] tau_fast ↑ {self.tau_fast:.3f}")
        elif not fast_path_used and reward > 0.8:
            self.tau_fast = max(self.tau_min, self.tau_fast - (lr / 2))

    def _normalize_embeddings(self, vecs: np.ndarray) -> np.ndarray:
        """Utility: L2-normalize a batch of vectors for cosine similarity."""
        v = vecs.copy().astype(np.float32)
        faiss.normalize_L2(v)
        return v

    def _check_sleep_trigger(self) -> bool:
        if len(self.memory.episodes) >= 200:
            return True
            
        if len(self.memory.episodes) >= 2:
            # BUG-4 FIX: normalize before computing dot product
            raw_vecs = np.array([ep['embedding'] for ep in self.memory.episodes], dtype=np.float32)
            vecs = self._normalize_embeddings(raw_vecs)
            sim_matrix = np.dot(vecs, vecs.T)
            # Zero out diagonal (self-similarity)
            np.fill_diagonal(sim_matrix, 0.0)
            dense_clusters = np.sum(sim_matrix > 0.6, axis=1)
            if np.any(dense_clusters >= 1):
                logging.info("[Sleep Trigger] Dense cluster detected.")
                return True
        return False

    def _sleep_phase(self):
        logging.info("=== [SLEEP PHASE] Crystallizing Memories ===")
        raw_vecs = np.array([ep['embedding'] for ep in self.memory.episodes], dtype=np.float32)
        # BUG-4 FIX: normalize
        vecs = self._normalize_embeddings(raw_vecs)
        sim_matrix = np.dot(vecs, vecs.T)
        np.fill_diagonal(sim_matrix, 0.0)
        dense_clusters = np.sum(sim_matrix > 0.6, axis=1)
        
        if np.max(dense_clusters) < 1:
            return
            
        core_idx = int(np.argmax(dense_clusters))
        cluster_indices = np.where(sim_matrix[core_idx] > 0.6)[0]
        
        cluster_episodes = [self.memory.episodes[i] for i in cluster_indices]
        
        # Use the normalized centroid embedding for the rule
        centroid = np.mean(vecs[cluster_indices], axis=0, keepdims=True).astype(np.float32)
        faiss.normalize_L2(centroid)
        
        # Consolidation: Check if a similar rule already exists
        existing_semantics = self.memory.retrieve_semantics(centroid, k=1)
        
        try:
            if existing_semantics and existing_semantics[0]['similarity'] > 0.7:
                logging.info(f"=== [SEMANTIC CONSOLIDATION] Merging into rule: {existing_semantics[0]['pattern']} ===")
                rule_idx = int(existing_semantics[0].get('memory_id', 0))
                updated_rule = llm.merge_heuristic_rules(existing_semantics[0], cluster_episodes)
                self.memory.update_semantic_rule(rule_idx, updated_rule, new_embedding=centroid)
                logging.info(f"[Consolidation] Refined: {updated_rule['pattern']}")
            else:
                logging.info("=== [SLEEP PHASE] Crystallizing New Rule ===")
                new_rule = llm.extract_heuristic_rule(cluster_episodes)
                self.memory.add_semantic_rule(new_rule, centroid)
                logging.info(f"[Crystallization] New Rule: {new_rule['pattern']}")
            
            # SUCCESS: Now compress the episodic memory
            to_delete = set(int(i) for i in cluster_indices) - {core_idx}
            self.memory.episodes = [ep for i, ep in enumerate(self.memory.episodes) if i not in to_delete]
            
            # Rebuild index with remaining episodes
            self.memory.episodic_index = faiss.IndexFlatIP(self.memory.embedding_dim)
            if self.memory.episodes:
                rebuild_vecs = np.array([ep['embedding'] for ep in self.memory.episodes], dtype=np.float32)
                rebuild_vecs = self._normalize_embeddings(rebuild_vecs)
                self.memory.episodic_index.add(rebuild_vecs)
            
            self.memory.save()
            logging.info(f"[Sleep] Compressed {len(to_delete)} episodes into 1 rule + 1 representative.")
            
        except Exception as e:
            logging.error(f"[Sleep Phase] Failed: {e}")

    def _save_episode(self, query: str, query_emb: list, best_cand: Dict, prompt: str):
        """Unified episode saving for ALL paths that generate new content."""
        episode_data = {
            "query": query,
            "context": prompt,
            "solution": best_cand['solution'],
            "reasoning_trace": best_cand.get('reasoning', 'N/A'),
            "score": best_cand.get('final_score', 0.5),
            "embedding": query_emb
        }
        added = self.memory.add_episode(episode_data, np.array([query_emb], dtype=np.float32))
        return added

    def solve(self, query: str) -> Dict[str, Any]:
        log = []
        log.append(f"Query: {query}")
        
        query_emb = llm.get_embedding(query)
        query_emb_np = np.array([query_emb], dtype=np.float32)
        
        # 1. Retrieval
        retrieved_eps = self.memory.retrieve_episodes(query_emb_np, k=3)
        all_semantics = self.memory.retrieve_semantics(query_emb_np, k=2)
        
        # Filter semantics by threshold AND confidence
        retrieved_semantics = [
            s for s in all_semantics 
            if s['similarity'] > 0.70 and s.get('confidence', 0.5) >= 0.70
        ]
        
        if all_semantics and not retrieved_semantics:
            reason = "similarity" if all_semantics[0]['similarity'] <= 0.70 else "low confidence"
            log.append(f"Heuristic rejected: {reason} ({all_semantics[0].get('confidence', 0.5):.2f})")
        
        max_sim = retrieved_eps[0]['similarity'] if retrieved_eps else 0.0
        
        # Entropy check: if ALL top-k similarities are suspiciously close to 1.0, force SLOW
        if len(retrieved_eps) >= 2:
            sims = [ep['similarity'] for ep in retrieved_eps]
            sim_std = np.std(sims)
            if sim_std < 0.01 and max_sim > 0.95:
                logging.info(f"[Entropy Check] All similarities ~{max_sim:.3f} (std={sim_std:.4f}). Possible self-match. Forcing SLOW.")
                max_sim = 0.0  # Force SLOW path
        
        route = "SLOW"
        alpha = 0.0
        candidates = []
        final_answer = ""
        best_cand = None
        prompt_used = ""
        
        # 2. Routing Logic
        
        # 2.1 First Priority: EXECUTABLE REFLEXES (Skill-Based Cognitive System)
        # We check ALL skills in procedural memory. If a skill applies, it REPLACES thinking.
        reflex_triggered = False
        for sem in self.memory.semantic_rules:
            if sem.get('confidence', 0.5) < 0.70:
                continue
            
            try:
                import re as _re, math as _math
                safe_globals = {"__builtins__": __builtins__, "re": _re, "math": _math}
                local_env = {}
                exec(sem['python_code'], safe_globals, local_env)
                
                if 'trigger' in local_env and 'execute' in local_env:
                    if local_env['trigger'](query):
                        log.append(f"Route: REFLEX PATH (Executed Skill: {sem.get('pattern')})")
                        
                        try:
                            final_answer = str(local_env['execute'](query))
                            route = "REFLEX"
                            reflex_triggered = True  # Only set to True if execution succeeds
                        except Exception as exec_err:
                            log.append(f"Execute failed: {exec_err}")
                            raise exec_err
                            
                        best_cand = {
                            'solution': final_answer,
                            'reasoning_trace': f"Executed Procedural Skill: {sem.get('pattern')}",
                            'final_score': 1.0
                        }
                        
                        if self._save_episode(query, query_emb, best_cand, "EXECUTABLE REFLEX"):
                            log.append("Saved REFLEX episode to memory.")
                            
                        # Reward rule
                        sem['confidence'] = min(1.0, sem.get('confidence', 0.5) + 0.1)
                        self.memory.update_semantic_rule(sem.get('memory_id', 0), sem) # Assuming index matches or handled in update
                        break
                        
            except Exception as e:
                log.append(f"Reflex Error for '{sem.get('pattern')}': {e}")
                # Penalize rule
                sem['confidence'] = max(0.0, sem.get('confidence', 0.5) - 0.2)
                self.memory.update_semantic_rule(sem.get('memory_id', 0), sem)
                
        if not reflex_triggered:
            # 2.2 Memory-Augmented Paths
            if max_sim >= self.tau_fast:
                route = "FAST"
                alpha = max_sim
                log.append(f"Route: FAST PATH (sim: {max_sim:.3f} >= {self.tau_fast})")
                
                best_cand = retrieved_eps[0]
                best_cand['final_score'] = alpha
                final_answer = best_cand['solution']
                
            elif max_sim >= self.tau_hybrid and retrieved_eps:
                route = "HYBRID"
                alpha = max_sim
                log.append(f"Route: HYBRID PATH (sim: {max_sim:.3f})")
                
                prompt_used = f"Task: {query}\n\n"
                prompt_used += "--- RETRIEVED PAST EXPERIENCE ---\n"
                prompt_used += f"Similar Past Task: {retrieved_eps[0]['query']}\n"
                prompt_used += f"Past Reasoning: {retrieved_eps[0]['reasoning_trace']}\n"
                prompt_used += f"Past Solution: {retrieved_eps[0]['solution']}\n\n"
                prompt_used += "Your task is SKILL FORMATION and REASONING COMPRESSION. Because you have already solved a similar task, DO NOT write a full step-by-step trace from scratch. Skip the basics. Provide a highly compressed reasoning trace that only addresses the DIFFERENCES between the Past Task and the New Task. Reuse computation as a reflex."
                
                candidates, _ = llm.generate_samples(prompt_used, n=1, mode="HYBRID")
                candidates = self.critic.evaluate(query, candidates)
                best_cand = candidates[0] if candidates else None
                
                if best_cand:
                    hybrid_score = (alpha * 1.0) + ((1 - alpha) * best_cand['final_score'])
                    best_cand['final_score'] = hybrid_score
                    final_answer = best_cand['solution']
                    
                    if self._save_episode(query, query_emb, best_cand, prompt_used):
                        log.append("Saved HYBRID episode to memory.")
                        
            else:
                route = "SLOW"
                alpha = max_sim
                log.append(f"Route: SLOW PATH (sim: {max_sim:.3f} < {self.tau_hybrid})")
                
                prompt_used = f"Task: {query}\n\n"
                
                if retrieved_eps:
                    prompt_used += "--- RELEVANT EPISODIC MEMORIES (Use as analogies) ---\n"
                    for ep in retrieved_eps:
                        prompt_used += f"Past Task: {ep['query']}\n"
                        prompt_used += f"Past Reasoning: {ep['reasoning_trace']}\n"
                        prompt_used += f"Past Solution: {ep['solution']}\n---\n"
                
                prompt_used += "\nSolve the new Task by synthesizing insights from the provided memories (if any) and applying deep reasoning."
                    
                candidates, _ = llm.generate_samples(prompt_used, n=2, mode="SLOW")
                candidates = self.critic.evaluate(query, candidates)
                
                best_cand = candidates[0] if candidates else None
                
                if best_cand:
                    final_answer = best_cand['solution']
                    if self._save_episode(query, query_emb, best_cand, prompt_used):
                        log.append("Saved SLOW episode to memory.")

        # 3. Calibration & Sleep
        if best_cand:
            self._calibrate_tau(reward=best_cand.get('final_score', 0.5), fast_path_used=(route == "FAST"))
            
        if self._check_sleep_trigger():
            self._sleep_phase()

        return {
            "route_decision": route,
            "retrieved_memories": retrieved_eps + retrieved_semantics,
            "generated_candidates": candidates,
            "critic_evaluation_table": candidates if candidates else ([best_cand] if best_cand else []),
            "final_answer": final_answer,
            "memory_update_log": log
        }
