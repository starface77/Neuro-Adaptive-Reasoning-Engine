import time
import numpy as np
import faiss
import logging
from typing import List, Dict, Any
from . import llm
from .memory import MemorySystem
from .metrics import MetricsTracker
from .graph_memory import EpisodeGraph
from .rl_retriever import RLRetriever
from .neural_memory import NeuralMemory
from .meta_abduction import MetaAbductionEngine

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
        self.metrics = MetricsTracker(persist_dir=self.memory.persist_dir)
        self.graph = EpisodeGraph(persist_dir=self.memory.persist_dir)
        self.rl_retriever = RLRetriever(persist_dir=self.memory.persist_dir)
        self.neural_memory = NeuralMemory(persist_dir=self.memory.persist_dir)
        self.meta_engine = MetaAbductionEngine(persist_dir=self.memory.persist_dir)
        
        self.tau_fast = 0.98
        self.tau_hybrid = 0.75   # Lowered: let HYBRID activate for similar queries
        self.tau_min = 0.95
        
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
            
        if len(self.memory.episodes) >= 3:
            # Use structural signature embeddings for clustering instead of raw query text
            raw_vecs = np.array([ep.get('signature_embedding', ep['embedding']) for ep in self.memory.episodes], dtype=np.float32)
            vecs = self._normalize_embeddings(raw_vecs)
            sim_matrix = np.dot(vecs, vecs.T)
            # Zero out diagonal (self-similarity)
            np.fill_diagonal(sim_matrix, 0.0)
            dense_clusters = np.sum(sim_matrix > 0.5, axis=1)
            if np.any(dense_clusters >= 1):
                logging.info(f"[Sleep Trigger] Dense cluster detected (min similarity: {np.max(sim_matrix):.2f})")
                return True
        return False

    def _sleep_phase(self):
        logging.info("=== [SLEEP PHASE] Crystallizing Memories ===")
        # Use structural signature embeddings
        raw_vecs = np.array([ep.get('signature_embedding', ep['embedding']) for ep in self.memory.episodes], dtype=np.float32)
        # BUG-4 FIX: normalize
        vecs = self._normalize_embeddings(raw_vecs)
        sim_matrix = np.dot(vecs, vecs.T)
        np.fill_diagonal(sim_matrix, 0.0)
        dense_clusters = np.sum(sim_matrix > 0.5, axis=1)
        
        if np.max(dense_clusters) < 1:
            # No clusters found, but we can still try to refine weak rules
            self._prune_weak_rules()
            self.memory.prune_fading_memories()
            return
            
        core_idx = int(np.argmax(dense_clusters))
        cluster_indices = np.where(sim_matrix[core_idx] > 0.5)[0]
        
        cluster_episodes = [self.memory.episodes[i] for i in cluster_indices]
        
        # Use the normalized centroid embedding for the rule
        centroid = np.mean(vecs[cluster_indices], axis=0, keepdims=True).astype(np.float32)
        faiss.normalize_L2(centroid)
        
        # Consolidation: Check if a similar rule already exists
        existing_semantics = self.memory.retrieve_semantics(centroid, k=1)
        
        try:
            if existing_semantics and existing_semantics[0]['similarity'] > 0.7:
                existing_rule = existing_semantics[0]
                rule_idx = int(existing_rule.get('memory_id', 0))
                existing_conf = existing_rule.get('confidence', 0.5)
                
                if existing_conf < 0.95:
                    # GRADED REFINEMENT: Weak/Hybrid skill needs more training
                    logging.info(f"=== [GRADED REFINEMENT] Upgrading rule '{existing_rule.get('pattern')}' (conf: {existing_conf:.2f}) ===")
                    updated_rule = llm.merge_heuristic_rules(existing_rule, cluster_episodes)
                    
                    # Re-validate the merged rule with stress tests
                    from nare.llm import generate_stress_tests, _validate_skill
                    stress_tests = generate_stress_tests(cluster_episodes)
                    all_tests = cluster_episodes + stress_tests
                    scores, error_msg = _validate_skill(updated_rule.get('python_code', ''), all_tests)
                    
                    updated_rule['confidence'] = scores['overall']
                    updated_rule['trigger_accuracy'] = scores['trigger_accuracy']
                    updated_rule['execute_accuracy'] = scores['execute_accuracy']
                    updated_rule['sleep_cycles'] = existing_rule.get('sleep_cycles', 0) + 1
                    self.memory.update_semantic_rule(rule_idx, updated_rule, new_embedding=centroid)
                    logging.info(f"[Refinement] '{updated_rule['pattern']}' conf: {existing_conf:.2f} -> {scores['overall']:.2f} (trigger={scores['trigger_accuracy']:.2f}, exec={scores['execute_accuracy']:.2f})")
                else:
                    # Already strong rule, just merge for broader coverage
                    logging.info(f"=== [SEMANTIC CONSOLIDATION] Merging into rule: {existing_rule['pattern']} ===")
                    updated_rule = llm.merge_heuristic_rules(existing_rule, cluster_episodes)
                    updated_rule['confidence'] = existing_conf  # Preserve high confidence
                    updated_rule['sleep_cycles'] = existing_rule.get('sleep_cycles', 0) + 1
                    self.memory.update_semantic_rule(rule_idx, updated_rule, new_embedding=centroid)
                    logging.info(f"[Consolidation] Refined: {updated_rule['pattern']}")
            else:
                logging.info("=== [SLEEP PHASE] Crystallizing New Rule ===")
                new_rule = llm.extract_heuristic_rule(cluster_episodes)
                if new_rule is None:
                    logging.warning("[Sleep] Skill failed validation completely (robustness < 0.40). Keeping episodes.")
                    return
                new_rule['sleep_cycles'] = 0
                self.memory.add_semantic_rule(new_rule, centroid)
                logging.info(f"[Crystallization] New Rule: {new_rule['pattern']} (confidence: {new_rule['confidence']:.2f})")
            
            # SUCCESS: Now compress the episodic memory
            to_delete = set(int(i) for i in cluster_indices) - {core_idx}
            self.memory.episodes = [ep for i, ep in enumerate(self.memory.episodes) if i not in to_delete]
            
            # Rebuild index with remaining episodes
            self.memory.episodic_index = faiss.IndexFlatIP(self.memory.embedding_dim)
            if self.memory.episodes:
                rebuild_vecs = np.array([ep['embedding'] for ep in self.memory.episodes], dtype=np.float32)
                rebuild_vecs = self._normalize_embeddings(rebuild_vecs)
                self.memory.episodic_index.add(rebuild_vecs)
            
            # Run pruning on weak rules and fading episodes
            self._prune_weak_rules()
            self.memory.prune_fading_memories()
            
            self.memory.save()
            logging.info(f"[Sleep] Compressed {len(to_delete)} episodes into 1 rule + 1 representative.")
            
        except Exception as e:
            logging.error(f"[Sleep Phase] Failed: {e}")

    def _rem_sleep_phase(self):
        """REM Sleep: Generative modeling / dreaming.
        
        Stochastically recombines concepts, generates edge-case scenarios,
        and stress-tests existing compiled reflexes.  Rules that fail are
        iteratively corrected via LLM; rules that pass gain confidence.
        
        Per theory doc: "If compiled code fails in a simulated situation,
        its structure is corrected, ensuring exceptional reliability of
        the skill registry."
        """
        logging.info("=== [REM SLEEP] Dreaming — stress-testing existing skills ===")
        if not self.memory.semantic_rules:
            logging.info("[REM] No rules to dream about.")
            return

        import random
        rules_to_test = random.sample(
            self.memory.semantic_rules,
            min(3, len(self.memory.semantic_rules)),
        )

        for rule in rules_to_test:
            pattern = rule.get("pattern", "Unknown")
            python_code = rule.get("python_code", "")
            if not python_code:
                continue

            logging.info(f"[REM] Dreaming about rule '{pattern}' ...")

            # Generate adversarial edge-cases using existing episodes as seed
            related_episodes = [
                ep for ep in self.memory.episodes
                if any(w in ep.get("query", "").lower() for w in pattern.lower().split()[:3])
            ][:3]
            if not related_episodes:
                related_episodes = self.memory.episodes[:3]
            if not related_episodes:
                continue

            try:
                from nare.llm import generate_stress_tests, _validate_skill, repair_skill
                dream_tests = generate_stress_tests(related_episodes)
                if not dream_tests:
                    continue

                scores, error_msg = _validate_skill(python_code, dream_tests)
                old_conf = rule.get("confidence", 0.5)
                
                # Check real-world execution accuracy before penalizing
                real_acc = rule.get('execute_accuracy', 0.0)
                logging.info(f"[REM] Rule '{pattern}' stats: conf={old_conf:.2f}, real_acc={real_acc:.2f}, dream_score={scores['overall']:.2f}")

                if scores["overall"] >= 0.80:
                    boost = min(0.05, (scores["overall"] - 0.80) * 0.25)
                    rule["confidence"] = min(0.99, old_conf + boost)
                    logging.info(f"[REM] '{pattern}' passed dream test (overall={scores['overall']:.2f}). conf: {old_conf:.2f} -> {rule['confidence']:.2f}")
                else:
                    if real_acc >= 0.80:
                        logging.info(f"[REM] '{pattern}' failed dream test (overall={scores['overall']:.2f}), but SKIPPING penalty due to high real accuracy ({real_acc:.2f}).")
                    else:
                        logging.warning(f"[REM] '{pattern}' FAILED dream test (overall={scores['overall']:.2f}). Attempting iterative repair...")
                        repaired_code = repair_skill(python_code, pattern, dream_tests, error_msg, scores, max_attempts=2)
                        
                        if repaired_code and repaired_code != python_code:
                            new_scores, new_err = _validate_skill(repaired_code, dream_tests)
                            if new_scores["overall"] > scores["overall"]:
                                rule["python_code"] = repaired_code
                                rule["confidence"] = max(old_conf * 0.9, new_scores["overall"])
                                rule["rem_repairs"] = rule.get("rem_repairs", 0) + 1
                                logging.info(f"[REM] '{pattern}' REPAIRED successfully! score: {scores['overall']:.2f} -> {new_scores['overall']:.2f}, conf: {old_conf:.2f} -> {rule['confidence']:.2f}")
                            else:
                                penalty = max(0.05, (0.80 - scores["overall"]) * 0.3)
                                rule["confidence"] = max(0.10, old_conf - penalty)
                                rule["maturity"] = max(0, rule.get("maturity", 0) - 1)
                                logging.warning(f"[REM] '{pattern}' repair did not improve. Penalizing: conf {old_conf:.2f} -> {rule['confidence']:.2f}")
                        else:
                            penalty = max(0.05, (0.80 - scores["overall"]) * 0.3)
                            rule["confidence"] = max(0.10, old_conf - penalty)
                            rule["maturity"] = max(0, rule.get("maturity", 0) - 1)
                            logging.warning(f"[REM] '{pattern}' could not be repaired. conf: {old_conf:.2f} -> {rule['confidence']:.2f}")

            except Exception as e:
                logging.error(f"[REM] Dream failed for '{pattern}': {e}")

        # Synaptic downscaling: weaken all graph edges slightly
        self.graph.weaken_all(decay=0.02)
        self.graph.save()
        
        self.memory.save()
        logging.info("=== [REM SLEEP] Dreaming complete ===")

    def _prune_weak_rules(self):
        """Garbage collect rules that are DEAD based on global_score."""
        pruned = []
        kept = []
        for i, rule in enumerate(self.memory.semantic_rules):
            g_score = rule.get('global_score', rule.get('confidence', 0.5))
            cycles = rule.get('sleep_cycles', 0)
            if g_score < 0.40 and cycles >= 2:
                pruned.append(rule.get('pattern', 'Unknown'))
            else:
                kept.append(rule)
        
        if pruned:
            logging.info(f"[Pruning] Garbage collected {len(pruned)} dead rules: {pruned}")
            self.memory.semantic_rules = kept
            # Rebuild semantic index
            self.memory.semantic_index = faiss.IndexFlatIP(self.memory.embedding_dim)
            for rule in kept:
                if 'embedding' in rule:
                    v = np.array(rule['embedding'], dtype=np.float32).flatten().reshape(1, -1)
                    faiss.normalize_L2(v)
                    self.memory.semantic_index.add(v)

    def _save_episode(self, query: str, query_emb: list, best_cand: Dict, prompt: str):
        """Unified episode saving for ALL paths that generate new content."""
        score = best_cand.get('final_score', 0.5)
        if score < 0.50:
            logging.info(f"[Memory] Episode rejected due to low score ({score:.2f} < 0.50)")
            return False
            
        episode_data = {
            "query": query,
            "context": prompt,
            "solution": best_cand['solution'],
            "reasoning_trace": best_cand.get('reasoning', 'N/A'),
            "abstract_signature": best_cand.get('abstract_signature', query),
            "score": score,
            "embedding": query_emb,
        }
        
        # Structure embedding for sleep clustering
        sig = episode_data["abstract_signature"]
        logging.info(f"[Memory] Signature: {sig}")
        if sig and sig != query:
            try:
                episode_data["signature_embedding"] = llm.get_embedding(sig)
            except Exception as e:
                logging.warning(f"Failed to embed abstract signature: {e}")
                episode_data["signature_embedding"] = query_emb
        else:
            episode_data["signature_embedding"] = query_emb
            
        added = self.memory.add_episode(episode_data, np.array([query_emb], dtype=np.float32))
        if not added:
            vec = np.array([query_emb], dtype=np.float32)
            retrieved = self.memory.retrieve_episodes(vec, k=1)
            if retrieved:
                idx = retrieved[0]['memory_id']
                self.memory.episodes[idx]['last_used'] = time.time()
                self.memory.episodes[idx]['strength'] = retrieved[0].get('strength', 1.0) + 0.2
                self.memory.save()
        else:
            # Add graph node and link to similar episodes
            new_idx = len(self.memory.episodes) - 1
            self.graph.add_node(new_idx, label=query[:50])
            vec = np.array([query_emb], dtype=np.float32)
            similar = self.memory.retrieve_episodes(vec, k=3)
            for ep in similar:
                mid = ep.get('memory_id', -1)
                if mid >= 0 and mid != new_idx:
                    self.graph.add_edge(new_idx, mid, weight=ep['similarity'])
                    self.graph.add_edge(mid, new_idx, weight=ep['similarity'])
            self.graph.save()
        return added

    def learn_fact(self, content: str, source: str = "user", category: str = "general") -> bool:
        """Ingest a factual knowledge entry into the RAG layer."""
        embedding = llm.get_embedding(content)
        return self.memory.add_fact(
            {"content": content, "source": source, "category": category},
            np.array(embedding, dtype=np.float32),
        )

    def wait_for_sleep(self, timeout=600):
        """Wait for background sleep phase to complete."""
        start = time.time()
        while hasattr(self, '_is_sleeping') and self._is_sleeping:
            if time.time() - start > timeout:
                logging.warning("[Agent] Timeout waiting for sleep phase.")
                break
            time.sleep(1)
            
    def solve(self, query: str) -> Dict[str, Any]:
        _solve_start = time.time()
        _solve_tokens = 0
        log = []
        log.append(f"Query: {query}")
        
        from nare.sandbox import safe_execute
        
        # =====================================================
        # LAYER 0: EXACT CACHE (O(1) Direct Lookup)
        # =====================================================
        # Check if we have an EXACT match in episodes with a high score
        for ep in self.memory.episodes:
            if ep['query'].strip() == query.strip() and ep.get('score', 0) > 0.8:
                log.append(f"Route: FAST CACHE (Exact match found, score: {ep['score']:.2f})")
                # Update usage metrics for forgetting logic
                ep['last_used'] = time.time()
                ep['strength'] = ep.get('strength', 1.0) + 0.1
                self.memory.save()
                # Record metrics for FAST CACHE hits
                self.metrics.record(
                    query=query, route="FAST",
                    elapsed=time.time() - _solve_start,
                    tokens_used=0,
                    similarity=1.0,
                    answer=ep['solution'],
                    score=ep.get('score', 0.8),
                )
                return {
                    "route_decision": "FAST",
                    "retrieved_memories": [ep],
                    "generated_candidates": [],
                    "critic_evaluation_table": [],
                    "final_answer": ep['solution'],
                    "memory_update_log": log
                }

        # =====================================================
        # LAYER 1: PROGRAMMATIC REFLEXES (AST Execution)
        # =====================================================
        for sem in sorted(self.memory.semantic_rules, key=lambda x: x.get('confidence', 0.5), reverse=True):
            conf = sem.get('confidence', 0.5)
            if conf < 0.40:
                continue
            
            try:
                # Use a single dictionary for exec to avoid NameErrors between functions
                import re as _re, math as _math, json as _json
                from .sandbox import ASTValidator
                builtins_dict = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
                env = {
                    "__builtins__": {k: builtins_dict[k] for k in ASTValidator.ALLOWED_BUILTINS if k in builtins_dict},
                    "re": _re, "math": _math, "json": _json
                }
                exec(sem['python_code'], env)
                
                if 'trigger' in env and env['trigger'](query):
                    logging.info(f"[Reflex] Triggered rule '{sem['pattern']}'")
                    final_answer = safe_execute(sem['python_code'], query)
                    
                    if not final_answer.startswith("Error"):
                        maturity = sem.get('maturity', 0)
                        if conf >= 0.45:
                            # If rule is highly accurate (from sleep validation), we can trust it more
                            if maturity < 3 and sem.get('execute_accuracy', 0.0) < 0.85:
                                # SHADOW VERIFICATION for low-accuracy or very young rules
                                check_prompt = f"### VERIFICATION TASK ###\nTask: {query}\nProposed Answer: {final_answer}\n\nIs this answer mathematically/factually correct for the given task? \nRules:\n1. If the answer is correct (even if concise), respond 'YES'.\n2. If the answer is wrong, respond 'NO'.\n3. Respond ONLY with 'YES' or 'NO' followed by a reason."
                                verification, _ = llm.generate_samples(check_prompt, n=1, temperature=0.0)
                                v_full = verification[0]['solution'].strip()
                                v_ans_line = v_full.split('\n')[0].upper()
                                
                                if "YES" in v_ans_line or (len(v_full) < 10 and "YES" in v_full.upper()):
                                    logging.info(f"[Shadow Mode] ACCEPTED '{sem['pattern']}'")
                                    self._record_skill_result(sem, success=True)
                                    return {
                                        "route_decision": "REFLEX_PROVISIONAL",
                                        "retrieved_memories": [],
                                        "generated_candidates": [],
                                        "critic_evaluation_table": [],
                                        "final_answer": final_answer,
                                        "memory_update_log": log + [f"Route: REFLEX_PROVISIONAL (conf={conf:.2f})"]
                                    }
                                else:
                                    logging.warning(f"[Shadow Mode] REJECTED '{sem['pattern']}'. Reason: {v_full}")
                                    log.append(f"Shadow Check REJECTED rule. Reasoning: {v_full}")
                                    # Reduce confidence slightly but not as harsh as a crash
                                    sem['confidence'] = max(0.1, sem.get('confidence', 0.5) - 0.05)
                                    # Fall through to HYBRID
                            else:
                                # MATURE or HIGH-ACCURACY SKILL: Full reflex speed
                                if maturity < 3:
                                    logging.info(f"[Reflex] Trusting high-accuracy young skill '{sem['pattern']}' (exec_acc={sem.get('execute_accuracy', 0.0):.2f})")
                                
                                self._record_skill_result(sem, success=True)
                                return {
                                    "route_decision": "REFLEX",
                                    "retrieved_memories": [],
                                    "generated_candidates": [],
                                    "critic_evaluation_table": [],
                                    "final_answer": final_answer,
                                    "memory_update_log": log + [f"Route: REFLEX (conf={conf:.2f})"]
                                }
                    else:
                        logging.warning(f"[Reflex] Execution error for '{sem['pattern']}': {final_answer}")
                        self._record_skill_result(sem, success=False)
                        sem['confidence'] = max(0.1, sem.get('confidence', 0.5) - 0.2)
                else:
                    # Optional: log why it didn't trigger if similarity is high
                    pass
            except Exception as e:
                logging.error(f"[Reflex] Crash in rule '{sem.get('pattern')}': {e}")
                self._record_skill_result(sem, success=False)
                log.append(f"Skill crash: {e}")

        # Need the query embedding for LAYER 3 (Episodic Cache)
        query_emb = llm.get_embedding(query)
        query_emb_np = np.array([query_emb], dtype=np.float32)
        
        # =====================================================
        # LAYER 3: SYSTEM 2 (Full LLM reasoning)
        # =====================================================
        retrieved_eps = self.memory.retrieve_episodes(query_emb_np, k=3)
        
        # Graph-augmented retrieval: multi-hop from FAISS hits
        if retrieved_eps:
            start_ids = [ep.get('memory_id', -1) for ep in retrieved_eps if ep.get('memory_id', -1) >= 0]
            graph_ids = self.graph.multi_hop_retrieve(start_ids, hops=2, min_weight=0.3)
            for gid in graph_ids[:2]:
                if gid < len(self.memory.episodes):
                    graph_ep = self.memory.episodes[gid].copy()
                    graph_ep['similarity'] = 0.5  # default for graph-retrieved
                    graph_ep['memory_id'] = gid
                    graph_ep['source'] = 'graph'
                    if gid not in start_ids:
                        retrieved_eps.append(graph_ep)
                        log.append(f"Graph retrieval: added episode {gid} via multi-hop")
            # Strengthen edges between co-retrieved episodes (Hebbian)
            for i, id_a in enumerate(start_ids):
                for id_b in start_ids[i+1:]:
                    self.graph.strengthen_edge(id_a, id_b, delta=0.05)
        
        # RL re-ranking of retrieved episodes
        retrieved_eps = self.rl_retriever.rerank(retrieved_eps)
        
        mem_avg = np.mean([ep.get('score', 0.5) for ep in retrieved_eps]) if retrieved_eps else 0.5
        all_semantics = self.memory.retrieve_semantics(query_emb_np, k=2)
        
        retrieved_semantics = [
            s for s in all_semantics 
            if s['similarity'] > 0.85 and s.get('confidence', 0.5) >= 0.70
        ]
        
        # RAG: Retrieve relevant facts
        retrieved_facts = self.memory.retrieve_facts(query_emb_np, k=3)
        if retrieved_facts:
            log.append(f"RAG: Retrieved {len(retrieved_facts)} relevant facts.")
        
        max_sim = retrieved_eps[0]['similarity'] if retrieved_eps else 0.0
        
        # (Entropy check removed as it was sabotaging high-certainty structural matches)
        
        route = "SLOW"
        alpha = 0.0
        candidates = []
        final_answer = ""
        best_cand = None
        prompt_used = ""
        
        if max_sim >= self.tau_fast:
            route = "FAST"
            alpha = max_sim
            log.append(f"Route: FAST PATH (sim: {max_sim:.3f} >= {self.tau_fast})")
            
            best_cand = retrieved_eps[0].copy()
            best_cand['final_score'] = alpha
            final_answer = best_cand['solution']
            
            if self._save_episode(query, query_emb, best_cand, "FAST PATH"):
                log.append("Saved FAST episode to memory.")
                
        elif max_sim >= self.tau_hybrid and retrieved_eps:
            route = "HYBRID"
            alpha = max_sim
            log.append(f"Route: HYBRID PATH (sim: {max_sim:.3f})")
            
            prompt_used = f"Task: {query}\n\n"
            prompt_used += "--- RETRIEVED PAST EXPERIENCE ---\n"
            prompt_used += f"Similar Past Task: {retrieved_eps[0]['query']}\n"
            prompt_used += f"Past Reasoning: {retrieved_eps[0]['reasoning_trace']}\n"
            prompt_used += f"Past Solution: {retrieved_eps[0]['solution']}\n\n"
            if retrieved_facts:
                prompt_used += "--- RELEVANT FACTUAL KNOWLEDGE (RAG) ---\n"
                for fact in retrieved_facts:
                    prompt_used += f"Fact: {fact.get('content', '')[:300]}\n"
                prompt_used += "\n"
            prompt_used += "Your task is SKILL FORMATION and REASONING COMPRESSION. Because you have already solved a similar task, DO NOT write a full step-by-step trace from scratch. Skip the basics. Provide a highly compressed reasoning trace that only addresses the DIFFERENCES between the Past Task and the New Task. Reuse computation as a reflex."
            
            candidates, _htokens = llm.generate_samples(prompt_used, n=1, mode="HYBRID")
            _solve_tokens += _htokens
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
            
            # Meta-abduction: inject cross-domain meta-rules as hints
            meta_rules = self.meta_engine.get_applicable_meta_rules(query)
            if meta_rules:
                prompt_used += "--- CROSS-DOMAIN META-RULES (from meta-abduction) ---\n"
                for mr in meta_rules[:2]:
                    prompt_used += f"Meta-Rule: {mr['name']}\n"
                    prompt_used += f"Pattern: {mr['abstract_pattern']}\n"
                    prompt_used += f"Common Operations: {', '.join(mr.get('common_operations', []))}\n---\n"
                log.append(f"Meta-abduction: applied {len(meta_rules)} meta-rules")
            
            if retrieved_eps:
                prompt_used += "--- RELEVANT EPISODIC MEMORIES (Use as analogies) ---\n"
                for ep in retrieved_eps:
                    prompt_used += f"Past Task: {ep['query']}\n"
                    prompt_used += f"Past Reasoning: {ep['reasoning_trace']}\n"
                    prompt_used += f"Past Solution: {ep['solution']}\n---\n"
            
            if retrieved_facts:
                prompt_used += "--- RELEVANT FACTUAL KNOWLEDGE (RAG) ---\n"
                for fact in retrieved_facts:
                    prompt_used += f"Fact: {fact.get('content', '')[:300]}\n"
                prompt_used += "---\n"
            prompt_used += "\nSolve the new Task by synthesizing insights from the provided memories (if any) and applying deep reasoning."
                
            # Use Tree-of-Thoughts for SLOW path (BFS with pruning)
            candidates, _stokens = llm.tree_of_thoughts(prompt_used, breadth=3, depth=1)
            _solve_tokens += _stokens
            candidates = self.critic.evaluate(query, candidates)
            
            best_cand = candidates[0] if candidates else None
            
            if best_cand:
                final_answer = best_cand['solution']
                if best_cand.get('final_score', 0.5) >= max(0.70, mem_avg) and self._save_episode(query, query_emb, best_cand, prompt_used):
                    log.append(f"Saved SLOW episode to memory (score {best_cand.get('final_score', 0.5):.2f} >= mem_avg {mem_avg:.2f}).")

        # 3. Calibration
        if best_cand:
            self._calibrate_tau(reward=best_cand.get('final_score', 0.5), fast_path_used=(route == "FAST"))

        result = {
            "route_decision": route,
            "retrieved_memories": retrieved_eps + retrieved_semantics,
            "generated_candidates": candidates,
            "critic_evaluation_table": candidates if candidates else ([best_cand] if best_cand else []),
            "final_answer": final_answer,
            "memory_update_log": log
        }

        # Record metrics
        _outcome_score = best_cand.get('final_score', 0.5) if best_cand else 0.0
        self.metrics.record(
            query=query, route=route,
            elapsed=time.time() - _solve_start,
            tokens_used=_solve_tokens,
            similarity=alpha,
            answer=final_answer,
            score=_outcome_score,
        )
        
        # RL Retriever feedback: update value function based on outcome
        if retrieved_eps:
            r_ids = [ep.get('memory_id', 0) for ep in retrieved_eps if 'embedding' in ep]
            r_embs = [np.array(ep['embedding'], dtype=np.float32) for ep in retrieved_eps if 'embedding' in ep]
            if r_ids and r_embs:
                self.rl_retriever.batch_update(r_ids, r_embs, _outcome_score)
                self.rl_retriever.save()

        # 4. Sleep Phase — DECOUPLED from solve() latency
        # Run in background thread so answers return immediately.
        if self._check_sleep_trigger() and not getattr(self, '_is_sleeping', False):
            self._is_sleeping = True
            import threading
            def sleep_wrapper():
                try:
                    self._sleep_phase()      # NREM: consolidation
                    # RELEASE BLOCKING LOCK: NREM is done, the main thread can continue.
                    # REM sleep will continue in the background without blocking benchmarks.
                    self._is_sleeping = False
                    
                    self._rem_sleep_phase()   # REM: dreaming / stress-testing
                    # Titans/MIRAS: Neural memory consolidation
                    self.neural_memory.consolidate(self.memory.episodes[-50:])
                    self.neural_memory.save()
                    # Meta-abduction: discover cross-domain patterns
                    self.meta_engine.analyze_skills(
                        self.memory.semantic_rules, self.memory.episodes
                    )
                except Exception as e:
                    logging.error(f"[Sleep Wrapper] Crash: {e}")
                finally:
                    self._is_sleeping = False
            t = threading.Thread(target=sleep_wrapper, daemon=True)
            t.start()
            self._sleep_thread = t  # Keep reference for joining if needed

        return result

    def _record_skill_result(self, rule: dict, success: bool):
        """Track skill execution history for lifecycle management."""
        history = rule.get('score_history', [])
        history.append(1.0 if success else 0.0)
        # Keep last 20 results
        rule['score_history'] = history[-20:]
        
        rule['reuse_rate'] = rule.get('reuse_rate', 0) + 1
        
        # Update confidence based on rolling success rate
        if len(history) >= 3:
            recent = history[-10:]  # Last 10 executions
            rolling_success = sum(recent) / len(recent)
            
            # Dynamic Confidence Promotion: success breeds stability
            old_conf = rule.get('confidence', 0.5)
            # Increase confidence by 0.01 on success, capped at 0.99
            rule['confidence'] = min(0.99, round((rolling_success * 0.7) + (old_conf * 0.3) + 0.01, 3))
            
        # Maturity update: increment maturity after successful combat trials
        if success:
            rule['success_streak'] = rule.get('success_streak', 0) + 1
            if rule['success_streak'] >= 5:
                rule['maturity'] = rule.get('maturity', 0) + 1
                rule['success_streak'] = 0
                logging.info(f"[Evolution] Skill '{rule['pattern']}' reached maturity level {rule['maturity']}")
        else:
            rule['success_streak'] = 0
            rule['maturity'] = max(0, rule.get('maturity', 0) - 1)

        # Global Score: mix of confidence and maturity
        maturity_bonus = min(0.3, rule.get('maturity', 0) * 0.1)
        conf = rule.get('confidence', 0.5)
        rule['global_score'] = round((conf * 0.7) + maturity_bonus, 3)
        
        # Find and update rule in memory
        for i, r in enumerate(self.memory.semantic_rules):
            if r.get('pattern') == rule.get('pattern'):
                self.memory.semantic_rules[i] = rule
                self.memory.save()
                break

