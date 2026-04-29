import time
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Callable, Optional, Tuple
from .. import llm
from ..memory import MemorySystem
from ..critic import HybridCritic
from ..sandbox import SecurityError, safe_call_trigger, safe_call_execute_in_namespace, safe_execute_freeform, extract_python_block
from ..config import NareConfig
from ..synthesis import verified_synthesis

class ReasoningRouter:
    """The central routing engine for NARE.
    
    Orchestrates the 4-tier reasoning pipeline:
    1. FAST (HNSW Cache)
    2. REFLEX (Programmatic Skills)
    3. HYBRID (Delta Reasoning)
    4. SLOW (Verified Synthesis / Best-of-N)
    """

    def __init__(
        self, 
        memory: MemorySystem, 
        critic: HybridCritic, 
        config: NareConfig,
        metrics: Any
    ):
        self.memory = memory
        self.critic = critic
        self.config = config
        self.metrics = metrics
        
        self.tau_fast = config.routing.tau_fast
        self.tau_hybrid = config.routing.tau_hybrid

    def route(
        self, 
        query: str, 
        oracle: Optional[Callable] = None, 
        expected_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        _solve_start = time.time()
        _solve_tokens = 0
        log = []
        
        # --- LAYER 0: FAST HNSW Cache ---
        if self.memory.episodic_index.ntotal > 0:
            fast_emb = llm.get_embedding(query)
            fast_vec = np.array([fast_emb], dtype=np.float32)
            faiss.normalize_L2(fast_vec)
            sims, indices = self.memory.episodic_index.search(fast_vec, 1)
            
            logging.info(f"[ROUTER] FAST path check: sim={sims[0][0]:.4f}, tau_fast={self.tau_fast}")
            if sims[0][0] >= self.tau_fast:
                idx = int(indices[0][0])
                if 0 <= idx < len(self.memory.episodes):
                    ep = self.memory.episodes[idx]
                    logging.info(f"[ROUTER] FAST candidate found: score={ep.get('score', 0)}")
                    if ep.get('score', 0) > 0.5:
                        fast_answer = self._post_process_answer(ep.get('solution', ''), "FAST", log)
                        
                        oracle_rejected = False
                        if oracle:
                            try:
                                ok, info = oracle(query, fast_answer)
                                logging.info(f"[ROUTER] FAST oracle check: ok={ok}, info={info}")
                                if not ok: oracle_rejected = True
                            except Exception as e:
                                logging.info(f"[ROUTER] FAST oracle exception: {e}")
                                oracle_rejected = True
                        
                        if not oracle_rejected:
                            logging.info(f"[ROUTER] Taking FAST route")
                            self.metrics.record(
                                query=query, route="FAST",
                                elapsed=time.time() - _solve_start,
                                tokens_used=0,
                                similarity=float(sims[0][0]),
                                answer=fast_answer,
                                score=ep.get('score', 0.8),
                            )
                            return self._wrap_result("FAST", fast_answer, [ep], [], log, float(sims[0][0]), _solve_start, 0)
                        else:
                            logging.info(f"[ROUTER] FAST candidate rejected by oracle.")
                else:
                    logging.info(f"[ROUTER] FAST idx out of bounds: {idx} >= {len(self.memory.episodes)}")

        # --- LAYER 1: REFLEX (Skills) ---
        reflex_result = self._try_reflex_path(query, oracle, log, _solve_start)
        if reflex_result:
            return reflex_result

        # --- PREP FOR LAYER 2/3 ---
        query_emb = llm.get_embedding(query)
        query_emb_np = np.array([query_emb], dtype=np.float32)
        retrieved_eps = self.memory.retrieve_episodes(query_emb_np, k=3)
        max_sim = max((float(r.get('similarity', 0.0)) for r in retrieved_eps), default=0.0) if retrieved_eps else 0.0
        
        kappa = self.config.amortization.kappa
        memory_size = len(self.memory.episodes)
        alpha_t = 1.0 - np.exp(-kappa * memory_size)
        
        # --- LAYER 2: HYBRID (Delta) ---
        if max_sim >= self.tau_hybrid and retrieved_eps:
            log.append(f"Route: HYBRID PATH (sim: {max_sim:.3f})")
            prompt = self._build_hybrid_prompt(query, retrieved_eps[0])
            candidates, h_tokens = llm.generate_samples(prompt, n=1, mode="HYBRID")
            _solve_tokens += h_tokens
            candidates = self.critic.evaluate(query, candidates, oracle=oracle)
            if candidates:
                best = candidates[0]
                best['solution'] = self._post_process_answer(best['solution'], "HYBRID", log)
                best['final_score'] = (max_sim * 1.0) + ((1 - max_sim) * best['final_score'])
                
                self.metrics.record(
                    query=query, route="HYBRID",
                    elapsed=time.time() - _solve_start,
                    tokens_used=_solve_tokens,
                    similarity=max_sim,
                    answer=best['solution'],
                    score=best['final_score'],
                )
                return self._wrap_result("HYBRID", best['solution'], retrieved_eps, candidates, log, max_sim, _solve_start, _solve_tokens, alpha_t)

        # --- LAYER 3: SLOW (Verified Synthesis) ---
        log.append(f"Route: SLOW PATH (sim: {max_sim:.3f})")
        prompt = self._build_slow_prompt(query, retrieved_eps)
        
        if oracle:
            vs_result = verified_synthesis(
                query=query,
                propose_fn=lambda p, priors: self._propose_for_vs(p, priors, llm),
                oracle=oracle,
                max_attempts=self.config.synthesis.max_attempts,
                expected_hint=expected_hint
            )
            candidates = [{
                'solution': vs_result.final_answer,
                'reasoning_trace': f"VS converged in {vs_result.total_attempts} attempts",
                'final_score': 0.95 if vs_result.converged else 0.30,
            }]
        else:
            candidates, s_tokens = llm.best_of_n_with_prescore(prompt, breadth=3)
            _solve_tokens += s_tokens
            candidates = self.critic.evaluate(query, candidates, oracle=oracle)

        best = candidates[0] if candidates else None
        if best:
            best['solution'] = self._post_process_answer(best['solution'], "SLOW", log)
            self.metrics.record(
                query=query, route="SLOW",
                elapsed=time.time() - _solve_start,
                tokens_used=_solve_tokens,
                similarity=max_sim,
                answer=best['solution'],
                score=best.get('final_score', 0.5),
            )
            return self._wrap_result("SLOW", best['solution'], retrieved_eps, candidates, log, max_sim, _solve_start, _solve_tokens, alpha_t)

        return self._wrap_result("ERROR", "No solution found", [], [], log, 0.0, _solve_start, _solve_tokens)

    def _try_reflex_path(self, query: str, oracle, log, start_time):
        rules = self.memory.retrieve_semantics(llm.get_embedding(query), k=3)
        for rule in rules:
            if rule.get('confidence', 0) < self.config.routing.tau_reflex: continue
            if safe_call_trigger(rule['python_code'], query):
                log.append(f"Route: REFLEX (Skill: {rule['pattern']})")
                try:
                    ans = safe_call_execute_in_namespace(rule['python_code'], query)
                    if ans and not ans.startswith("Error:"):
                        self.metrics.record(query=query, route="REFLEX", elapsed=time.time()-start_time, tokens_used=0, similarity=1.0, answer=ans, score=rule['confidence'])
                        return self._wrap_result("REFLEX", ans, [rule], [], log, 1.0, start_time, 0)
                except Exception as e:
                    log.append(f"Reflex failed: {e}")
        return None

    def _post_process_answer(self, raw: str, route: str, log: list) -> str:
        if not raw: return raw
        py_block = extract_python_block(raw)
        if not py_block: return raw
        try:
            executed = safe_execute_freeform(py_block)
            if executed and not executed.startswith("Error:"):
                log.append(f"[{route}] Executed inline code block.")
                return executed
        except: pass
        return raw

    def _propose_for_vs(self, prompt, priors, llm_mod):
        temp = 0.0 if not priors else min(0.3 + 0.2 * len(priors), 1.0)
        cands, _ = llm_mod.generate_samples(prompt, n=1, temperature=temp, mode="SLOW")
        return cands[0]['solution'] if cands else ""

    def _build_hybrid_prompt(self, query, ep):
        return f"Task: {query}\n\nSimilar Past Task: {ep['query']}\nPast Solution: {ep['solution']}\n\nProvide a compressed delta-reasoning."

    def _build_slow_prompt(self, query, retrieved_eps):
        p = f"Task: {query}\n\n"
        if retrieved_eps:
            p += "--- RELEVANT MEMORIES ---\n"
            for ep in retrieved_eps:
                p += f"Past Task: {ep['query']}\nSolution: {ep['solution']}\n---\n"
        p += "Solve the task with deep reasoning."
        return p

    def _wrap_result(self, route, answer, memories, candidates, log, alpha, start_time, tokens, alpha_t=0.0):
        return {
            "route_decision": route,
            "final_answer": answer,
            "retrieved_memories": memories,
            "generated_candidates": candidates,
            "memory_update_log": log,
            "alpha": float(alpha),
            "alpha_t": float(alpha_t),
            "novelty": 0.0,
            "elapsed": time.time() - start_time,
            "tokens": tokens
        }
