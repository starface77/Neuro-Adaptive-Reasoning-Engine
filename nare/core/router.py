import time
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Callable, Optional, Tuple
from ..reasoning import llm
from ..memory.memory import MemorySystem
from ..reasoning.critic import Critic
from ..execution.sandbox import SecurityError, safe_call_trigger, safe_call_execute_in_namespace, safe_execute_freeform, extract_python_block
from ..config import NareConfig
from .synthesis import verified_synthesis
from ..tools.solve_context import SolveContext
from ..tools.domain_detector import get_adaptive_tau_fast

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
        critic: Critic,
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
        _cached_for_verification = None  # For memory stability check

        # Adaptive tau_fast based on domain detection
        adaptive_tau_fast = get_adaptive_tau_fast(query, self.config)
        logging.info(f"[ROUTER] Adaptive tau_fast: {adaptive_tau_fast:.2f} (base: {self.tau_fast:.2f})")

        # --- LAYER -1: COMPILED SKILLS (before FAST) ---
        query_emb = llm.get_embedding(query)
        skills = self.memory.retrieve_skills(query_emb, k=1)

        if skills and skills[0].get('similarity', 0) >= 0.90:  # High threshold for skills
            skill = skills[0]
            log.append(f"Route: COMPILED_SKILL (pattern: {skill['pattern']})")
            try:
                from ..sandbox import safe_execute_freeform
                answer = safe_execute_freeform(skill['code'])
                if answer and not answer.startswith("Error:"):
                    # Update skill use count
                    skill_id = skill.get('skill_id', 0)
                    if 0 <= skill_id < len(self.memory.compiled_skills):
                        self.memory.compiled_skills[skill_id]['use_count'] += 1
                        self.memory.save()

                    self.metrics.record(
                        query=query, route="COMPILED_SKILL",
                        elapsed=time.time() - _solve_start,
                        tokens_used=0,
                        similarity=skill['similarity'],
                        answer=answer,
                        score=1.0
                    )
                    return self._wrap_result("COMPILED_SKILL", answer, [skill], [], log, skill['similarity'], _solve_start, 0)
            except Exception as e:
                log.append(f"Skill execution failed: {e}")

        # --- LAYER 0: FAST HNSW Cache ---
        if self.memory.episodic_index.ntotal > 0:
            fast_emb = query_emb  # Reuse embedding
            fast_vec = np.array([fast_emb], dtype=np.float32)
            faiss.normalize_L2(fast_vec)
            sims, indices = self.memory.episodic_index.search(fast_vec, 1)

            # FAST path: high similarity with verified solution
            if sims[0][0] >= adaptive_tau_fast:
                idx = int(indices[0][0])
                if 0 <= idx < len(self.memory.episodes):
                    ep = self.memory.episodes[idx]
                    logging.info(f"[ROUTER] FAST candidate found: score={ep.get('score', 0)}")
                    if ep.get('score', 0) >= 0.80:  # Only use verified solutions
                        fast_answer = self._post_process_answer(ep.get('solution', ''), "FAST", log)

                        # CRITICAL: Re-validate cached answer with oracle if available
                        if oracle:
                            ok, info = oracle(query, fast_answer)
                            if not ok:
                                logging.warning(f"[ROUTER] FAST cache INVALID: {info}")
                                # Downgrade episode score - it was a false positive
                                ep['score'] = max(0.3, ep.get('score', 0.8) - 0.3)
                                ep['validation_failures'] = ep.get('validation_failures', 0) + 1
                                self.memory.save()
                                # Fall through to SLOW path
                            else:
                                logging.info(f"[ROUTER] FAST cache validated by oracle")
                                # Increment access count for successful reuse
                                ep['access_count'] = ep.get('access_count', 0) + 1
                                self.memory.save()

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
                            # No oracle - trust the cache but mark as unvalidated
                            logging.info(f"[ROUTER] Taking FAST route (no oracle validation)")
                            ep['access_count'] = ep.get('access_count', 0) + 1
                            self.memory.save()

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
                    logging.info(f"[ROUTER] FAST idx out of bounds: {idx} >= {len(self.memory.episodes)}")

        # --- LAYER 1: REFLEX (Skills) ---
        reflex_result = self._try_reflex_path(query, oracle, log, _solve_start)
        if reflex_result:
            return reflex_result

        query_emb_np = np.array([query_emb], dtype=np.float32)
        retrieved_eps = self.memory.retrieve_episodes(query_emb_np, k=3)
        max_sim = max((float(r.get('similarity', 0.0)) for r in retrieved_eps), default=0.0) if retrieved_eps else 0.0

        alpha_t = max_sim

        # HYBRID path: moderate similarity - use cached solution as template
        if max_sim >= self.tau_hybrid and retrieved_eps:
            log.append(f"Route: HYBRID PATH (sim: {max_sim:.3f})")
            prompt = self._build_hybrid_prompt(query, retrieved_eps[0])
            logging.info(f"[HYBRID] Full prompt:\n{prompt}\n---END PROMPT---")
            candidates, h_tokens = llm.generate_samples(prompt, n=1, mode="ADAPTIVE")
            logging.info(f"[HYBRID] LLM returned {len(candidates)} candidates, {h_tokens} tokens")
            if candidates:
                logging.info(f"[HYBRID] LLM solution: {candidates[0].get('solution', '')[:300]}")
            _solve_tokens += h_tokens
            candidates = self.critic.evaluate(query, candidates, oracle=oracle)
            logging.info(f"[HYBRID] After critic: {len(candidates)} candidates")
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

        log.append(f"Route: SLOW PATH (sim: {max_sim:.3f})")

        #  Assess task complexity and adjust parameters
        # Only on first encounter (max_sim < 0.3 = truly novel task)
        adaptive_params = self._assess_task_complexity(query) if max_sim < 0.3 else {}

        # CRITICAL: Override temperature for SWE-bench precision
        # Adaptive assessment suggests 0.5-0.9 for complexity, but for code generation
        # we need surgical precision (0.1-0.2), not creativity
        if 'temperature' in adaptive_params:
            adaptive_params['temperature'] = 0.1
            logging.info(f"[ROUTER] Overriding temperature to 0.1 for code precision")

        prompt = self._build_slow_prompt(query, retrieved_eps)

        # Use Verified Synthesis when oracle is available (theory.md requirement)
        if oracle:
            max_attempts = adaptive_params.get('max_attempts', self.config.synthesis.max_attempts)

            # Create SolveContext for component coordination
            solve_ctx = SolveContext(query=query, oracle=oracle)

            # For SWE-bench: use the full prompt (already formatted with File: instructions)
            # instead of letting VS create code-execution prompt
            vs_query = prompt  # Use the formatted SLOW prompt, not raw query

            vs_result = verified_synthesis(
                query=vs_query,
                propose_fn=lambda p, priors: self._propose_for_vs(p, priors, llm, adaptive_params),
                oracle=oracle,
                max_attempts=max_attempts,
                expected_hint=expected_hint,
                context=solve_ctx
            )

            # Determine final score based on convergence and IoU
            if vs_result.converged:
                final_score = 0.95
            elif solve_ctx.best_iou >= 0.95:
                # Very close but didn't converge - still valuable
                final_score = solve_ctx.best_iou
            elif solve_ctx.best_iou >= 0.80:
                # Good solution even without full convergence
                final_score = solve_ctx.best_iou
            else:
                final_score = 0.30

            candidates = [{
                'solution': vs_result.final_answer,
                'reasoning_trace': f"VS converged in {vs_result.total_attempts} attempts (adaptive: {adaptive_params}, best_iou: {solve_ctx.best_iou:.2f})",
                'final_score': final_score,
                'solve_context': solve_ctx,  # Pass context for memory saving
            }]
        else:
            # Fallback to best-of-N when no oracle
            breadth = adaptive_params.get('breadth', self.config.synthesis.slow_path_breadth)
            candidates, s_tokens = llm.best_of_n_with_prescore(prompt, breadth=breadth)
            _solve_tokens += s_tokens
            candidates = self.critic.evaluate(query, candidates, oracle=oracle)

        best = candidates[0] if candidates else None
        if best:
            best['solution'] = self._post_process_answer(best['solution'], "SLOW", log)

            # CRITICAL: Path Validation (prevent hallucinated paths)
            # NOTE: Disabled for SWE-bench - validation happens in swe_bench_official.py
            # because each task has different repo_path
            from ..path_validator import check_solution_paths, suggest_corrections

            # Only validate if project_root is current directory (not SWE-bench)
            import os
            if os.path.exists('.git'):  # We're in a real project
                path_check = check_solution_paths(best['solution'])
            else:
                path_check = {'hallucination_detected': False}

            if path_check.get('hallucination_detected', False):
                invalid = path_check['invalid_paths']
                logging.warning(f"[ROUTER] Hallucinated paths detected: {invalid}")
                log.append(f"Path Validation: Found {len(invalid)} invalid paths")

                # Get suggestions
                suggestions = suggest_corrections(invalid)

                # Build correction prompt
                correction_prompt = f"""Task: {query}

YOUR SOLUTION CONTAINS INVALID FILE PATHS:
{best['solution'][:500]}

INVALID PATHS DETECTED:
{', '.join(invalid)}

SUGGESTIONS:
"""
                for inv, sugg in suggestions.items():
                    correction_prompt += f"- {inv} → {sugg}\n"

                correction_prompt += """
Please provide a CORRECTED solution with valid file paths.
Check if the file should be:
1. A package __init__.py (e.g., django/db/models/fields/__init__.py)
2. A different file in the same directory
3. A file in a parent/child directory

Provide your corrected answer with ONLY valid paths."""

                # Single retry with path correction
                retry_candidates, retry_tokens = llm.generate_samples(correction_prompt, n=1, temperature=0.5, mode="ANALYTIC")
                _solve_tokens += retry_tokens

                if retry_candidates:
                    retry_solution = self._post_process_answer(retry_candidates[0]['solution'], "SLOW-PATH-FIX", log)
                    retry_check = check_solution_paths(retry_solution)

                    if not retry_check['hallucination_detected']:
                        logging.info(f"[ROUTER] Path correction SUCCESS")
                        log.append(f"Path Validation: SUCCESS - all paths valid")
                        best['solution'] = retry_solution
                    else:
                        logging.warning(f"[ROUTER] Path correction FAILED - still has invalid paths")
                        log.append(f"Path Validation: FAILED - keeping original")

            # CRITICAL: Self-Correction Loop
            # If oracle fails, extract error info and retry once
            if oracle:
                ok, info = oracle(query, best['solution'])
                if not ok and isinstance(info, str):
                    # Oracle failed - try self-correction
                    logging.warning(f"[ROUTER] Oracle failed: {info[:200]}")
                    log.append(f"Self-Correction: Oracle failed, attempting retry")

                    # Build correction prompt with error feedback
                    correction_prompt = f"""Task: {query}

YOUR PREVIOUS ATTEMPT FAILED:
{best['solution'][:500]}

ERROR FROM ORACLE:
{info}

ANALYSIS:
The oracle indicates your solution is incorrect. Common issues:
- Wrong file paths (check if files exist in the project)
- Missing files (e.g., __init__.py in packages)
- Incorrect module names

Please provide a CORRECTED solution. Think carefully about:
1. Are all file paths correct and existing?
2. Did you check __init__.py files for Django apps?
3. Are you targeting the right module?

Provide your corrected answer."""

                    # Single retry attempt
                    retry_candidates, retry_tokens = llm.generate_samples(correction_prompt, n=1, temperature=0.7, mode="ANALYTIC")
                    _solve_tokens += retry_tokens

                    if retry_candidates:
                        retry_solution = self._post_process_answer(retry_candidates[0]['solution'], "SLOW-RETRY", log)
                        ok_retry, info_retry = oracle(query, retry_solution)

                        if ok_retry:
                            logging.info(f"[ROUTER] Self-correction SUCCESS")
                            log.append(f"Self-Correction: SUCCESS on retry")
                            best['solution'] = retry_solution
                            best['final_score'] = 0.85  # Good but not perfect
                        else:
                            logging.warning(f"[ROUTER] Self-correction FAILED: {info_retry[:100]}")
                            log.append(f"Self-Correction: FAILED on retry")

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

    def _propose_for_vs(self, prompt, priors, llm_mod, adaptive_params=None):
        adaptive_params = adaptive_params or {}
        # Use temperature from adaptive_params (already overridden to 0.1 in router)
        # Default to 0.1 if not set
        temp = adaptive_params.get('temperature', 0.1)
        # Use SWE mode for file generation (not ANALYTIC mode with <solution> tags)
        cands, _ = llm_mod.generate_samples(prompt, n=1, temperature=temp, mode="SYNTHESIS")
        return cands[0]['solution'] if cands else ""

    def _assess_task_complexity(self, query: str) -> Dict[str, Any]:
        """Assess task complexity and return adaptive parameters.

        Uses LLM to analyze query and determine:
        - max_attempts: How many VS iterations needed
        - breadth: How many candidates to generate
        - temperature: Sampling temperature

        Only called on first encounter (max_sim < 0.3).
        """
        assessment_prompt = f"""Analyze this task and assess its complexity:

Task: {query}

Respond with JSON only:
{{
  "complexity": "simple|medium|hard|extreme",
  "reasoning": "brief explanation",
  "max_attempts": 3-12,
  "breadth": 3-8,
  "temperature": 0.5-0.9
}}

Simple: straightforward logic, 3 attempts, breadth 3, temp 0.5
Medium: multi-step reasoning, 5 attempts, breadth 5, temp 0.7
Hard: complex algorithms, 8 attempts, breadth 6, temp 0.8
Extreme: novel patterns, 12 attempts, breadth 8, temp 0.9"""

        try:
            # Quick assessment with low temperature
            cands, _ = llm.generate_samples(assessment_prompt, n=1, temperature=0.3, mode="DIRECT")
            if not cands:
                return {}

            response = cands[0].get('solution', '{}')

            # Extract JSON
            import json
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group(0))
                logging.info(f"[ADAPTIVE] Task complexity: {params.get('complexity')} - {params.get('reasoning')}")
                return {
                    'max_attempts': params.get('max_attempts', 5),
                    'breadth': params.get('breadth', 5),
                    'temperature': params.get('temperature', 0.7)
                }
        except Exception as e:
            logging.warning(f"[ADAPTIVE] Assessment failed: {e}")

        # Fallback to defaults
        return {}

    def _build_hybrid_prompt(self, query, ep):
        p = f"Task: {query}\n\n"

        # Inject learned rules
        query_emb = llm.get_embedding(query)
        learned_rules = self.memory.retrieve_semantics(query_emb, k=2)

        if learned_rules:
            p += "--- LEARNED RULES ---\n"
            for rule in learned_rules:
                if rule.get('confidence', 0) >= 0.50:
                    pattern = rule.get('pattern', 'Unknown')
                    code = rule.get('python_code', '')
                    import re
                    file_paths = re.findall(r'["\']([a-z_/]+\.py)["\']', code)
                    if file_paths:
                        p += f"RULE: {pattern} → Common paths: {', '.join(set(file_paths))}\n"
            p += "---\n\n"

        p += f"Similar Past Task: {ep['query']}\nPast Solution: {ep['solution']}\n\n"
        p += "Provide a compressed delta-reasoning. Adapt the logic from the past solution to the NEW task.\n"
        p += "DO NOT blindly copy file paths, names, or values from the past task.\n"
        p += "Use the LEARNED RULES above to guide your file path selection.\n"
        p += "You MUST use the exact paths and variables required for the NEW task."
        return p

    def _build_slow_prompt(self, query, retrieved_eps):
        p = f"Task: {query}\n\n"

        # CRITICAL: Inject learned rules from Library Learning
        query_emb = llm.get_embedding(query)
        learned_rules = self.memory.retrieve_semantics(query_emb, k=3)

        if learned_rules:
            p += "--- LEARNED RULES (from past similar tasks) ---\n"
            for rule in learned_rules:
                if rule.get('confidence', 0) >= 0.50:  # Only high-confidence rules
                    pattern = rule.get('pattern', 'Unknown')
                    # Extract key insights from the rule code
                    code = rule.get('python_code', '')
                    # Try to extract file paths or patterns from the code
                    import re
                    file_paths = re.findall(r'["\']([a-z_/]+\.py)["\']', code)
                    if file_paths:
                        p += f"RULE: {pattern}\n"
                        p += f"  Common file paths: {', '.join(set(file_paths))}\n"
                        p += f"  Confidence: {rule.get('confidence', 0):.2f}\n"
                    else:
                        p += f"RULE: {pattern} (confidence: {rule.get('confidence', 0):.2f})\n"
            p += "---\n\n"

        # CRITICAL: Aggressive Django-specific rules
        query_lower = query.lower()
        if 'django' in query_lower:
            p += "--- MANDATORY DJANGO RULES ---\n"
            if 'model' in query_lower or 'field' in query_lower:
                p += "⚠ CRITICAL: Django models/fields issues ALWAYS check:\n"
                p += "  1. django/db/models/fields/__init__.py (NOT fields.py)\n"
                p += "  2. The app's __init__.py for registration\n"
                p += "  3. django/db/models/__init__.py for imports\n"
            if 'migration' in query_lower or 'migrate' in query_lower:
                p += "⚠ CRITICAL: Django migration issues ALWAYS check:\n"
                p += "  1. django/core/management/commands/sqlmigrate.py\n"
                p += "  2. django/db/migrations/ directory\n"
                p += "  3. The app's migrations/ folder\n"
            if 'admin' in query_lower or 'register' in query_lower:
                p += "⚠ CRITICAL: Django admin issues ALWAYS check:\n"
                p += "  1. django/contrib/admin/__init__.py\n"
                p += "  2. The app's admin.py\n"
            if 'setting' in query_lower or 'config' in query_lower:
                p += "⚠ CRITICAL: Django settings issues ALWAYS check:\n"
                p += "  1. django/conf/global_settings.py\n"
                p += "  2. django/conf/__init__.py\n"
            p += "---\n\n"

        if retrieved_eps:
            p += "--- RELEVANT MEMORIES ---\n"
            for ep in retrieved_eps:
                p += f"Past Task: {ep['query']}\nSolution: {ep['solution']}\n---\n"

        p += "\nIMPORTANT: Use the learned rules above as MANDATORY guidance, not suggestions.\n"
        p += "For Django tasks, you MUST check __init__.py files in packages.\n"
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
