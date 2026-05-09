import time
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Callable, Optional, Tuple
from ...reasoning import llm
from ...memory.engine import MemorySystem
from ...reasoning.generation.ranker import Critic
from ...execution.sandboxes.base import SecurityError, safe_call_trigger, safe_call_execute_in_namespace, safe_execute_freeform, extract_python_block, safe_load_module
from ...config import NareConfig
from ..synthesis.engine import verified_synthesis
from ..solve_context import SolveContext
from .detector import get_adaptive_tau_fast
from ...agents.planning import PlanningAgent
from ...memory.cache import ReasoningCache
from .metrics import RouteMetrics
from ...core.events import ToolStart, ToolEnd

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
        metrics: Any,
        evolution: Optional[Any] = None,
        bus: Optional[Any] = None
    ):
        self.memory = memory
        self.critic = critic
        self.config = config
        self.metrics = metrics
        self.evolution = evolution
        self.bus = bus

        self.tau_fast = config.routing.tau_fast
        self.tau_hybrid = config.routing.tau_hybrid

        self.planner = PlanningAgent()

        self.reasoning_cache = ReasoningCache(
            cache_dir=".nare_cache",
            ttl=3600
        )

        self.route_metrics = RouteMetrics()

        # Amortization tracking
        self._query_count = 0
        self._amortized_count = 0

    async def route(
        self,
        query: str,
        oracle: Optional[Callable] = None,
        expected_hint: Optional[str] = None,
        file_provider: Optional[Callable] = None,
        thinking_display=None,
        working_dir: str = ".",
        chat_history: Optional[str] = None,
        repo_map: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        _solve_start = time.time()
        _solve_tokens = 0
        log = []
        _cached_for_verification = None

        if intent is None:
            intent = self._classify_intent(query)
            log.append(f"Classified intent: {intent}")

        if intent != "EDIT":
            context = f"{chat_history or ''}{repo_map or ''}"
            cached_result = self.reasoning_cache.get(query, context)
            if cached_result:
                log.append("Route: REASONING_CACHE (intermediate result)")
                if thinking_display:
                    thinking_display.update_waiting("Using cached reasoning...")
                return self._wrap_result(
                    "FAST",
                    cached_result['answer'],
                    cached_result.get('episodes', []),
                    cached_result.get('skills', []),
                    log,
                    cached_result.get('similarity', 0.95),
                    _solve_start,
                    0
                )

        if intent != "EDIT" and self._is_conversational(query):
            if thinking_display:
                thinking_display.show_route("DIRECT")
                thinking_display.switch_to_solution()

            # Check if it's a simple greeting - return instant response
            q = query.strip().lower()
            greetings_map = {
                'привет': 'Привет! Чем могу помочь?',
                'ку': 'Привет! Чем могу помочь?',
                'хай': 'Привет! Чем могу помочь?',
                'здарова': 'Привет! Чем могу помочь?',
                'здравствуйте': 'Здравствуйте! Чем могу помочь?',
                'hi': 'Hi! How can I help you?',
                'hello': 'Hello! How can I help you?',
                'hey': 'Hey! What can I do for you?',
                'yo': 'Hey! What can I do for you?',
                'спасибо': 'Пожалуйста!',
                'thanks': "You're welcome!",
                'thank you': "You're welcome!",
                'что?': 'Уточни, пожалуйста, что именно тебя интересует?',
                'что': 'Уточни, пожалуйста, что именно тебя интересует?',
                'как дела?': 'Всё отлично, работаю! Чем помочь?',
                'как дела': 'Всё отлично, работаю! Чем помочь?',
                'как ты?': 'Всё в порядке, готов помочь!',
                'как ты': 'Всё в порядке, готов помочь!',
                'ок': 'Хорошо!',
                'ok': 'Okay!',
                'понятно': 'Отлично! Что дальше?',
                'да': 'Хорошо!',
                'нет': 'Понял.',
            }

            if q in greetings_map:
                answer = greetings_map[q]
                log.append("Route: DIRECT (instant greeting)")
                return self._wrap_result("DIRECT", answer, [], [], log, 0.0, _solve_start, 0,
                                        query=query, chat_history=chat_history, repo_map=repo_map, intent=intent)

            # For other conversational queries, use LLM
            direct_prompt = query
            if chat_history:
                direct_prompt = chat_history + query

            import asyncio
            candidates, d_tokens = await asyncio.to_thread(
                llm.generate_samples,
                direct_prompt, n=1, temperature=0.3, mode="DIRECT",
                thinking_display=thinking_display
            )
            _solve_tokens += d_tokens
            answer = "Привет! Чем могу помочь?"
            if candidates and len(candidates) > 0 and isinstance(candidates[0], dict):
                answer = candidates[0].get('solution', answer)
            log.append("Route: DIRECT (conversational)")
            return self._wrap_result("DIRECT", answer, [], [], log, 0.0, _solve_start, _solve_tokens,
                                    query=query, chat_history=chat_history, repo_map=repo_map, intent=intent)

        if thinking_display:
            thinking_display.update_waiting("Routing...")

        adaptive_tau_fast = get_adaptive_tau_fast(query, self.config)
        logging.info(f"[ROUTER] Adaptive tau_fast: {adaptive_tau_fast:.2f} (base: {self.tau_fast:.2f})")

        query_emb = llm.get_embedding(query)

        if thinking_display:
            thinking_display.update_waiting("Matching skills...")

        # Unified skill path: check both compiled_skills and semantic_rules
        skills = self.memory.retrieve_skills(query_emb, k=3)

        for skill in skills:
            similarity = skill.get('similarity', 0)
            confidence = skill.get('confidence', 0)

            # Lower threshold: 0.75 instead of 0.90
            if similarity < 0.75 and confidence < self.config.routing.tau_reflex:
                continue

            pattern = skill.get('pattern', 'unknown')
            log.append(f"Route: COMPILED_SKILL (pattern: {pattern}, sim: {similarity:.2f}, conf: {confidence:.2f})")

            try:
                safe_globals = safe_load_module(skill['code'])

                if 'trigger' in safe_globals and 'execute' in safe_globals:
                    trigger_fn = safe_globals['trigger']
                    execute_fn = safe_globals['execute']

                    if trigger_fn(query):
                        answer = str(execute_fn(query))

                        # Check if skill returned a real answer or just a placeholder
                        is_placeholder = (
                            not answer or
                            answer.startswith("Error") or
                            "Skill for pattern:" in answer or
                            "Based on" in answer and "similar tasks" in answer
                        )

                        if answer and not is_placeholder:
                            # Record success for feedback loop
                            if hasattr(self, 'evolution'):
                                self.evolution.record_skill_result(skill, success=True)

                            # Record metrics
                            self.route_metrics.record_route("COMPILED_SKILL", pattern=pattern)
                            logging.info(f"[METRICS] Route: COMPILED_SKILL, pattern: {pattern}, confidence: {confidence:.2f}")

                            skill_id = skill.get('skill_id', 0)
                            if 0 <= skill_id < len(self.memory.compiled_skills):
                                self.memory.increment_skill_usage(skill_id)
                                self.memory._mark_dirty()

                            self.metrics.record(
                                query=query, route="COMPILED_SKILL",
                                elapsed=time.time() - _solve_start,
                                tokens_used=0,
                                similarity=similarity,
                                answer=answer,
                                score=confidence
                            )

                            if thinking_display:
                                thinking_display.update_waiting(f"✓ Using skill: {pattern}")

                            return self._wrap_result("COMPILED_SKILL", answer, [skill], [], log, similarity, _solve_start, 0)
                        else:
                            # Record failure
                            if hasattr(self, 'evolution'):
                                self.evolution.record_skill_result(skill, success=False)
                            log.append(f"Skill returned error: {answer}")
                    else:
                        log.append(f"Skill trigger returned False for this query")
                else:
                    log.append(f"Skill missing trigger() or execute() functions")

            except Exception as e:
                # Record failure
                if hasattr(self, 'evolution'):
                    self.evolution.record_skill_result(skill, success=False)
                log.append(f"Skill execution failed: {e}")

        if intent != "EDIT" and self.memory.episodic_index.ntotal > 0:
            if thinking_display:
                thinking_display.update_waiting("Searching episodic memory...")

            fast_emb = query_emb
            fast_vec = np.array([fast_emb], dtype=np.float32)

            # Verify dimension matches index
            if fast_vec.shape[1] != self.memory.episodic_index.d:
                logging.warning(f"[ROUTER] Query embedding dim {fast_vec.shape[1]} != index dim {self.memory.episodic_index.d}, skipping FAST route")
            else:
                faiss.normalize_L2(fast_vec)
                sims, indices = self.memory.episodic_index.search(fast_vec, 1)

                if sims[0][0] >= adaptive_tau_fast:
                    if thinking_display:
                        thinking_display.show_route("FAST")
                        thinking_display.start_waiting("Validating cached solution...")

                    idx = int(indices[0][0])
                    if 0 <= idx < len(self.memory.episodes):
                        ep = self.memory.episodes[idx]
                        logging.info(f"[ROUTER] FAST candidate found: score={ep.get('score', 0)}")
                        if ep.get('score', 0) >= 0.80:
                            fast_answer = self._post_process_answer(ep.get('solution', ''), "FAST", log)

                            # Pre-check: if the solution is empty or consists entirely of
                            # XML tool calls that _wrap_result will strip, fall through
                            import re as _re
                            _stripped = fast_answer or ''
                            _stripped = _re.sub(r'<(?:create_file|read_file|list_files|edit_file|write_file)>.*?</(?:create_file|read_file|list_files|edit_file|write_file)>', '', _stripped, flags=_re.DOTALL)
                            _stripped = _re.sub(r'<tool_call[^>]*>.*?</tool_call>', '', _stripped, flags=_re.DOTALL)
                            _stripped = _re.sub(r'<(?:reasoning|delta_reasoning|abstract_signature)\s*>.*?</(?:reasoning|delta_reasoning|abstract_signature)\s*>', '', _stripped, flags=_re.DOTALL)
                            _stripped = _re.sub(r'<(?:final_answer|solution)\s*>|</(?:final_answer|solution)\s*>', '', _stripped)
                            _stripped = _stripped.strip()

                            if not _stripped:
                                logging.info("[ROUTER] FAST cache solution is empty after XML cleanup, falling through to HYBRID/SLOW")
                                log.append("FAST cache empty after cleanup - trying HYBRID/SLOW")
                            elif oracle:
                                if thinking_display:
                                    thinking_display.update_waiting("Validating with oracle...")

                                ok, info = oracle(query, fast_answer)
                                if not ok:
                                    logging.warning(f"[ROUTER] FAST cache INVALID: {info} - falling back to HYBRID/SLOW")

                                    self.memory.update_episode_validation(idx, success=False)
                                    self.memory._mark_dirty()
                                    log.append(f"FAST validation failed: {info} - trying HYBRID")

                                    if thinking_display:
                                        thinking_display.update_waiting("FAST failed, trying HYBRID...")

                                else:
                                    logging.info(f"[ROUTER] FAST cache validated by oracle")

                                    self.memory.update_episode_validation(idx, success=True)
                                    self.memory._mark_dirty()

                                    self.metrics.record(
                                        query=query, route="FAST",
                                        elapsed=time.time() - _solve_start,
                                        tokens_used=0,
                                        similarity=float(sims[0][0]),
                                        answer=fast_answer,
                                        score=ep.get('score', 0.8),
                                    )
                                    return self._wrap_result("FAST", fast_answer, [ep], [], log, float(sims[0][0]), _solve_start, 0,
                                                            query=query, chat_history=chat_history, repo_map=repo_map, intent=intent)
                            else:
                                logging.info(f"[ROUTER] Taking FAST route (no oracle validation)")
                                self.memory.update_episode_validation(idx, success=True)
                                self.memory._mark_dirty()

                                if intent == "EDIT":
                                    from ...tools.parsing.executor import execute_tools_from_response, parse_tool_calls
                                    from ...cli.display.file_writing import get_file_writing_display

                                    tool_calls = parse_tool_calls(fast_answer)

                                    if tool_calls:
                                        logging.info(f"[ROUTER] FAST path executing {len(tool_calls)} cached tools")

                                        if thinking_display:
                                            thinking_display.update_waiting(f"Executing {len(tool_calls)} cached actions...")

                                        file_display = get_file_writing_display()

                                        def stream_callback(event_type, filepath, chunk):
                                            if event_type == 'start':
                                                file_display.start_file(filepath)
                                            elif event_type == 'chunk':
                                                file_display.add_chunk(filepath, chunk)
                                            elif event_type == 'complete':
                                                file_display.complete_file(filepath)

                                        try:
                                            # Emit ToolStart events
                                            if self.bus:
                                                for tool_call in tool_calls:
                                                    self.bus.emit(ToolStart(
                                                        name=tool_call.get('name', 'unknown'),
                                                        args=tool_call.get('args', {}),
                                                        display_verb=None
                                                    ))

                                            execution_result = execute_tools_from_response(
                                                fast_answer,
                                                working_dir=working_dir,
                                                stream_callback=stream_callback
                                            )

                                            # Emit ToolEnd events
                                            if self.bus:
                                                for tool_call in tool_calls:
                                                    self.bus.emit(ToolEnd(
                                                        name=tool_call.get('name', 'unknown'),
                                                        args=tool_call.get('args', {}),
                                                        ok=execution_result.get('success', False),
                                                        summary=None,
                                                        body=None,
                                                        error=execution_result.get('error'),
                                                        meta={},
                                                        display_verb=None,
                                                        body_lang=None
                                                    ))

                                            if execution_result.get('success'):
                                                logging.info(f"[ROUTER] FAST path tools executed successfully")
                                            else:
                                                logging.warning(f"[ROUTER] FAST path tool execution failed: {execution_result.get('error')}")
                                        except Exception as e:
                                            logging.error(f"[ROUTER] FAST path tool execution error: {e}")

                                # Record metrics
                                self.route_metrics.record_route("FAST")
                                logging.info(f"[METRICS] Route: FAST, similarity: {sims[0][0]:.2f}")

                                self.metrics.record(
                                    query=query, route="FAST",
                                    elapsed=time.time() - _solve_start,
                                    tokens_used=0,
                                    similarity=float(sims[0][0]),
                                    answer=fast_answer,
                                    score=ep.get('score', 0.8),
                                )
                                return self._wrap_result("FAST", fast_answer, [ep], [], log, float(sims[0][0]), _solve_start, 0,
                                                        query=query, chat_history=chat_history, repo_map=repo_map, intent=intent)
                    else:
                        logging.info(f"[ROUTER] FAST idx out of bounds: {idx} >= {len(self.memory.episodes)}")

        # REFLEX path removed - now unified with COMPILED_SKILL above

        query_emb_np = np.array([query_emb], dtype=np.float32)
        retrieved_eps = self.memory.retrieve_episodes(query_emb_np, k=3)
        max_sim = max((float(r.get('similarity', 0.0)) for r in retrieved_eps), default=0.0) if retrieved_eps else 0.0

        if thinking_display and retrieved_eps:

            pass

        alpha_t = max_sim

        prompt_prefix = ""
        if chat_history:
            prompt_prefix += chat_history
        if repo_map:
            prompt_prefix += f"--- REPOSITORY MAP ---\n{repo_map}\n----------------------\n\n"

        full_query_context = prompt_prefix + query

        # Check if query requires real action execution (not just cached answer)
        action_signals = [
            'изучай', 'найди', 'покажи', 'прочитай', 'создай', 'напиши', 'измени', 'удали', 'добавь', 'исправь',
            'study', 'explore', 'find', 'show', 'read', 'create', 'write', 'edit', 'delete', 'add', 'fix'
        ]
        requires_action = any(sig in query.lower() for sig in action_signals)

        if max_sim >= self.tau_hybrid and retrieved_eps and not requires_action:
            if thinking_display:
                thinking_display.show_route("HYBRID")
                thinking_display.start_waiting("Adapting previous solution...")

            log.append(f"Route: HYBRID PATH (sim: {max_sim:.3f})")
            prompt = self._build_hybrid_prompt(full_query_context, retrieved_eps[0])
            logging.info(f"[HYBRID] Full prompt:\n{prompt}\n---END PROMPT---")

            import asyncio
            candidates, h_tokens = await asyncio.to_thread(
                llm.generate_samples,
                prompt, n=1, mode="ADAPTIVE", thinking_display=thinking_display
            )
            logging.info(f"[HYBRID] LLM returned {len(candidates)} candidates, {h_tokens} tokens")
            if candidates:
                logging.info(f"[HYBRID] LLM solution: {candidates[0].get('solution', '')[:300]}")
            _solve_tokens += h_tokens

            if thinking_display:
                thinking_display.update_waiting("Evaluating quality...")

            candidates = self.critic.evaluate(query, candidates, oracle=oracle)
            logging.info(f"[HYBRID] After critic: {len(candidates)} candidates")
            if candidates and isinstance(candidates[0], dict) and 'solution' in candidates[0]:
                best = candidates[0]

                # Save tool results that were collected during generation
                tool_results_from_generation = ""
                if thinking_display and hasattr(thinking_display, '_tool_results'):
                    tool_results_from_generation = ''.join(thinking_display._tool_results)
                    thinking_display._tool_results = []  # Clear after saving

                from ...tools.parsing.executor import ToolExecutor
                executor = ToolExecutor(working_dir=".")

                # Emit ToolStart event for HYBRID execution
                if self.bus:
                    self.bus.emit(ToolStart(
                        name="hybrid_execution",
                        args={"solution": best['solution'][:100]},
                        display_verb="Execute"
                    ))

                cleaned_solution, modified_files, tool_results = executor.parse_and_execute(best['solution'])

                logging.info(f"[HYBRID] parse_and_execute returned {len(tool_results)} tool results")
                logging.info(f"[HYBRID] tool_results_from_generation: {len(tool_results_from_generation)} chars")
                if tool_results:
                    for i, result in enumerate(tool_results):
                        logging.info(f"[HYBRID] tool_result[{i}]: {result[:200]}")

                # Emit ToolEnd event
                if self.bus:
                    self.bus.emit(ToolEnd(
                        name="hybrid_execution",
                        args={},
                        ok=True,
                        summary=f"Modified {len(modified_files)} files" if modified_files else "Executed",
                        body=None,
                        error=None,
                        meta={"modified_files": modified_files},
                        display_verb="Execute",
                        body_lang=None
                    ))

                # Build final solution from cleaned text + tool results
                final_parts = []
                if cleaned_solution.strip():
                    final_parts.append(cleaned_solution.strip())

                analysis_text = None
                # If we have tool results, ask model to analyze them
                if tool_results:
                    logging.info(f"[HYBRID] Asking model to analyze {len(tool_results)} tool results")

                    analysis_prompt = f"""The following tool calls were executed:

{chr(10).join(tool_results)}

Provide a brief analysis/summary of what you found. Be concise and focus on key insights."""

                    if thinking_display:
                        thinking_display.update_waiting("Analyzing results...")

                    import asyncio
                    analysis_candidates, analysis_tokens = await asyncio.to_thread(
                        llm.generate_samples,
                        analysis_prompt, n=1, temperature=0.3, mode="DIRECT", thinking_display=thinking_display
                    )
                    _solve_tokens += analysis_tokens

                    if analysis_candidates and analysis_candidates[0].get('solution'):
                        analysis_text = analysis_candidates[0]['solution'].strip()
                        logging.info(f"[HYBRID] Model analysis: {analysis_text[:200]}")
                        final_parts.append(analysis_text)
                    else:
                        # Fallback: just show tool results
                        final_parts.extend(tool_results)
                else:
                    # No tool results, just use cleaned solution
                    pass

                if tool_results_from_generation.strip():
                    final_parts.append(tool_results_from_generation.strip())

                best['solution'] = "\n\n".join(final_parts) if final_parts else "Executed successfully."

                logging.info(f"[HYBRID] Final solution length: {len(best['solution'])} chars")
                logging.info(f"[HYBRID] Final solution preview: {best['solution'][:300]}")

                # Stream analysis to user if thinking_display is active
                if thinking_display and analysis_text:
                    logging.info(f"[HYBRID] Streaming analysis: {len(analysis_text)} chars")

                    # Ensure we're in solution mode
                    if hasattr(thinking_display, 'mode') and thinking_display.mode != 'solution':
                        if hasattr(thinking_display, 'switch_to_solution'):
                            thinking_display.switch_to_solution()
                            logging.info(f"[HYBRID] Switched to solution mode")

                    thinking_display.stream_token(f"\n\n{analysis_text}")

                    # Flush to ensure output is visible
                    if hasattr(thinking_display, '_stop_live_and_spinner'):
                        thinking_display._stop_live_and_spinner()
                else:
                    logging.info(f"[HYBRID] NOT streaming analysis")

                best['solution'] = self._post_process_answer(best['solution'], "HYBRID", log)
                best['final_score'] = (max_sim * 1.0) + ((1 - max_sim) * best['final_score'])

                # Record metrics
                self.route_metrics.record_route("HYBRID")
                logging.info(f"[METRICS] Route: HYBRID, similarity: {max_sim:.2f}")

                self.metrics.record(
                    query=query, route="HYBRID",
                    elapsed=time.time() - _solve_start,
                    tokens_used=_solve_tokens,
                    similarity=max_sim,
                    answer=best['solution'],
                    score=best['final_score'],
                )
                return self._wrap_result("HYBRID", best['solution'], retrieved_eps, candidates, log, max_sim, _solve_start, _solve_tokens, alpha_t,
                                        query=query, chat_history=chat_history, repo_map=repo_map, intent=intent)
            else:

                logging.warning(f"[ROUTER] HYBRID produced no valid candidates - falling back to SLOW")
                log.append("HYBRID failed: no valid candidates - trying SLOW")
                if thinking_display:
                    thinking_display.show_route("SLOW")
                    thinking_display.start_waiting("Falling back to synthesis...")

        log.append(f"Route: SLOW PATH (sim: {max_sim:.3f})")

        if thinking_display:
            thinking_display.show_route("SLOW")
            thinking_display.start_waiting("Synthesizing solution...")

        adaptive_params = self._assess_task_complexity(full_query_context, thinking_display=None) if max_sim < 0.3 else {}

        if 'temperature' in adaptive_params:
            adaptive_params['temperature'] = 0.1
            logging.info(f"[ROUTER] Overriding temperature to 0.1 for code precision")

        if "File:" in query and "```python" in query and ("EXACT_PATH" in query or "CRITICAL FORMAT REQUIREMENT" in query):
            prompt = query
            logging.info(f"[ROUTER] Using original SWE-bench prompt (format instructions detected)")
        else:
            prompt = self._build_slow_prompt(full_query_context, query_emb, retrieved_eps)

        if oracle:
            max_attempts = adaptive_params.get('max_attempts', self.config.synthesis.max_attempts)

            solve_ctx = SolveContext(query=query, oracle=oracle)

            should_plan = self._should_generate_plan(full_query_context)

            plan_result = None
            if should_plan:
                if thinking_display:
                    thinking_display.update_waiting("Generating execution plan...")

                plan_result = self.planner.generate_plan(
                    task=full_query_context,
                    repo_map=repo_map,
                    existing_context=None,
                    thinking_display=thinking_display
                )

                if thinking_display and plan_result and plan_result.get('plan_steps'):
                    thinking_display.stop_waiting()
                    from ...cli.display import ui
                    ui.print_plan(plan_result)
                    thinking_display.start_waiting("Executing plan")

            if plan_result and plan_result.get('plan_steps'):
                plan_text = "\n".join(f"{i}. {step}" for i, step in enumerate(plan_result['plan_steps'], 1))
                vs_query = f"{prompt}\n\nEXECUTION PLAN:\n{plan_text}\n\nFollow this plan step by step."
            else:
                vs_query = prompt

            if "File:" not in vs_query:
                vs_query = prompt

            import asyncio
            vs_result = await asyncio.to_thread(
                verified_synthesis,
                query=vs_query,
                propose_fn=lambda p, priors: self._propose_for_vs(p, priors, llm, adaptive_params, thinking_display),
                oracle=oracle,
                max_attempts=max_attempts,
                expected_hint=expected_hint,
                context=solve_ctx,
                file_provider=file_provider,
            )

            if vs_result.converged:
                final_score = 0.95
            elif solve_ctx.best_iou >= 0.95:

                final_score = solve_ctx.best_iou
            elif solve_ctx.best_iou >= 0.80:

                final_score = solve_ctx.best_iou
            else:
                final_score = 0.30

            candidates = [{
                'solution': vs_result.final_answer,
                'reasoning_trace': f"VS converged in {vs_result.total_attempts} attempts (adaptive: {adaptive_params}, best_iou: {solve_ctx.best_iou:.2f})",
                'final_score': final_score,
                'solve_context': solve_ctx,
            }]
        else:

            # Detect DATA MODE queries (read-only, no interpretation)
            query_lower = query.lower()
            data_keywords = ['изучай', 'покажи', 'дай', 'прочитай', 'read', 'show', 'display', 'cat', 'view']
            is_data_mode = any(query_lower.startswith(kw) for kw in data_keywords)
            if is_data_mode:
                # Check if query has analysis keywords
                analysis_keywords = ['анализ', 'проанализ', 'объясни', 'explain', 'analyze', 'why', 'how', 'почему', 'как']
                if any(kw in query_lower for kw in analysis_keywords):
                    is_data_mode = False

            # Don't plan for DATA MODE queries
            should_plan = self._should_generate_plan(query) and not is_data_mode

            plan_result = None
            if should_plan:
                if thinking_display:

                    thinking_display.start_waiting("Planning")

                plan_result = self.planner.generate_plan(
                    task=full_query_context,
                    repo_map=repo_map,
                    existing_context=None,
                    thinking_display=thinking_display
                )

                if thinking_display and plan_result.get('plan_steps'):
                    thinking_display.stream_token(f"\n| Plan ({plan_result['complexity']})\n")
                    for i, step in enumerate(plan_result['plan_steps'], 1):
                        thinking_display.stream_token(f"  {i}. {step}\n")
                    thinking_display.stream_token("\n")

            if plan_result and plan_result.get('plan_steps'):
                plan_text = "\n".join(f"{i}. {step}" for i, step in enumerate(plan_result['plan_steps'], 1))
                prompt_with_plan = f"{prompt}\n\nEXECUTION PLAN:\n{plan_text}\n\nFollow this plan step by step."
            else:
                prompt_with_plan = prompt

            max_auto_iters = adaptive_params.get('max_attempts', 10)
            iter_count = 0
            current_prompt = prompt_with_plan

            best = None
            final_solution_text = ""

            recent_tool_signatures = []  # Track tool call signatures for loop detection

            while iter_count < max_auto_iters:
                iter_count += 1

                # Use DATA mode if detected earlier
                generation_mode = "DATA" if is_data_mode else "ANALYTIC"

                candidates, s_tokens = llm.generate_samples(current_prompt, n=1, temperature=0.2, mode=generation_mode, thinking_display=thinking_display)
                _solve_tokens += s_tokens

                if not candidates or not isinstance(candidates[0], dict) or 'solution' not in candidates[0]:
                    break

                best = candidates[0]

                from ...tools.parsing.executor import execute_tools_from_response, parse_tool_calls
                from ...cli.display.file_writing import get_file_writing_display

                tool_calls = parse_tool_calls(best['solution'])

                if not tool_calls:

                    final_solution_text += "\n\n" + best['solution']
                    break

                def _compute_tool_signature(tool_calls):
                    """Extract signature of actions (files + operations)."""
                    signature = set()
                    for call in tool_calls:
                        tool_name = call.get('tool', '')
                        file_path = call.get('path', '')
                        if file_path:
                            signature.add(f"{tool_name}:{file_path}")
                    return signature

                tool_signature = _compute_tool_signature(tool_calls)
                recent_tool_signatures.append(tool_signature)

                # Check for looping via Jaccard similarity
                if len(recent_tool_signatures) >= 3:
                    last_3 = recent_tool_signatures[-3:]

                    # Compute pairwise similarities
                    similarities = []
                    for i in range(len(last_3)):
                        for j in range(i+1, len(last_3)):
                            if last_3[i] and last_3[j]:
                                intersection = last_3[i] & last_3[j]
                                union = last_3[i] | last_3[j]
                                sim = len(intersection) / len(union) if union else 0.0
                                similarities.append(sim)

                    # If average similarity > 0.8, we're looping
                    if similarities and sum(similarities) / len(similarities) > 0.8:
                        if thinking_display:
                            thinking_display.stream_token("\n[red]| Loop detected! Aborting autopilot.[/]\n")
                        final_solution_text += "\n\n" + best['solution'] + "\n\n[SYSTEM: Autopilot aborted due to repeated similar actions. Please re-evaluate the approach.]"
                        break

                if thinking_display:
                    thinking_display.stream_token(f"\n| Executing {len(tool_calls)} actions\n")

                file_display = get_file_writing_display()

                def stream_callback(event_type, filepath, chunk):
                    if event_type == 'start':
                        if thinking_display:
                            import os
                            filename = os.path.basename(filepath)
                            thinking_display.print_action(f"| Modifying {filename}")
                        file_display.start_writing(filepath)
                    elif event_type == 'chunk':
                        file_display.stream_content(chunk)
                    elif event_type == 'finish':
                        file_display.finish_writing()

                # Emit ToolStart events
                if self.bus:
                    for tool_call in tool_calls:
                        self.bus.emit(ToolStart(
                            name=tool_call.get('tool', 'unknown'),
                            args=tool_call,
                            display_verb=None
                        ))

                tool_results = execute_tools_from_response(best['solution'], stream_callback=stream_callback if thinking_display else None, working_dir=working_dir)

                # Emit ToolEnd events
                if self.bus:
                    for tool_call in tool_calls:
                        self.bus.emit(ToolEnd(
                            name=tool_call.get('tool', 'unknown'),
                            args=tool_call,
                            ok=bool(tool_results),
                            summary=None,
                            body=None,
                            error=None,
                            meta={},
                            display_verb=None,
                            body_lang=None
                        ))

                if tool_results:
                    log.append(f"Executed {len(tool_results)} tool calls")

                    import re
                    solution_clean = best['solution']

                    solution_clean = re.sub(r'<(create_file|edit_file|read_file|list_files)>.*?</\1>', '', solution_clean, flags=re.DOTALL)

                    solution_clean = re.sub(r'(create_file|edit_file|read_file|list_files)\s*\([^)]*\)', '', solution_clean, flags=re.DOTALL)

                    solution_clean = re.sub(r'```[\s\S]*?```', '', solution_clean)

                    solution_clean = re.sub(r'\n{3,}', '\n\n', solution_clean).strip()

                    if len(solution_clean) > 500:
                        lines = solution_clean.split('\n\n')
                        solution_clean = lines[0] if lines else solution_clean[:200]

                    final_solution_text += "\n\n" + solution_clean + "\n" + "\n".join(tool_results)

                    current_prompt += f"\n\nASSISTANT (Step {iter_count}):\n{solution_clean}\n\nSYSTEM (Tool Results):\n" + "\n".join(tool_results) + "\n\nContinue executing the plan. If you are finished, summarize your work and DO NOT call any more tools."

                    if thinking_display and iter_count < max_auto_iters:
                        thinking_display.print_action(f"| Auto-continuing to step {iter_count + 1}")
                else:
                    final_solution_text += "\n\n" + best['solution']
                    break

            if best:
                best['solution'] = self._post_process_answer(final_solution_text.strip(), "SLOW", log)
                best['final_score'] = 0.85

            from ...tools.path_validator import check_solution_paths, suggest_corrections

            import os
            if os.path.exists('.git'):
                path_check = check_solution_paths(best['solution'])
            else:
                path_check = {'hallucination_detected': False}

            if path_check.get('hallucination_detected', False):
                invalid = path_check['invalid_paths']
                logging.warning(f"[ROUTER] Hallucinated paths detected: {invalid}")
                log.append(f"Path Validation: Found {len(invalid)} invalid paths")

                suggestions = suggest_corrections(invalid)
                correction_prompt = f"""
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

                retry_candidates, retry_tokens = llm.generate_samples(correction_prompt, n=1, temperature=0.5, mode="ANALYTIC", thinking_display=thinking_display)
                _solve_tokens += retry_tokens

                if retry_candidates and len(retry_candidates) > 0 and isinstance(retry_candidates[0], dict) and 'solution' in retry_candidates[0]:
                    retry_solution = self._post_process_answer(retry_candidates[0]['solution'], "SLOW-PATH-FIX", log)
                    retry_check = check_solution_paths(retry_solution)

                    if not retry_check['hallucination_detected']:
                        logging.info(f"[ROUTER] Path correction SUCCESS")
                        log.append(f"Path Validation: SUCCESS - all paths valid")
                        best['solution'] = retry_solution
                    else:
                        logging.warning(f"[ROUTER] Path correction FAILED - still has invalid paths")
                        log.append(f"Path Validation: FAILED - keeping original")

            if oracle:
                ok, info = oracle(query, best['solution'])

                if ok is False and isinstance(info, str):

                    logging.warning(f"[ROUTER] Oracle failed: {info[:200]}")
                    log.append(f"Self-Correction: Oracle failed, attempting retry")
                    correction_prompt = f"""
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

                    retry_candidates, retry_tokens = llm.generate_samples(correction_prompt, n=1, temperature=0.7, mode="ANALYTIC", thinking_display=thinking_display)
                    _solve_tokens += retry_tokens

                    if retry_candidates and len(retry_candidates) > 0 and isinstance(retry_candidates[0], dict) and 'solution' in retry_candidates[0]:
                        retry_solution = self._post_process_answer(retry_candidates[0]['solution'], "SLOW-RETRY", log)
                        ok_retry, info_retry = oracle(query, retry_solution)

                        if ok_retry:
                            logging.info(f"[ROUTER] Self-correction SUCCESS")
                            log.append(f"Self-Correction: SUCCESS on retry")
                            best['solution'] = retry_solution
                            best['final_score'] = 0.85
                        else:
                            logging.warning(f"[ROUTER] Self-correction FAILED: {info_retry[:100]}")
                            log.append(f"Self-Correction: FAILED on retry")

            # Record metrics
            self.route_metrics.record_route("SLOW")
            logging.info(f"[METRICS] Route: SLOW, similarity: {max_sim:.2f}")

            self.metrics.record(
                query=query, route="SLOW",
                elapsed=time.time() - _solve_start,
                tokens_used=_solve_tokens,
                similarity=max_sim,
                answer=best['solution'],
                score=best.get('final_score', 0.5),
            )

            # Save successful SLOW execution as episode
            if best and best.get('final_score', 0) >= 0.80:
                try:
                    query_emb = llm.get_embedding(query)
                    episode_data = {
                        "query": query,
                        "solution": best['solution'],
                        "reasoning_trace": f"SLOW route: {len(candidates)} candidates",
                        "score": best.get('final_score', 0.85),
                        "metadata": {
                            "source": "router_slow",
                            "tokens": _solve_tokens,
                            "elapsed": time.time() - _solve_start,
                        }
                    }
                    self.memory.add_episode(episode_data, np.array([query_emb], dtype=np.float32))
                    logging.info(f"[ROUTER] Saved SLOW result as episode")
                except Exception as e:
                    logging.warning(f"[ROUTER] Failed to save episode: {e}")

            return self._wrap_result("SLOW", best['solution'], retrieved_eps, candidates, log, max_sim, _solve_start, _solve_tokens, alpha_t,
                                    query=query, chat_history=chat_history, repo_map=repo_map, intent=intent)

        return self._wrap_result("ERROR", "No solution found", [], [], log, 0.0, _solve_start, _solve_tokens,
                                query=query, chat_history=chat_history, repo_map=repo_map, intent=intent)

    def _post_process_answer(self, raw: str, route: str, log: list) -> str:
        if not raw: return raw
        py_block = extract_python_block(raw)
        if not py_block: return raw
        try:
            executed = safe_execute_freeform(py_block)
            if executed and not executed.startswith("Error:"):
                log.append(f"[{route}] Executed inline code block.")
                return executed
        except Exception as e:
            logging.warning(f"[Router] Failed to execute inline code: {e}")
        return raw

    def _propose_for_vs(self, prompt, priors, llm_mod, adaptive_params=None, thinking_display=None):
        adaptive_params = adaptive_params or {}

        temp = adaptive_params.get('temperature', 0.1)

        cands, _ = llm_mod.generate_samples(prompt, n=1, temperature=temp, mode="SYNTHESIS", thinking_display=thinking_display)
        if cands and len(cands) > 0 and isinstance(cands[0], dict):
            return cands[0].get('solution', '')
        return ""

    def _assess_task_complexity(self, query: str, thinking_display=None) -> Dict[str, Any]:
        """Assess task complexity and return adaptive parameters.

        Uses LLM to analyze query and determine:
        - max_attempts: How many VS iterations needed
        - breadth: How many candidates to generate
        - temperature: Sampling temperature

        Only called on first encounter (max_sim < 0.3).
        """
        assessment_prompt = f"""
Task: {query}

Respond with JSON only:
{{{{
  "complexity": "simple|medium|hard|extreme",
  "reasoning": "brief explanation",
  "max_attempts": 3-12,
  "breadth": 3-8,
  "temperature": 0.5-0.9
}}}}

Simple: straightforward logic, 3 attempts, breadth 3, temp 0.5
Medium: multi-step reasoning, 5 attempts, breadth 5, temp 0.7
Hard: complex algorithms, 8 attempts, breadth 6, temp 0.8
Extreme: novel patterns, 12 attempts, breadth 8, temp 0.9"""

        try:

            cands, _ = llm.generate_samples(assessment_prompt, n=1, temperature=0.3, mode="DIRECT", thinking_display=None)
            if not cands:
                return {}

            response = cands[0].get('solution', '{}')

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

        return {}

    def _build_hybrid_prompt(self, query, ep):
        p = f"Task: {query}\n\n"

        import re
        past_text = ep.get('solution', '')

        past_text = re.sub(r'<read_file>.*?</read_file>', '[read files]', past_text, flags=re.DOTALL)
        past_text = re.sub(r'<edit_file>.*?</edit_file>', '[edit files]', past_text, flags=re.DOTALL)
        past_text = re.sub(r'<write_file>.*?</write_file>', '[write files]', past_text, flags=re.DOTALL)
        past_text = re.sub(r'<bash_command>.*?</bash_command>', '[run commands]', past_text, flags=re.DOTALL)
        past_text = past_text[:400]
        if len(ep.get('solution', '')) > 400:
            past_text += "..."

        p += f"Similar Past Task: {ep.get('query', '')[:200]}\n"
        p += f"Past Approach: {past_text}\n\n"
        p += "CRITICAL: You MUST execute the NEW task, not just describe it.\n"
        p += "1. Briefly explain what you'll do\n"
        p += "2. Use tool calls to perform ACTUAL actions\n"
        p += "3. NEVER say 'task is identical' - always execute\n\n"
        p += "Example response format:\n"
        p += "I'll add the greeting to ui.py.\n\n"
        p += "<tool_call>\n"
        p += '{"name": "edit_file", "args": {"path": "cli/display/ui.py", "old": "    console.print(f\\"Done\\")", "new": "    console.print(f\\"Done\\")\\n\\n# Привет от NARE"}}\n'
        p += "</tool_call>\n\n"
        p += "Done! Added greeting to ui.py.\n"
        return p

    def _build_slow_prompt(self, query, query_emb, retrieved_eps):
        p = f"Task: {query}\n\n"

        learned_rules = self.memory.retrieve_semantics(query_emb, k=2)

        if learned_rules:
            p += "--- LEARNED RULES ---\n"
            for rule in learned_rules:
                if rule.get('confidence', 0) >= 0.70:
                    pattern = rule.get('pattern', 'Unknown')

                    p += f"- {pattern}\n"
            p += "---\n\n"

        if retrieved_eps:

            p += "--- RELEVANT MEMORIES ---\n"
            for ep in retrieved_eps[:2]:

                solution = ep.get('solution', '')[:500]
                if len(ep.get('solution', '')) > 500:
                    solution += "..."
                p += f"Past: {ep.get('query', '')[:100]}\nSolution: {solution}\n---\n"

        p += "\nIMPORTANT: Use the learned rules above as MANDATORY guidance, not suggestions.\n"
        p += "For Django tasks, you MUST check __init__.py files in packages.\n"
        p += "Solve the task with deep reasoning.\n\n"
        p += "Example response format:\n"
        p += "I'll add the greeting to ui.py.\n\n"
        p += "<tool_call>\n"
        p += '{"name": "edit_file", "args": {"path": "cli/display/ui.py", "old": "    console.print(f\\"Done\\")", "new": "    console.print(f\\"Done\\")\\n\\n# Привет от NARE"}}\n'
        p += "</tool_call>\n\n"
        p += "Done! Added greeting to ui.py.\n"
        return p

    def _classify_intent(self, query: str) -> str:
        """Classify user intent using LLM for ambiguous cases.

        Uses LLM classification to handle creative phrasings and ambiguous queries
        that don't match simple keyword patterns.
        """
        query_lower = query.lower().strip()

        # Fast path: obvious greetings
        greetings = ['привет', 'ку', 'hello', 'hi', 'hey', 'здравствуй', 'добрый день']
        if query_lower in greetings or len(query_lower) < 5:
            return "CONVERSATIONAL"

        # Fast path: questions (check before actions)
        question_words = [
            'что', 'как', 'почему', 'зачем', 'когда', 'где', 'кто', 'какой',
            'расскажи', 'объясни', 'покажи', 'изучи', 'проанализируй',
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'tell', 'explain', 'show', 'analyze', 'describe'
        ]

        if any(query_lower.startswith(word) for word in question_words):
            return "CONVERSATIONAL"

        # Fast path: obvious action keywords
        obvious_actions = [
            'создай', 'сделай', 'напиши', 'реализуй', 'добавь', 'измени',
            'исправь', 'удали', 'create', 'make', 'write', 'implement',
            'add', 'change', 'fix', 'delete', 'remove', 'build', 'refactor'
        ]

        if any(word in query_lower for word in obvious_actions):
            return "EDIT"

        # Ambiguous case: ask LLM
        prompt = f"""Classify the user's intent:

Query: "{query}"

Is this:
A) A request to CREATE/MODIFY/DELETE code or files (action)
B) A question or conversation (no action needed)

Answer with just "ACTION" or "CONVERSATION"."""

        try:
            from ...reasoning import llm

            response = llm._post_anthropic("messages", {
                "model": "claude-3-haiku-20240307",  # Fast model
                "max_tokens": 10,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}]
            })

            result = response.get("content", [{}])[0].get("text", "").strip().upper()

            if "ACTION" in result:
                return "EDIT"
            else:
                return "CONVERSATIONAL"

        except Exception as e:
            logging.warning(f"[ROUTER] Intent classification failed: {e}, defaulting to EDIT")
            return "EDIT"  # Default to action

    def _should_generate_plan(self, query: str) -> bool:
        """Determine if query needs planning based on complexity.

        Only plan for truly complex multi-step tasks, not simple queries.
        """
        query_lower = query.lower().strip()

        # Never plan for greetings or very short queries
        greetings = ['привет', 'ку', 'hello', 'hi', 'hey', 'здравствуй', 'добрый день']
        if query_lower in greetings or len(query_lower) < 10:
            return False

        # Only plan if user explicitly asks for a plan
        plan_keywords = ['план', 'plan', 'спланируй', 'распиши шаги', 'как будешь']
        for keyword in plan_keywords:
            if keyword in query_lower:
                return True

        # Don't auto-plan for simple read/explore tasks
        simple_tasks = ['изучай', 'покажи', 'дай', 'прочитай', 'read', 'show', 'display']
        for task in simple_tasks:
            if query_lower.startswith(task):
                return False

        return False
        if len(query_lower) > 50:
            return True

        return False

    def _is_conversational(self, query: str) -> bool:
        """Detect trivial conversational queries that need no tools/planning.

        Matches greetings, meta-questions about capabilities, and other
        short non-actionable queries in both Russian and English.
        Returns True if the query should be handled via DIRECT path.

        CRITICAL: Action keywords (создай, напиши, измени, доработай, etc.)
        should NEVER be conversational - they need SLOW path with real tools.
        """
        q = query.strip().lower()

        greetings = {
            'ку', 'привет', 'хай', 'здарова', 'здравствуйте', 'добрый день',
            'доброе утро', 'добрый вечер', 'салам', 'йо', 'хелло',
            'hi', 'hello', 'hey', 'yo', 'sup', 'howdy', 'greetings',
            'спасибо', 'thanks', 'thank you', 'ок', 'ok', 'понятно', 'да', 'нет',
        }
        if q in greetings:
            return True

        action_signals = [
            'создай', 'напиши', 'измени', 'удали', 'добавь', 'исправь', 'доработай',
            'найди', 'покажи', 'прочитай', 'открой', 'запусти', 'изучай', 'улучши',
            'create', 'write', 'edit', 'delete', 'add', 'fix', 'find', 'improve',
            'show', 'read', 'open', 'run', 'build', 'implement', 'refactor',
        ]
        if any(sig in q for sig in action_signals):
            return False

        meta_patterns = [
            'что ты умеешь', 'кто ты', 'что ты такое', 'как тебя зовут',
            'что ты можешь', 'как ты работаешь', 'what can you do',
            'who are you', 'what are you', 'how do you work',
            'помощь', 'help', 'как пользоваться',
        ]
        for pattern in meta_patterns:
            if pattern in q:
                return True

        return False

    def _get_hardcoded_response(self, query: str) -> str:
        """Get instant hardcoded response for common greetings.

        Avoids LLM call for trivial 1-token responses (ISSUE #1 from audit).
        Returns None if no hardcoded response available.
        """
        q = query.strip().lower()

        # Russian greetings
        russian_greetings = {
            'ку': 'Привет! Чем могу помочь?',
            'привет': 'Привет! Чем могу помочь?',
            'хай': 'Привет! Чем могу помочь?',
            'здарова': 'Привет! Чем могу помочь?',
            'здравствуйте': 'Здравствуйте! Чем могу помочь?',
            'добрый день': 'Добрый день! Чем могу помочь?',
            'доброе утро': 'Доброе утро! Чем могу помочь?',
            'добрый вечер': 'Добрый вечер! Чем могу помочь?',
        }

        # English greetings
        english_greetings = {
            'hi': 'Hi! How can I help?',
            'hello': 'Hello! How can I help?',
            'hey': 'Hey! How can I help?',
            'yo': 'Hey! What can I do for you?',
            'sup': 'Hey! What can I do for you?',
            'howdy': 'Howdy! How can I help?',
        }

        if q in russian_greetings:
            return russian_greetings[q]
        if q in english_greetings:
            return english_greetings[q]

        return None

    def _wrap_result(self, route, answer, memories, candidates, log, alpha, start_time, tokens, alpha_t=0.0, query="", chat_history="", repo_map="", intent=""):
        # Track amortization
        self._query_count += 1
        if route in ("FAST", "REFLEX", "COMPILED_SKILL", "DIRECT"):
            self._amortized_count += 1
        alpha_t_empirical = self._amortized_count / self._query_count if self._query_count > 0 else 0.0
        c_llm = self.config.amortization.c_llm
        c_mem = self.config.amortization.c_mem
        blended_cost = (1.0 - alpha_t_empirical) * c_llm + alpha_t_empirical * c_mem
        logging.info(f"[AMORTIZATION] α_t={alpha_t_empirical:.3f}, C_t={blended_cost:.1f}, route={route}, queries={self._query_count}")
        # Clean up any XML tags that model might have generated by mistake
        import re
        if answer:
            # Filter out lines containing <tool_call> tags
            lines = answer.split('\n')
            filtered_lines = []
            skip_until_close = False

            for i, line in enumerate(lines):
                if '<tool_call' in line:
                    skip_until_close = True
                    # Remove previous line ONLY if it's a short prefix (< 20 chars and ends with >)
                    if filtered_lines:
                        prev = filtered_lines[-1].strip()
                        if len(prev) < 20 and prev.endswith('>'):
                            filtered_lines.pop()
                if skip_until_close:
                    if '</tool_call' in line:
                        skip_until_close = False
                    continue
                filtered_lines.append(line)

            answer = '\n'.join(filtered_lines)

            # Remove XML-style tool calls: <read_file>path</read_file>, <list_files>path</list_files>, etc.
            answer = re.sub(r'<read_file>.*?</read_file>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'<list_files>.*?</list_files>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'<create_file>.*?</create_file>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'<edit_file>.*?</edit_file>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'<write_file>.*?</write_file>', '', answer, flags=re.DOTALL)

            answer = re.sub(r'<final_answer\s*>|</final_answer\s*>', '', answer)
            answer = re.sub(r'<reasoning\s*>.*?</reasoning\s*>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'<delta_reasoning\s*>.*?</delta_reasoning\s*>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'<abstract_signature\s*>.*?</abstract_signature\s*>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'<solution\s*>|</solution\s*>', '', answer)
            answer = re.sub(
                r'\{\s*"name"\s*:\s*"(?:create_file|edit_file|read_file|list_files|list_dir|write_file)"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\}',
                '', answer
            )
            answer = re.sub(r'\n{3,}', '\n\n', answer).strip()

        result = {
            "route_decision": route,
            "final_answer": answer,
            "retrieved_memories": memories,
            "generated_candidates": candidates,
            "memory_update_log": log,
            "alpha": float(alpha),
            "alpha_t": float(alpha_t_empirical),
            "alpha_t_theoretical": float(alpha_t),
            "blended_cost": float(blended_cost),
            "amortization_ratio": float(alpha_t_empirical),
            "novelty": 0.0,
            "elapsed": time.time() - start_time,
            "tokens": tokens
        }

        if query and intent != "EDIT" and route in ("FAST", "HYBRID", "SLOW") and alpha >= 0.75:
            context = f"{chat_history or ''}{repo_map or ''}"
            cache_data = {
                'answer': answer,
                'episodes': memories,
                'skills': [],
                'similarity': alpha,
                'route': route,
            }
            try:
                self.reasoning_cache.set(query, cache_data, context)
            except Exception:
                pass

        return result
