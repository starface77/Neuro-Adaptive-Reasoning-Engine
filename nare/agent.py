import os
import time
import numpy as np
import faiss
import logging
from typing import List, Dict, Any, Callable, Optional, Tuple, TYPE_CHECKING
from . import llm

if TYPE_CHECKING:
    from .oracle import Oracle  # noqa: F401
from .config import DEFAULT_CONFIG, NareConfig
from .memory import MemorySystem
from .metrics import MetricsTracker
from .graph_memory import EpisodeGraph
from .rl_retriever import RLRetriever
from .neural_memory import NeuralMemory
from .meta_abduction import MetaAbductionEngine
from .query_fingerprint import (
    average_pairwise_jaccard,
    query_fingerprint,
)
from . import sandbox
from .sandbox import (
    SecurityError,
    safe_call_execute_in_namespace,
    safe_call_trigger,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')

class HybridCritic:
    """Convergent selection critic.

    Combines four independent signals for robust candidate ranking:
      1. Ground Truth Validator — compilation / sandbox execution for
         code solutions.  Mandatory when applicable.
      2. LLM-critic — pairwise Elo tournament as a fallback for "soft"
         cases without a deterministic verifier.
      3. Self-Consistency — majority-vote over extracted short answers
         when multiple reasoning chains are available.
      4. Rule-based heuristic — fast syntactic checks (code markers,
         error strings).
    """

    def __init__(self, config: NareConfig = DEFAULT_CONFIG):
        cfg = config.critic
        self.weights = (cfg.w_llm, cfg.w_rule, cfg.w_neural)
        self.elo_k = cfg.elo_k_factor
        self.elo_init = cfg.elo_initial_rating

    # ---- Signal 1: Ground Truth Validator ----

    def _ground_truth_validate(self, solution: str) -> Optional[float]:
        """Attempt objective verification for code solutions.

        Returns a score in [0, 1] if validation was possible, or None
        if the solution is not code (falls back to LLM critic).
        """
        code = NAREProductionAgent._extract_code(solution)
        if code is None:
            return None

        try:
            from .sandbox import validate_code
            validate_code(code)
            return 0.8
        except Exception:
            return 0.2

    # ---- Signal 2: Rule-based heuristic ----

    def _rule_based_check(self, solution: str) -> float:
        score = 0.5
        if "```" in solution or "def " in solution:
            score += 0.3
        if "Error" in solution or "Exception" in solution:
            score -= 0.4
        return max(0.0, min(1.0, score))

    # ---- Signal 3: Self-Consistency voting ----

    @staticmethod
    def _extract_short_answer(solution: str) -> str:
        """Extract a normalised short answer for majority voting.

        Heuristic: take the last non-empty line, strip markdown fences
        and whitespace.  For numerical answers this collapses to the
        number; for textual answers it grabs the final conclusion.
        """
        lines = [ln.strip() for ln in solution.strip().splitlines() if ln.strip()]
        if not lines:
            return ""
        answer = lines[-1]
        for prefix in ("Answer:", "Result:", "Output:", "answer:", "result:"):
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        answer = answer.strip("`").strip("*").strip()
        return answer.lower()

    def _self_consistency_scores(self, candidates: List[Dict]) -> Dict[int, float]:
        """Compute majority-vote score for each candidate.

        Returns a dict mapping candidate index → SC score ∈ [0, 1].
        """
        answers = [self._extract_short_answer(c['solution']) for c in candidates]
        if not answers:
            return {}
        # Count frequency of each distinct short answer
        from collections import Counter
        counts = Counter(answers)
        total = len(answers)
        return {i: counts[a] / total for i, a in enumerate(answers)}

    # ---- Main evaluation ----

    def evaluate(
        self, query: str, candidates: List[Dict], oracle: Optional[Any] = None
    ) -> List[Dict]:
        if not candidates:
            return []
        if len(candidates) == 1:
            candidates[0]['llm_score'] = 0.5
            candidates[0]['rule_score'] = self._rule_based_check(
                candidates[0]['solution']
            )
            if oracle:
                try:
                    passed, _ = oracle(query, candidates[0]['solution'])
                    gt = 1.0 if passed else 0.0
                except:
                    gt = 0.0
            else:
                gt = self._ground_truth_validate(candidates[0]['solution'])
            candidates[0]['gt_score'] = gt
            candidates[0]['sc_score'] = 1.0  # single candidate is trivially consistent
            candidates[0]['final_score'] = gt if gt is not None else 0.5
            return candidates

        # 1. Ground Truth Validator — mandatory for code
        for c in candidates:
            if oracle:
                try:
                    passed, _ = oracle(query, c['solution'])
                    c['gt_score'] = 1.0 if passed else 0.0
                except:
                    c['gt_score'] = 0.0
            else:
                c['gt_score'] = self._ground_truth_validate(c['solution'])

        # 2. Self-Consistency voting
        sc_scores = self._self_consistency_scores(candidates)
        for i, c in enumerate(candidates):
            c['sc_score'] = sc_scores.get(i, 0.0)

        # 3. Elo tournament (LLM-critic backup for soft cases)
        for c in candidates:
            c['elo'] = self.elo_init

        for i in range(len(candidates) - 1):
            c1, c2 = candidates[i], candidates[i + 1]
            try:
                winner = llm.llm_pairwise_judge(query, c1['solution'], c2['solution'])
            except Exception as e:
                logging.error(f"Pairwise judge failed: {e}")
                winner = 1

            ea = 1.0 / (1.0 + 10.0 ** ((c2['elo'] - c1['elo']) / 400.0))
            if winner == 1:
                c1['elo'] += self.elo_k * (1.0 - ea)
                c2['elo'] -= self.elo_k * (1.0 - ea)
            else:
                c1['elo'] -= self.elo_k * ea
                c2['elo'] += self.elo_k * ea

        elos = [c['elo'] for c in candidates]
        e_min, e_max = min(elos), max(elos)

        # 4. Combine all signals
        for c in candidates:
            if e_max == e_min:
                c['llm_score'] = 0.5
            else:
                c['llm_score'] = (c['elo'] - e_min) / (e_max - e_min)

            c['rule_score'] = self._rule_based_check(c['solution'])

            # Combined ranking: GT > SC > LLM > rule
            if c['gt_score'] is not None:
                c['final_score'] = max(
                    0.0,
                    0.40 * c['gt_score']
                    + 0.25 * c['sc_score']
                    + 0.20 * c['llm_score']
                    + 0.15 * c['rule_score'],
                )
            else:
                c['final_score'] = max(
                    0.0,
                    0.30 * c['sc_score']
                    + self.weights[0] * 0.70 * c['llm_score']
                    + self.weights[1] * 0.70 * c['rule_score'],
                )

        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates


class NAREProductionAgent:
    def __init__(
        self,
        config: NareConfig = DEFAULT_CONFIG,
        oracle: "Oracle | None" = None,
    ):
        self.config = config
        # Optional external oracle used during sleep/REM phases when
        # validating compiled skills. When None, _validate_skill falls
        # back to a heuristic string/numeric overlap. See nare.oracle.
        self.oracle = oracle
        self.memory = MemorySystem(config=config)
        self.critic = HybridCritic(config=config)
        self.metrics = MetricsTracker(persist_dir=self.memory.persist_dir)
        self.graph = EpisodeGraph(persist_dir=self.memory.persist_dir)
        self.rl_retriever = RLRetriever(
            persist_dir=self.memory.persist_dir, config=config
        )
        self.neural_memory = NeuralMemory(persist_dir=self.memory.persist_dir)
        self.meta_engine = MetaAbductionEngine(persist_dir=self.memory.persist_dir)

        self.tau_fast = config.routing.tau_fast
        self.tau_hybrid = config.routing.tau_hybrid
        self.tau_min = config.routing.tau_min
        self.tau_max = config.routing.tau_max

    def _calibrate_tau(self, reward: float, fast_path_used: bool):
        """Adjust tau_fast based on FAST-path outcomes. Clamped to config range."""
        lr = self.config.routing.calibration_lr
        if fast_path_used and reward < 0.5:
            self.tau_fast = min(self.tau_max, self.tau_fast + lr)
            logging.info(f"[Calibration] tau_fast \u2191 {self.tau_fast:.3f}")
        elif not fast_path_used and reward > 0.8:
            self.tau_fast = max(self.tau_min, self.tau_fast - (lr / 2))

    def calibrate_tau_fast_from_replay(
        self,
        target_precision: float = 0.95,
        *,
        apply: bool = True,
        embedding_key: str = "embedding",
        tau_min: Optional[float] = None,
        tau_max: Optional[float] = None,
        tau_step: float = 0.005,
    ) -> Dict[str, Any]:
        """Replay-driven calibration of ``tau_fast`` over cached episodes.

        Sweeps a range of candidate ``tau`` values, computes precision /
        coverage of the FAST path on actual cached pairs (judged by the
        strict ``cached_episode_oracle``), and optionally applies the
        smallest ``tau`` whose precision meets ``target_precision``.

        This is **offline only** — no LLM calls — and is intended to be
        run after warm-up of the episode store. The online per-call
        ``_calibrate_tau`` continues to handle drift after this is set.

        Returns the full calibration report (see
        :func:`nare.tau_calibration.calibrate_tau_fast`). When ``apply``
        is True and a precision-passing tau is found, ``self.tau_fast``
        is updated (clamped into the existing ``[tau_min, tau_max]``
        config band) and the change is logged.
        """
        from .tau_calibration import calibrate_tau_fast

        with self.memory._lock:
            episodes_snapshot = list(self.memory.episodes)

        sweep_min = tau_min if tau_min is not None else max(0.50, self.tau_min - 0.05)
        sweep_max = tau_max if tau_max is not None else min(0.999, self.tau_max + 0.005)

        report = calibrate_tau_fast(
            episodes_snapshot,
            tau_min=sweep_min,
            tau_max=sweep_max,
            tau_step=tau_step,
            target_precision=target_precision,
            embedding_key=embedding_key,
        )
        if apply and report.get("recommended_tau") is not None:
            new_tau = float(report["recommended_tau"])
            clamped = max(self.tau_min, min(self.tau_max, new_tau))
            old = self.tau_fast
            self.tau_fast = clamped
            logging.info(
                f"[Replay Calibration] tau_fast {old:.3f} -> {clamped:.3f} "
                f"(unclamped recommendation {new_tau:.3f}, target precision "
                f"{target_precision:.2f}, n_episodes={report['n_episodes']})"
            )
        return report

    def _normalize_embeddings(self, vecs: np.ndarray) -> np.ndarray:
        """Utility: L2-normalize a batch of vectors for cosine similarity."""
        v = vecs.copy().astype(np.float32)
        faiss.normalize_L2(v)
        return v

    def _check_sleep_trigger(self) -> bool:
        sleep_cfg = self.config.sleep

        if len(self.memory.episodes) >= sleep_cfg.max_episodes_before_sleep:
            return True

        if len(self.memory.episodes) >= sleep_cfg.cluster_density_threshold:
            raw_vecs = np.array(
                [
                    ep.get('signature_embedding', ep['embedding'])
                    for ep in self.memory.episodes
                ],
                dtype=np.float32,
            )
            vecs = self._normalize_embeddings(raw_vecs)
            sim_matrix = np.dot(vecs, vecs.T)
            np.fill_diagonal(sim_matrix, 0.0)
            # Require a *dense* cluster: a single point with at least N
            # neighbours above the similarity threshold. Previously this
            # fired on ANY single pair (>=1), which over-triggered.
            density_above = np.sum(
                sim_matrix > sleep_cfg.cluster_similarity_threshold, axis=1
            )
            min_neighbours = max(1, sleep_cfg.cluster_density_threshold - 1)
            dense_idx = np.where(density_above >= min_neighbours)[0]
            for core_idx in dense_idx:
                cluster_indices = np.where(
                    sim_matrix[core_idx] > sleep_cfg.cluster_similarity_threshold
                )[0].tolist()
                cluster_indices.append(int(core_idx))
                # Optional secondary gate: query structural fingerprint
                # Jaccard among cluster members must also exceed a
                # threshold. This catches cases where embeddings group
                # surface-similar but structurally different queries.
                if sleep_cfg.use_query_fingerprint_gate:
                    fps = [
                        self.memory.episodes[i].get('query_fingerprint')
                        for i in cluster_indices
                    ]
                    # Backward compatibility: episodes saved before this
                    # PR have no ``query_fingerprint`` field. If fewer
                    # than 2 cluster members carry fingerprint data the
                    # gate cannot be evaluated; default to *passing*
                    # (preserve legacy embedding-only behaviour) and
                    # log distinctly so the operator can tell the gate
                    # is inert from missing data, not from a structural
                    # mismatch.
                    non_none_fps = [fp for fp in fps if fp]
                    if len(non_none_fps) < 2:
                        logging.info(
                            f"[Sleep Trigger] Fingerprint gate skipped: "
                            f"only {len(non_none_fps)}/{len(fps)} cluster "
                            f"members carry query_fingerprint data "
                            f"(likely a pre-Tier-B episode store)."
                        )
                    else:
                        avg_jacc = average_pairwise_jaccard(fps)
                        if avg_jacc < sleep_cfg.query_fingerprint_threshold:
                            logging.info(
                                f"[Sleep Trigger] Embedding cluster rejected: "
                                f"query fingerprint Jaccard {avg_jacc:.2f} < "
                                f"{sleep_cfg.query_fingerprint_threshold:.2f}"
                            )
                            continue
                logging.info(
                    "[Sleep Trigger] Dense cluster detected "
                    f"(>= {min_neighbours} neighbours @ sim > "
                    f"{sleep_cfg.cluster_similarity_threshold})."
                )
                return True
        return False

    def _symbolic_lift(self, episodes: List[Dict]) -> List[Dict]:
        """Symbolic Lifting : replace concrete constants with abstract variables.

        Before LLM induction, scan solutions for hard-coded numbers, strings,
        etc. and wrap them with placeholder tokens so that the LLM produces
        generalised code rather than constants bound to specific examples.
        """
        import re as _re
        lifted = []
        for ep in episodes:
            ep_copy = ep.copy()
            solution = ep_copy.get('solution', '')
            # Replace concrete numbers (not inside variable names) with <NUM>
            solution_lifted = _re.sub(r'(?<![a-zA-Z_])\b\d{2,}\b(?![a-zA-Z_])', '<NUM>', solution)
            # Replace quoted string literals with <STR>
            solution_lifted = _re.sub(r'"[^"]{5,}"', '"<STR>"', solution_lifted)
            solution_lifted = _re.sub(r"'[^']{5,}'", "'<STR>'", solution_lifted)
            if solution_lifted != solution:
                ep_copy['solution_original'] = solution
                ep_copy['solution'] = solution_lifted
                ep_copy['symbolic_lifted'] = True
            lifted.append(ep_copy)
        return lifted

    def _sleep_phase(self):
        logging.info("=== [SLEEP PHASE] Crystallizing Memories ===")
        sleep_cfg = self.config.sleep
        sim_threshold = sleep_cfg.cluster_similarity_threshold

        # Use structural signature embeddings.
        raw_vecs = np.array(
            [
                ep.get('signature_embedding', ep['embedding'])
                for ep in self.memory.episodes
            ],
            dtype=np.float32,
        )
        vecs = self._normalize_embeddings(raw_vecs)
        sim_matrix = np.dot(vecs, vecs.T)
        np.fill_diagonal(sim_matrix, 0.0)
        dense_clusters = np.sum(sim_matrix > sim_threshold, axis=1)

        if np.max(dense_clusters) < 1:
            # No clusters above threshold; still try to refine weak rules.
            self._prune_weak_rules()
            self.memory.prune_fading_memories()
            return

        core_idx = int(np.argmax(dense_clusters))
        cluster_indices = np.where(sim_matrix[core_idx] > sim_threshold)[0]
        
        cluster_episodes = [self.memory.episodes[i] for i in cluster_indices]

        # Symbolic Lifting : abstract concrete constants before
        # sending cluster to LLM for induction.
        cluster_episodes = self._symbolic_lift(cluster_episodes)
        
        # Use the normalized centroid embedding for the rule
        centroid = np.mean(vecs[cluster_indices], axis=0, keepdims=True).astype(np.float32)
        faiss.normalize_L2(centroid)
        
        # Consolidation: Check if a similar rule already exists
        existing_semantics = self.memory.retrieve_semantics(centroid, k=1)
        existing_threshold = self.config.sleep.existing_rule_match_threshold

        try:
            if existing_semantics and existing_semantics[0]['similarity'] > existing_threshold:
                existing_rule = existing_semantics[0]
                rule_idx = int(existing_rule.get('memory_id', 0))
                existing_conf = existing_rule.get('confidence', 0.5)
                
                # Quarantined rules are not refined at all — they have
                # repeatedly failed REM dream-tests and resetting their
                # confidence on every sleep cycle is what produced the
                # 0.45 ↔ 0.23 oscillation seen in the benchmark log.
                if existing_rule.get('quarantined'):
                    logging.info(
                        f"[Refinement] Skipping quarantined rule "
                        f"'{existing_rule.get('pattern')}' "
                        f"(rem_penalty_count={existing_rule.get('rem_penalty_count', 0)}, "
                        f"peak_confidence={existing_rule.get('peak_confidence', existing_conf):.2f})"
                    )
                elif existing_conf < 0.95:
                    # GRADED REFINEMENT: Weak/Hybrid skill needs more training
                    logging.info(f"=== [GRADED REFINEMENT] Upgrading rule '{existing_rule.get('pattern')}' (conf: {existing_conf:.2f}) ===")
                    updated_rule = llm.merge_heuristic_rules(existing_rule, cluster_episodes)

                    # Re-validate the merged rule with stress tests
                    from nare.llm import generate_stress_tests, _validate_skill
                    stress_tests = generate_stress_tests(cluster_episodes)
                    all_tests = cluster_episodes + stress_tests
                    scores, error_msg = _validate_skill(
                        updated_rule.get('python_code', ''),
                        all_tests,
                        oracle=self.oracle,
                        config=self.config,
                    )

                    # --- Confidence-bounce hysteresis ---
                    # If this rule has been REM-penalised at least once,
                    # clamp the bump so the next REM cycle does not
                    # restart the 0.45 ↔ 0.23 oscillation seen in the
                    # benchmark log.  See SkillConfig.skill_refinement_-
                    # max_bump_after_penalty for tuning.
                    raw_score = scores['overall']
                    rem_penalties = existing_rule.get('rem_penalty_count', 0)
                    if rem_penalties > 0:
                        max_bump = self.config.skill.skill_refinement_max_bump_after_penalty
                        clamped = min(raw_score, existing_conf + max_bump)
                        if clamped < raw_score:
                            logging.info(
                                f"[Refinement Hysteresis] '{updated_rule['pattern']}' "
                                f"raw={raw_score:.2f} -> clamped={clamped:.2f} "
                                f"(rem_penalty_count={rem_penalties})"
                            )
                        applied_conf = clamped
                    else:
                        applied_conf = raw_score

                    updated_rule['confidence'] = applied_conf
                    updated_rule['trigger_accuracy'] = scores['trigger_accuracy']
                    updated_rule['execute_accuracy'] = scores['execute_accuracy']
                    updated_rule['sleep_cycles'] = existing_rule.get('sleep_cycles', 0) + 1
                    # Track best-ever confidence for quarantine decisions.
                    updated_rule['peak_confidence'] = max(
                        existing_rule.get('peak_confidence', existing_conf),
                        applied_conf,
                    )
                    # Carry over penalty/quarantine state.
                    updated_rule['rem_penalty_count'] = rem_penalties
                    updated_rule['quarantined'] = existing_rule.get('quarantined', False)
                    self.memory.update_semantic_rule(rule_idx, updated_rule, new_embedding=centroid)
                    logging.info(f"[Refinement] '{updated_rule['pattern']}' conf: {existing_conf:.2f} -> {applied_conf:.2f} (trigger={scores['trigger_accuracy']:.2f}, exec={scores['execute_accuracy']:.2f})")
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

                # ─── Phase 6: held-out validation ─────────────────
                # Reserve ``holdout_n`` episodes from the cluster, induct
                # on the rest, then validate the resulting rule's
                # execute() on the held-out set. This is the FIRST
                # honest transfer test in the sleep pipeline — pre-
                # Phase-6 the same episodes were used for both
                # training and validation (self-referential), which
                # is exactly why skills overfitted to seen examples
                # and failed on paraphrases.
                sleep_cfg = self.config.sleep
                use_holdout = (
                    sleep_cfg.use_holdout_validation
                    and len(cluster_episodes) >= sleep_cfg.holdout_min_cluster_size
                )
                if use_holdout:
                    h = max(1, int(sleep_cfg.holdout_n))
                    train_eps = cluster_episodes[:-h]
                    holdout_eps = cluster_episodes[-h:]
                    logging.info(
                        f"[Holdout] {len(train_eps)} train + "
                        f"{len(holdout_eps)} held-out from "
                        f"{len(cluster_episodes)} cluster episodes"
                    )
                else:
                    train_eps = cluster_episodes
                    holdout_eps = []

                new_rule = llm.extract_heuristic_rule(
                    train_eps,
                    oracle=self.oracle,
                    config=self.config,
                )
                if new_rule is None:
                    logging.warning("[Sleep] Skill failed validation completely (robustness < 0.40). Keeping episodes.")
                    return

                if holdout_eps:
                    from nare.llm import _validate_skill
                    h_scores, h_report = _validate_skill(
                        new_rule.get('python_code', ''),
                        holdout_eps,
                        oracle=self.oracle,
                        config=self.config,
                    )
                    h_acc = h_scores.get('execute_accuracy', 0.0)
                    h_min = sleep_cfg.holdout_min_accuracy
                    logging.info(
                        f"[Holdout] '{new_rule.get('pattern')}' "
                        f"execute_acc={h_acc:.2f} (min={h_min:.2f})"
                    )
                    if h_acc < h_min:
                        logging.warning(
                            f"[Holdout] REJECTING '{new_rule.get('pattern')}' "
                            f"— failed transfer test "
                            f"(execute_acc={h_acc:.2f} < {h_min:.2f}). "
                            f"Keeping {len(cluster_episodes)} episodes for "
                            f"a future sleep cycle. Report: {h_report[:200]}"
                        )
                        return
                    # Honest transfer signal goes into the rule's
                    # confidence floor — mark it so downstream code
                    # can distinguish "passed held-out" from "trained
                    # only".
                    new_rule['holdout_accuracy'] = float(h_acc)
                    new_rule['holdout_n'] = len(holdout_eps)
                new_rule['sleep_cycles'] = 0
                # Track source episodes for Penalty Backpropagation. We
                # store CONTENT KEYS (sha1(query+solution)) rather than
                # positional indices because the next few lines delete
                # the cluster episodes from memory.episodes — positional
                # indices become stale immediately, which silently
                # corrupted the immune system's τ updates (Devin Review
                # finding on PR #5).
                from .memory import episode_content_key  # local import to avoid cycle
                # Build the full set of cluster member positions:
                # ``cluster_indices`` excludes ``core_idx`` because the
                # similarity matrix diagonal was zeroed at line 400, so
                # without this we'd record only the soon-to-be-deleted
                # episodes and the *surviving* representative would
                # never be in source_keys — making penalty backprop a
                # no-op on every new rule (Devin Review on 20c3fa3).
                all_member_indices = list({int(i) for i in cluster_indices} | {core_idx})
                source_keys = []
                for i in all_member_indices:
                    ep = self.memory.episodes[i]
                    key = ep.get('episode_key') or episode_content_key(
                        ep.get('query', ''), ep.get('solution', ''),
                    )
                    source_keys.append(key)
                new_rule['source_episode_keys'] = source_keys
                # Keep legacy field for one release so older serialised
                # rules still load — but mark it as best-effort. Newer
                # code paths must read source_episode_keys.
                new_rule['source_episode_ids'] = list(all_member_indices)
                self.memory.add_semantic_rule(new_rule, centroid)
                logging.info(f"[Crystallization] New Rule: {new_rule['pattern']} (confidence: {new_rule['confidence']:.2f})")
            
            # SUCCESS: Now compress the episodic memory (under lock
            # to avoid race with concurrent solve() reads).
            to_delete = set(int(i) for i in cluster_indices) - {core_idx}
            with self.memory._lock:
                self.memory.episodes = [ep for i, ep in enumerate(self.memory.episodes) if i not in to_delete]
                self.memory._rebuild_episodic_index()
            
            # Run pruning on weak rules and fading episodes
            self._prune_weak_rules()
            self.memory.prune_fading_memories()
            
            self.memory.save()
            logging.info(f"[Sleep] Compressed {len(to_delete)} episodes into 1 rule + 1 representative.")
            
        except Exception as e:
            logging.error(f"[Sleep Phase] Failed: {e}")

    def _rem_cached_replay(self, rule: dict):
        """Run a rule against every cached episode it triggers on.

        The "cached-replay" REM signal is the paper's intended ground
        truth: a skill must reproduce the verified solution stored on
        a prior episode, judged by the strict ``cached_episode_oracle``
        (or a per-episode ``oracle_spec`` / global oracle when present).

        Returns ``(scores_dict, failing_episodes, n_triggered)``. When
        ``n_triggered == 0`` the rule never fires on the cache and we
        return ``(None, [], 0)`` so the caller can fall back to the
        legacy LLM-stress path.

        ``scores_dict`` mirrors the keys produced by
        :func:`nare.llm._validate_skill` so REM can feed it directly
        into the existing repair loop.
        """
        from nare.sandbox import safe_load_module, SecurityError
        from nare.oracle import (
            build_oracle_from_spec,
            cached_episode_oracle,
        )

        python_code = rule.get("python_code", "") or ""
        if not python_code:
            return None, [], 0

        try:
            ns = safe_load_module(python_code)
        except SecurityError as e:
            logging.warning(
                f"[REM Replay] Rule '{rule.get('pattern')}' rejected by "
                f"AST validator: {e}"
            )
            return None, [], 0
        except Exception as e:  # noqa: BLE001
            logging.warning(
                f"[REM Replay] Rule '{rule.get('pattern')}' failed to load: {e}"
            )
            return None, [], 0

        trigger_fn = ns.get("trigger")
        execute_fn = ns.get("execute")
        if trigger_fn is None or execute_fn is None:
            return None, [], 0

        # Snapshot under lock so we never iterate while sleep mutates the
        # episode list.
        with self.memory._lock:
            episodes_snapshot = list(self.memory.episodes)

        n_triggered = 0
        n_correct = 0
        failing: list = []

        for ep in episodes_snapshot:
            query = ep.get("query", "")
            try:
                if not trigger_fn(query):
                    continue
            except Exception:  # noqa: BLE001 - trigger crash counts as triggered+failed
                n_triggered += 1
                failing.append({
                    "query": query,
                    "solution": ep.get("solution", ""),
                    "type": "POSITIVE",
                    "info": "trigger() crashed on cached episode",
                })
                continue

            n_triggered += 1

            try:
                output = str(execute_fn(query))
            except Exception as e:  # noqa: BLE001
                failing.append({
                    "query": query,
                    "solution": ep.get("solution", ""),
                    "type": "POSITIVE",
                    "info": f"execute() crashed: {type(e).__name__}: {e}",
                })
                continue

            spec = ep.get("oracle_spec")
            if spec:
                try:
                    ep_oracle = build_oracle_from_spec(spec)
                except Exception:  # noqa: BLE001
                    ep_oracle = cached_episode_oracle(ep.get("solution", ""))
            elif self.oracle is not None:
                ep_oracle = self.oracle
            else:
                ep_oracle = cached_episode_oracle(ep.get("solution", ""))

            ok, info = ep_oracle(query, output)
            if ok:
                n_correct += 1
            else:
                failing.append({
                    "query": query,
                    "solution": ep.get("solution", ""),
                    "type": "POSITIVE",
                    "info": info,
                    "got": output[:120],
                })

        if n_triggered == 0:
            return None, [], 0

        score = n_correct / n_triggered if n_triggered > 0 else 0.0
        scores_dict = {
            "trigger_accuracy": 1.0,
            "execute_accuracy": score,
            "negative_trap_accuracy": 1.0,
            "positive_no_crash_rate": 1.0,
            "overall": score,
            "replay_n_triggered": n_triggered,
            "replay_n_correct": n_correct,
        }
        return scores_dict, failing, n_triggered

    def _rem_sleep_phase(self):
        """REM Sleep: cached-replay validation + iterative repair.

        Order of validation (Tier A revision):
          1. **Cached-replay first.** Run the skill against every cached
             episode it triggers on; score with the strict
             ``cached_episode_oracle``. This is the paper's intended
             ground-truth signal: real solved tasks, deterministic
             match. No LLM judging itself.
          2. **LLM-generated stress tests fall back** only when the
             cache cannot produce ``rem_min_replay_episodes`` triggered
             episodes (e.g. for very new skills with little history).
          3. Repair loop runs against the failing real episodes when
             they exist, otherwise against LLM stress tests.
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

        skill_cfg = self.config.skill
        replay_threshold = skill_cfg.rem_replay_pass_threshold
        min_replay = skill_cfg.rem_min_replay_episodes

        for rule in rules_to_test:
            pattern = rule.get("pattern", "Unknown")
            python_code = rule.get("python_code", "")
            if not python_code:
                continue

            logging.info(f"[REM] Dreaming about rule '{pattern}' ...")

            try:
                from nare.llm import generate_stress_tests, _validate_skill, repair_skill

                # 1. Cached-replay first (paper's ground-truth signal).
                replay_scores, replay_failing, n_triggered = (
                    self._rem_cached_replay(rule)
                )
                used_replay = (
                    replay_scores is not None
                    and n_triggered >= min_replay
                )

                if used_replay:
                    scores = replay_scores
                    error_msg = (
                        f"[CACHED REPLAY] {scores['replay_n_correct']}/"
                        f"{scores['replay_n_triggered']} cached episodes "
                        f"reproduced under strict oracle "
                        f"(score={scores['execute_accuracy']:.2f})."
                    )
                    dream_tests = replay_failing  # repair loop sees real failures
                    logging.info(
                        f"[REM] '{pattern}' cached-replay: "
                        f"{scores['replay_n_correct']}/{scores['replay_n_triggered']} "
                        f"(score={scores['execute_accuracy']:.2f})"
                    )
                else:
                    # 2. Fallback: LLM-generated stress tests when the
                    # cache cannot give us enough triggered episodes.
                    related_episodes = [
                        ep for ep in self.memory.episodes
                        if any(
                            w in ep.get("query", "").lower()
                            for w in pattern.lower().split()[:3]
                        )
                    ][:3]
                    if not related_episodes:
                        related_episodes = self.memory.episodes[:3]
                    if not related_episodes:
                        continue

                    dream_tests = generate_stress_tests(related_episodes)
                    if not dream_tests:
                        continue

                    scores, error_msg = _validate_skill(
                        python_code, dream_tests,
                        oracle=self.oracle, config=self.config,
                    )
                old_conf = rule.get("confidence", 0.5)
                # In cached-replay mode use the configured threshold
                # (paper's "≥80% real reproductions"); fall back to the
                # historical 0.80 in LLM-stress mode.
                pass_threshold = (
                    replay_threshold if used_replay else 0.80
                )

                if scores["overall"] >= pass_threshold:
                    boost = min(0.05, (scores["overall"] - pass_threshold) * 0.25)
                    rule["confidence"] = min(0.99, old_conf + boost)
                    if used_replay:
                        rule["rem_replay_passes"] = (
                            rule.get("rem_replay_passes", 0) + 1
                        )
                    logging.info(
                        f"[REM] '{pattern}' passed "
                        f"({'cached-replay' if used_replay else 'LLM-stress'}, "
                        f"overall={scores['overall']:.2f}). "
                        f"conf: {old_conf:.2f} -> {rule['confidence']:.2f}"
                    )
                else:
                    # Iterative code correction: attempt to repair the skill
                    logging.warning(
                        f"[REM] '{pattern}' FAILED "
                        f"({'cached-replay' if used_replay else 'LLM-stress'}, "
                        f"overall={scores['overall']:.2f}). "
                        f"Attempting iterative repair..."
                    )
                    # Build a validator closure so repair_skill can iterate
                    # internally and pick the best candidate, instead of
                    # returning the first generation regardless of quality.
                    if used_replay:
                        def _validate_candidate(code: str) -> float:
                            patched = dict(rule)
                            patched["python_code"] = code
                            cand_scores, _failing, _n = (
                                self._rem_cached_replay(patched)
                            )
                            if cand_scores is None:
                                return 0.0
                            return float(cand_scores.get("overall", 0.0))
                    else:
                        def _validate_candidate(code: str) -> float:
                            cand_scores, _err = _validate_skill(
                                code, dream_tests,
                                oracle=self.oracle, config=self.config,
                            )
                            return float(cand_scores.get("overall", 0.0))

                    repaired_code = repair_skill(
                        python_code, pattern, dream_tests,
                        error_msg, scores, max_attempts=3,
                        validator=_validate_candidate,
                        baseline_score=scores.get("overall", 0.0),
                    )
                    if repaired_code and repaired_code != python_code:
                        # Re-validate the repaired code against the SAME
                        # signal we used to flag it (cached-replay or
                        # LLM-stress) so the comparison is apples-to-apples.
                        if used_replay:
                            patched_rule = dict(rule)
                            patched_rule["python_code"] = repaired_code
                            new_scores, _failing_after, n_after = (
                                self._rem_cached_replay(patched_rule)
                            )
                            if new_scores is None:
                                # Repaired code no longer triggers on the
                                # cache — treat as regression.
                                new_scores = {
                                    "overall": 0.0,
                                    "execute_accuracy": 0.0,
                                }
                            new_err = (
                                f"[CACHED REPLAY] post-repair "
                                f"score={new_scores.get('execute_accuracy', 0.0):.2f}"
                            )
                        else:
                            new_scores, new_err = _validate_skill(
                                repaired_code, dream_tests,
                                oracle=self.oracle, config=self.config,
                            )
                        if new_scores["overall"] > scores["overall"]:
                            rule["python_code"] = repaired_code
                            rule["confidence"] = max(old_conf * 0.9, new_scores["overall"])
                            rule["rem_repairs"] = rule.get("rem_repairs", 0) + 1
                            rule["peak_confidence"] = max(
                                rule.get("peak_confidence", old_conf),
                                rule["confidence"],
                            )
                            logging.info(
                                f"[REM] '{pattern}' REPAIRED successfully! "
                                f"score: {scores['overall']:.2f} -> {new_scores['overall']:.2f}, "
                                f"conf: {old_conf:.2f} -> {rule['confidence']:.2f}"
                            )
                        else:
                            # Repair didn't improve; apply penalty
                            self._apply_rem_penalty(rule, scores, pass_threshold, old_conf, pattern)
                    else:
                        # Repair failed or returned same code; apply penalty
                        self._apply_rem_penalty(rule, scores, pass_threshold, old_conf, pattern)

            except Exception as e:
                logging.error(f"[REM] Dream failed for '{pattern}': {e}")

        # Synaptic downscaling: weaken all graph edges slightly
        self.graph.weaken_all(decay=0.02)
        self.graph.save()
        
        self.memory.save()
        logging.info("=== [REM SLEEP] Dreaming complete ===")

    @staticmethod
    def _extract_code(solution: str) -> Optional[str]:
        """Extract Python code from a solution string.

        Handles both raw code and markdown-fenced ```python blocks.
        Returns None if no code detected.
        """
        if "```python" in solution:
            import re
            m = re.search(r'```python\n(.*?)\n```', solution, re.DOTALL)
            if m:
                return m.group(1)
        if "def " in solution and ("return" in solution or "print" in solution):
            return solution
        return None

    def _background_validate_episodes(self):
        """Background Validation: periodically audit random episodes.

        For code episodes, attempt compilation.  For all others, check
        basic sanity.  Update τ_i accordingly and persist changes.
        """
        import random
        count = self.config.immune.background_audit_count
        if not self.memory.episodes:
            return

        sample_size = min(count, len(self.memory.episodes))
        indices = random.sample(range(len(self.memory.episodes)), sample_size)

        for idx in indices:
            ep = self.memory.episodes[idx]
            solution = ep.get('solution', '')
            code = self._extract_code(solution)
            try:
                if code is not None:
                    from .sandbox import validate_code
                    validate_code(code)
                    self.memory.update_episode_tau(idx, +1.0)
                    logging.info(f"[Background Audit] Episode {idx} code valid, τ boosted")
                else:
                    if len(solution.strip()) > 10:
                        self.memory.update_episode_tau(idx, +0.5)
                    else:
                        self.memory.update_episode_tau(idx, -0.5)
                        logging.warning(f"[Background Audit] Episode {idx} has very short answer")
            except Exception:
                self.memory.update_episode_tau(idx, -1.0)
                logging.warning(f"[Background Audit] Episode {idx} failed validation, τ penalised")

        # Prune episodes that fell below immune threshold
        self.memory.prune_untrusted_episodes()
        # Persist tau updates even if nothing was pruned
        self.memory.save()

    def _prune_weak_rules(self):
        """Garbage collect rules that are DEAD based on global_score."""
        sleep_cfg = self.config.sleep
        pruned = []
        kept = []
        for rule in self.memory.semantic_rules:
            g_score = rule.get('global_score', rule.get('confidence', 0.5))
            cycles = rule.get('sleep_cycles', 0)
            if (
                g_score < sleep_cfg.weak_rule_global_score
                and cycles >= sleep_cfg.weak_rule_min_cycles
            ):
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

    def _post_process_answer(self, raw: str, route: str, log: list) -> str:
        """If the answer is a fenced ``\u200b```python ...\u200b``` block, execute it
        in the sandbox and substitute the captured stdout/result.

        This is what stops FAST / HYBRID / SLOW from echoing back the
        LLM's verbatim code block (the user's last benchmark log showed
        Tasks 5, 14, 15, 20, 21, 22 all returning fenced code instead of
        the executed value, regressing accuracy from 91.7% to 83.3%).

        Behaviour is deliberately conservative: if extraction yields no
        block, if the sandbox rejects the code, or if execution returns
        nothing useful, the original answer is returned unchanged so we
        never *lose* a correct plain-text answer to a failed exec attempt.
        """
        if not raw:
            return raw
        py_block = sandbox.extract_python_block(raw)
        if not py_block.strip():
            return raw
        try:
            executed = sandbox.safe_execute_freeform(py_block)
        except sandbox.SecurityError as exc:
            log.append(f"[{route}] Inline code blocked by sandbox: {exc}")
            return raw
        except Exception as exc:  # noqa: BLE001 — sandbox surface is broad
            log.append(f"[{route}] Inline code raised: {exc}")
            return raw
        if not executed or executed.startswith("Error:"):
            log.append(f"[{route}] Inline code produced no usable output")
            return raw
        log.append(
            f"[{route}] Executed inline code block ({len(py_block)} chars) "
            f"-> '{executed[:60]}'"
        )
        return executed

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
            "query_fingerprint": query_fingerprint(query),
        }
        
        # Structure embedding for sleep clustering
        sig = episode_data["abstract_signature"]
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

    def wait_for_sleep(self, timeout=300):
        """Wait for background sleep phase to complete."""
        start = time.time()
        while hasattr(self, '_is_sleeping') and self._is_sleeping:
            if time.time() - start > timeout:
                logging.warning("[Agent] Timeout waiting for sleep phase.")
                break
            time.sleep(1)
            
    def _is_cold_start(self) -> bool:
        """Check if the system is in cold-start mode ."""
        return len(self.memory.episodes) < self.config.bootstrap.cold_start_threshold

    def _bootstrap_load_seeds(self):
        """Load pre-warmed seed examples on first run ."""
        import json as _json
        path = self.config.bootstrap.seed_examples_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                seeds = _json.load(f)
            loaded = 0
            for seed in seeds:
                if not seed.get("query") or not seed.get("solution"):
                    continue
                emb = llm.get_embedding(seed["query"])
                ep_data = {
                    "query": seed["query"],
                    "solution": seed["solution"],
                    "reasoning_trace": seed.get("reasoning_trace", "seed"),
                    "context": "bootstrap_seed",
                    "abstract_signature": seed.get("query"),
                    "score": 0.7,
                    "embedding": emb,
                    "signature_embedding": emb,
                }
                added = self.memory.add_episode(ep_data, np.array([emb], dtype=np.float32))
                if added:
                    loaded += 1
            if loaded:
                logging.info(f"[Bootstrap] Loaded {loaded} seed examples from {path}")
        except Exception as e:
            logging.warning(f"[Bootstrap] Failed to load seeds: {e}")

    def solve(
        self,
        query: str,
        oracle: Optional[Callable[[str, str], Tuple[bool, dict]]] = None,
        expected_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Solve a query.

        Parameters
        ----------
        query
            Natural-language input.
        oracle
            Optional ``(query, candidate_answer) -> (passed, info)``
            judge. When provided, the SLOW path runs Verified
            Synthesis (execution-feedback loop) instead of plain
            best-of-N — this is the first lever NARE has over vanilla
            CoT, since the LLM gets to see its own code's output and
            self-correct. With ``oracle=None`` SLOW behaves identically
            to the pre-Phase-6 implementation, which is provably no
            worse than vanilla.
        expected_hint
            Optional ground-truth hint passed verbatim into VS feedback
            prompts. Only set this when the caller actually has the
            answer (LEARN phase, hold-out validation, REM replay,
            A/B benchmark with ``oracle_spec``).
        """
        # Stash on instance so the SLOW path branch can read them
        # without threading through every helper. They live for the
        # duration of this single solve() call.
        self._current_oracle = oracle
        self._current_expected_hint = expected_hint
        _solve_start = time.time()
        _solve_tokens = 0
        log = []
        log.append(f"Query: {query}")

        # Bootstrap: load seeds on first solve if memory is empty
        if len(self.memory.episodes) == 0:
            self._bootstrap_load_seeds()

        # =====================================================
        # LAYER 0: FAST CACHE — O(log N) via HNSW embedding search
        # =====================================================
        # Theory: deterministic return when semantic similarity exceeds
        # tau_fast.  Uses the HNSW index for O(log N) lookup instead of
        # linear string comparison.
        if self.memory.episodic_index.ntotal > 0:
            fast_emb = llm.get_embedding(query)
            fast_vec = np.array([fast_emb], dtype=np.float32)
            faiss.normalize_L2(fast_vec)
            sims, indices = self.memory.episodic_index.search(fast_vec, 1)
            if sims[0][0] >= self.tau_fast:
                idx = int(indices[0][0])
                if 0 <= idx < len(self.memory.episodes):
                    ep = self.memory.episodes[idx]
                    if ep.get('score', 0) > 0.5:
                        log.append(f"Route: FAST (HNSW hit, sim={sims[0][0]:.3f}, score={ep.get('score', 0):.2f})")
                        ep['last_used'] = time.time()
                        ep['strength'] = ep.get('strength', 1.0) + 0.1
                        # Immune system: boost τ for successfully reused episodes
                        self.memory.update_episode_tau(idx, +1.0)
                        self.memory.save()
                        # If the cached episode's stored solution is itself a
                        # fenced ```python``` block (likely because the
                        # original SLOW path saved the raw LLM response),
                        # execute it now so FAST never echoes back code.
                        fast_answer = self._post_process_answer(
                            ep.get('solution', '') or '', "FAST", log
                        )
                        # ─── Phase 6: oracle-gated cache hit ─────────
                        # If the caller supplied an oracle and the
                        # cached answer fails it, fall through to SLOW
                        # so Verified Synthesis can self-correct. This
                        # prevents stale cache from poisoning A/B
                        # benchmarks. Without an oracle, behaviour is
                        # unchanged from pre-Phase-6.
                        cur_oracle = getattr(self, '_current_oracle', None)
                        oracle_rejected_cache = False
                        if cur_oracle is not None:
                            try:
                                ok, _info = cur_oracle(query, fast_answer)
                            except Exception:  # noqa: BLE001
                                ok = False
                            if not ok:
                                log.append(
                                    "[FAST] cached answer failed oracle "
                                    "→ falling through to SLOW (VS)"
                                )
                                oracle_rejected_cache = True
                        if not oracle_rejected_cache:
                            self.metrics.record(
                                query=query, route="FAST",
                                elapsed=time.time() - _solve_start,
                                tokens_used=0,
                                similarity=float(sims[0][0]),
                                answer=fast_answer,
                                score=ep.get('score', 0.8),
                            )
                            return {
                                "route_decision": "FAST",
                                "retrieved_memories": [ep],
                                "generated_candidates": [],
                                "critic_evaluation_table": [],
                                "final_answer": fast_answer,
                                "memory_update_log": log,
                                "alpha": float(sims[0][0]),
                                # Schema-parity with the main solve() return
                                # path: downstream consumers can always read
                                # alpha_t / novelty without checking route.
                                "alpha_t": 0.0,
                                "novelty": 0.0,
                            }

        # =====================================================
        # LAYER 1: PROGRAMMATIC REFLEXES (AST Execution)
        # =====================================================
        # ALL skill code goes through sandbox.safe_call_trigger /
        # safe_execute, which run ASTValidator first. There is no
        # direct exec() of LLM-generated code in this module.
        skill_min_conf = self.config.routing.skill_min_confidence
        for sem in sorted(
            self.memory.semantic_rules,
            key=lambda x: x.get('confidence', 0.5),
            reverse=True,
        ):
            if sem.get('quarantined'):
                # Quarantined rules are explicitly excluded from REFLEX
                # matching to break the conf-oscillation loop.
                continue
            conf = sem.get('confidence', 0.5)
            if conf < skill_min_conf:
                continue

            python_code = sem.get('python_code', '')
            try:
                triggered, namespace = safe_call_trigger(python_code, query)
            except SecurityError as e:
                logging.warning(
                    f"[Sandbox] Rule '{sem.get('pattern')}' rejected by validator: {e}"
                )
                # Hard penalty + drop maturity for code that can't even pass
                # the AST gate. Do NOT keep retrying it.
                self._record_skill_result(sem, success=False)
                sem['confidence'] = 0.05
                continue
            except Exception as e:
                self._record_skill_result(sem, success=False)
                log.append(f"Skill crash (trigger): {e}")
                continue

            if not triggered:
                continue

            try:
                final_answer = safe_call_execute_in_namespace(namespace, query)
            except Exception as e:
                self._record_skill_result(sem, success=False)
                sem['confidence'] = max(0.1, sem.get('confidence', 0.5) - 0.2)
                log.append(f"Skill crash (execute): {e}")
                continue

            if final_answer.startswith("Error"):
                self._record_skill_result(sem, success=False)
                sem['confidence'] = max(0.1, sem.get('confidence', 0.5) - 0.2)
                log.append(f"Skill returned Error: {final_answer[:60]}")
                continue

            maturity = sem.get('maturity', 0)
            shadow_until = self.config.skill.shadow_check_until_maturity

            # Provisional skills get an LLM shadow-check before we trust them.
            # Documented limitation: this verifier is the SAME LLM family that
            # wrote the skill, so it is not an independent oracle. A real
            # deployment should plug an external oracle (pytest / SymPy / unit
            # tests) here via the Oracle interface.
            if maturity < shadow_until:
                check_prompt = (
                    f"### VERIFICATION TASK ###\nTask: {query}\n"
                    f"Proposed Answer: {final_answer}\n\n"
                    "Does this answer correctly solve the task? Respond ONLY with "
                    "'YES' if it is correct, or 'NO' if it is incorrect. "
                    "DO NOT repeat the answer or provide any other text."
                )
                verification, _ = llm.generate_samples(
                    check_prompt, n=1, temperature=0.0
                )
                v_ans = verification[0]['solution'].upper()
                if "YES" not in v_ans:
                    logging.warning(
                        f"[Shadow Mode] REJECTED '{sem['pattern']}': {v_ans}"
                    )
                    log.append(
                        f"Shadow Check FAILED for '{sem['pattern']}'. "
                        "Applying penalty."
                    )
                    sem['confidence'] = max(
                        0.1,
                        sem.get('confidence', 0.5)
                        - self.config.skill.shadow_reject_penalty,
                    )
                    sem['maturity'] = max(0, maturity - 1)
                    # Fall through to next skill (or downstream layers).
                    continue

                logging.info(f"[Shadow Mode] ACCEPTED '{sem['pattern']}'")
                log.append(f"Shadow Check PASSED for '{sem['pattern']}'.")
                # Phase 6: oracle-gated REFLEX. If the caller supplied
                # an oracle and this skill's output fails it, treat
                # the shadow check as a miss and fall through to SLOW.
                cur_oracle = getattr(self, '_current_oracle', None)
                if cur_oracle is not None:
                    try:
                        ok, _info = cur_oracle(query, final_answer)
                    except Exception:  # noqa: BLE001
                        ok = False
                    if not ok:
                        log.append(
                            f"[REFLEX_PROVISIONAL] '{sem['pattern']}' "
                            f"failed oracle \u2192 falling through to SLOW (VS)"
                        )
                        continue
                self._record_skill_result(sem, success=True)
                return {
                    "route_decision": "REFLEX_PROVISIONAL",
                    "retrieved_memories": [],
                    "generated_candidates": [],
                    "critic_evaluation_table": [],
                    "final_answer": final_answer,
                    "memory_update_log": log + [
                        f"Route: REFLEX_PROVISIONAL (conf={conf:.2f}, "
                        f"maturity={maturity})"
                    ],
                    # Schema parity with FAST and the main return path.
                    "alpha": 1.0,
                    "alpha_t": 0.0,
                    "novelty": 0.0,
                }

            # MATURE skill: trust it (unless an oracle vetoes).
            cur_oracle = getattr(self, '_current_oracle', None)
            if cur_oracle is not None:
                try:
                    ok, _info = cur_oracle(query, final_answer)
                except Exception:  # noqa: BLE001
                    ok = False
                if not ok:
                    log.append(
                        f"[REFLEX] mature '{sem['pattern']}' failed "
                        f"oracle \u2192 falling through to SLOW (VS)"
                    )
                    continue
            self._record_skill_result(sem, success=True)
            return {
                "route_decision": "REFLEX",
                "retrieved_memories": [],
                "generated_candidates": [],
                "critic_evaluation_table": [],
                "final_answer": final_answer,
                "memory_update_log": log + [
                    f"Route: REFLEX (mature, conf={conf:.2f})"
                ],
                # Schema parity with FAST and the main return path.
                "alpha": 1.0,
                "alpha_t": 0.0,
                "novelty": 0.0,
            }

        # Reuse embedding from FAST path if available, otherwise compute
        if self.memory.episodic_index.ntotal > 0:
            query_emb = fast_emb          # computed earlier for FAST lookup
        else:
            query_emb = llm.get_embedding(query)
        query_emb_np = np.array([query_emb], dtype=np.float32)

        # Neural Memory surprise — influences candidate budget for SLOW.
        # High surprise → more candidates (harder problem needs wider search).
        novelty = 0.0
        try:
            novelty = self.neural_memory.compute_surprise(query_emb)
            log.append(f"NeuralMemory novelty: {novelty:.4f}")
        except Exception as e:
            log.append(f"NeuralMemory novelty failed: {e}")
        
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
        
        # FAST gate uses RAW similarity (matches Layer-0 HNSW gate). The
        # ``retrieved_eps`` pool is sorted by trust-weighted (effective)
        # similarity, so ``retrieved_eps[0]['similarity']`` is the raw sim
        # of the top *trust-weighted* episode — not necessarily the highest
        # raw sim. Take the actual max to keep the gate consistent with
        # Layer-0 and with how ``self.tau_fast`` was calibrated.
        max_sim = (
            max((float(r.get('similarity', 0.0)) for r in retrieved_eps), default=0.0)
            if retrieved_eps else 0.0
        )

        # Dynamic α: formal amortization coefficient
        # α_t = 1 - exp(-κ·|M_t|) — how "familiar" the task domain is.
        kappa = self.config.amortization.kappa
        memory_size = len(self.memory.episodes)
        alpha_t = 1.0 - np.exp(-kappa * memory_size)
        log.append(f"Amortization α_t={alpha_t:.4f} (|M|={memory_size})")

        route = "SLOW"
        alpha = alpha_t
        candidates = []
        final_answer = ""
        best_cand = None
        prompt_used = ""
        
        if max_sim >= self.tau_fast:
            route = "FAST"
            alpha = max_sim
            log.append(f"Route: FAST PATH (sim: {max_sim:.3f} >= {self.tau_fast})")

            # Pick the actual highest-raw-sim episode (the pool was
            # trust-reranked, so retrieved_eps[0] may not be the
            # top-raw-sim one — see comment above on max_sim).
            best_cand = max(
                retrieved_eps,
                key=lambda r: float(r.get('similarity', 0.0)),
            ).copy()
            best_cand['final_score'] = alpha
            best_cand['solution'] = self._post_process_answer(
                best_cand.get('solution', '') or '', "FAST", log
            )
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
            candidates = self.critic.evaluate(query, candidates, oracle=getattr(self, '_current_oracle', None))
            best_cand = candidates[0] if candidates else None

            if best_cand:
                hybrid_score = (alpha * 1.0) + ((1 - alpha) * best_cand['final_score'])
                best_cand['final_score'] = hybrid_score

                # If the LLM emitted a fenced ```python``` block instead
                # of a direct answer, execute it in the sandbox and use
                # the captured stdout. Centralised in _post_process_answer
                # so FAST/HYBRID/SLOW behave consistently.
                raw_solution = best_cand.get('solution', '') or ''
                executed = self._post_process_answer(raw_solution, "HYBRID", log)
                if executed != raw_solution:
                    best_cand['hybrid_executed_code'] = True
                best_cand['solution'] = executed
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

            # ─── Phase 6: Verified Synthesis branch ──────────────
            # If the caller supplied an oracle, switch SLOW from blind
            # best-of-N to a closed execution-feedback loop. This is
            # the first NARE capability vanilla CoT cannot replicate:
            # the LLM gets to see what its own code printed, plus the
            # oracle's diagnostic, and try again.
            #
            # Without an oracle we degrade to pre-Phase-6 best-of-N,
            # which is provably no worse than vanilla.
            current_oracle = getattr(self, '_current_oracle', None)
            if current_oracle is not None:
                from .synthesis import verified_synthesis
                vs_max = self.config.synthesis.max_attempts
                expected = getattr(self, '_current_expected_hint', None)
                _stokens = 0

                def _propose(prompt_text, prior_attempts):
                    nonlocal _stokens
                    # Attempt 1 is greedy (temperature=0) so it matches
                    # what a deterministic vanilla CoT call would
                    # produce given the same prompt — this lets us
                    # measure VS's value cleanly: any Δ vs vanilla on
                    # attempt 1 is sampling noise, on retries is the
                    # feedback loop. Retries ramp temperature aggressively
                    # (0.5 / 0.7 / 0.9 / 1.0) to break out of bad
                    # fixed-points like "kept printing factorial(30)".
                    n_prior = len(prior_attempts)
                    if n_prior == 0:
                        temp = 0.0
                    else:
                        temp = min(0.3 + 0.2 * n_prior, 1.0)
                    cands, st = llm.generate_samples(
                        prompt_text, n=1,
                        temperature=temp,
                        mode="SLOW",
                    )
                    _stokens += st
                    if cands:
                        return cands[0].get('solution', '') or ''
                    return ''

                vs_result = verified_synthesis(
                    query=query,
                    propose_fn=_propose,
                    oracle=current_oracle,
                    max_attempts=vs_max,
                    expected_hint=expected,
                )
                _solve_tokens += _stokens
                log.append(
                    f"[VS] attempts={vs_result.total_attempts} "
                    f"converged={vs_result.converged}"
                )
                # Materialise as a single candidate so the rest of the
                # SLOW path (critic, post-process, save) keeps working.
                candidates = [{
                    'solution': vs_result.final_answer,
                    'reasoning_trace': (
                        f"VS converged in {vs_result.total_attempts} attempts"
                        if vs_result.converged
                        else f"VS exhausted {vs_result.total_attempts} attempts"
                    ),
                    'final_score': 0.95 if vs_result.converged else 0.30,
                    'critic_passed': vs_result.converged,
                }]
            elif self._is_cold_start() and self.config.bootstrap.cold_start_use_simple_cot:
                # Cold Start: simplified CoT to conserve API tokens.
                log.append("[Bootstrap] Cold start — using simplified CoT instead of best-of-N")
                candidates, _stokens = llm.generate_samples(prompt_used, n=1, temperature=0.5, mode="SLOW")
                _solve_tokens += _stokens
                candidates = self.critic.evaluate(query, candidates, oracle=getattr(self, '_current_oracle', None))
            else:
                # Neural Memory surprise biases candidate budget:
                # higher novelty → wider search (more branches in best-of-N).
                base_breadth = 3
                if novelty > 0.5:
                    breadth = min(base_breadth + 2, 6)
                    log.append(f"High novelty ({novelty:.3f}) → expanded best-of-N breadth={breadth}")
                else:
                    breadth = base_breadth
                # SLOW path: best-of-N with LLM pre-scoring (depth=1).
                # Historical name `tree_of_thoughts` aliased to be honest about
                # what the function actually does — see nare.llm.
                candidates, _stokens = llm.best_of_n_with_prescore(prompt_used, breadth=breadth)
                _solve_tokens += _stokens
                candidates = self.critic.evaluate(query, candidates, oracle=getattr(self, '_current_oracle', None))
            
            best_cand = candidates[0] if candidates else None
            
            if best_cand:
                # Same centralised code-block executor as FAST/HYBRID:
                # if the SLOW best candidate is a fenced ```python``` block
                # (Tasks 14/15/22 in the user's last benchmark log), run
                # it and substitute the executed value as the answer.
                raw_solution = best_cand.get('solution', '') or ''
                best_cand['solution'] = self._post_process_answer(
                    raw_solution, "SLOW", log
                )
                final_answer = best_cand['solution']
                if best_cand.get('final_score', 0.5) >= max(0.70, mem_avg) and self._save_episode(query, query_emb, best_cand, prompt_used):
                    log.append(f"Saved SLOW episode to memory (score {best_cand.get('final_score', 0.5):.2f} >= mem_avg {mem_avg:.2f}).")

        # 3. Calibration
        if best_cand:
            self._calibrate_tau(reward=best_cand.get('final_score', 0.5), fast_path_used=(route == "FAST"))

        # Neural Memory: update with surprise-driven priority.
        # High-surprise episodes get stronger weight in neural memory.
        if best_cand and route in ("SLOW", "HYBRID"):
            importance = max(1.0, 1.0 + novelty)
            try:
                target = np.array(query_emb, dtype=np.float32).flatten()[:self.neural_memory.hidden_dim]
                self.neural_memory.update(
                    np.array(query_emb, dtype=np.float32),
                    target,
                    importance=importance,
                )
            except Exception:
                pass

        result = {
            "route_decision": route,
            "retrieved_memories": retrieved_eps + retrieved_semantics,
            "generated_candidates": candidates,
            "critic_evaluation_table": candidates if candidates else ([best_cand] if best_cand else []),
            "final_answer": final_answer,
            "memory_update_log": log,
            # Schema-parity with the Layer-0 FAST early return: the same
            # set of keys is returned regardless of which path solve()
            # took. ``alpha`` is the route's chosen weighting (sim or
            # alpha_t depending on path).
            "alpha": float(alpha),
            "alpha_t": float(alpha_t),
            "novelty": float(novelty),
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
                    self._rem_sleep_phase()   # REM: dreaming / stress-testing
                    # Background Validation : random episode audit
                    self._background_validate_episodes()
                    # Titans/MIRAS: Neural memory consolidation
                    self.neural_memory.consolidate(self.memory.episodes[-50:])
                    self.neural_memory.save()
                    # Meta-abduction: discover cross-domain patterns
                    self.meta_engine.analyze_skills(
                        self.memory.semantic_rules, self.memory.episodes
                    )
                finally:
                    self._is_sleeping = False
            t = threading.Thread(target=sleep_wrapper, daemon=True)
            t.start()
            self._sleep_thread = t  # Keep reference for joining if needed

        return result

    def _apply_rem_penalty(self, rule: dict, scores: dict, pass_threshold: float,
                           old_conf: float, pattern: str) -> None:
        """Apply REM-cycle penalty + track oscillation for quarantine.

        Replaces the in-place penalty block that used to be duplicated in the
        REM dream-test handler. Crucially this also:
        - increments ``rem_penalty_count`` so the next sleep-refinement can
          apply hysteresis instead of resetting confidence to a fresh score;
        - quarantines rules that have been REM-penalised repeatedly with no
          peak above ``skill.skill_quarantine_peak_threshold``, so we stop
          burning sleep budget on structurally unfixable skills.
        """
        penalty = max(0.05, (pass_threshold - scores["overall"]) * 0.3)
        rule["confidence"] = max(0.10, old_conf - penalty)
        rule["maturity"] = max(0, rule.get("maturity", 0) - 1)
        rule["rem_penalty_count"] = rule.get("rem_penalty_count", 0) + 1

        # Track best-ever confidence so quarantine logic can distinguish
        # "never worked" from "worked once and then regressed".
        rule["peak_confidence"] = max(
            rule.get("peak_confidence", old_conf),
            old_conf,
        )

        cfg = self.config.skill
        if (
            not rule.get("quarantined", False)
            and rule["rem_penalty_count"] >= cfg.skill_quarantine_after_penalties
            and rule["peak_confidence"] < cfg.skill_quarantine_peak_threshold
        ):
            rule["quarantined"] = True
            logging.warning(
                f"[REM Quarantine] '{pattern}' quarantined after "
                f"{rule['rem_penalty_count']} REM penalties "
                f"(peak_confidence={rule['peak_confidence']:.2f} < "
                f"{cfg.skill_quarantine_peak_threshold:.2f}). "
                f"Excluded from REFLEX matching and refinement."
            )

        logging.warning(
            f"[REM] '{pattern}' repair did not improve "
            f"(score={scores['overall']:.2f}). "
            f"Penalizing: conf {old_conf:.2f} -> {rule['confidence']:.2f} "
            f"(rem_penalty_count={rule['rem_penalty_count']})"
        )

    def _record_skill_result(self, rule: dict, success: bool):
        """Track skill execution history and apply Penalty Backpropagation .

        When a skill fails, its confidence drops and the penalty is
        propagated to all source episodes via their τ_i trust coefficients.
        """
        history = rule.get('score_history', [])
        history.append(1.0 if success else 0.0)
        rule['score_history'] = history[-20:]
        
        rule['reuse_rate'] = rule.get('reuse_rate', 0) + 1
        
        if len(history) >= 3:
            recent = history[-10:]
            rolling_success = sum(recent) / len(recent)
            old_conf = rule.get('confidence', 0.5)
            rule['confidence'] = min(0.99, round((rolling_success * 0.7) + (old_conf * 0.3) + 0.01, 3))
            
        if success:
            rule['success_streak'] = rule.get('success_streak', 0) + 1
            if rule['success_streak'] >= self.config.skill.success_streak_for_maturity:
                rule['maturity'] = rule.get('maturity', 0) + 1
                rule['success_streak'] = 0
                logging.info(f"[Evolution] Skill '{rule['pattern']}' reached maturity level {rule['maturity']}")
        else:
            rule['success_streak'] = 0
            rule['maturity'] = max(0, rule.get('maturity', 0) - 1)

        maturity_bonus = min(0.3, rule.get('maturity', 0) * 0.1)
        conf = rule.get('confidence', 0.5)
        rule['global_score'] = round((conf * 0.7) + maturity_bonus, 3)

        # --- Penalty Backpropagation  ---
        # If skill fails, propagate penalty to source episodes. We
        # resolve indices lazily through stable content keys: positional
        # ids stored at crystallisation time become stale the moment
        # _sleep_phase deletes the cluster episodes (Devin Review #1).
        delta_v = 1.0 if success else -1.0
        source_keys = rule.get('source_episode_keys') or []
        if source_keys:
            ep_indices = self.memory.find_episode_indices_by_keys(source_keys)
        else:
            # Backwards-compat for rules created before content keys
            # existed: trust the legacy positional ids only when the
            # episodes list hasn't shrunk below them.
            legacy_ids = rule.get('source_episode_ids', [])
            ep_indices = [
                int(i) for i in legacy_ids
                if 0 <= int(i) < len(self.memory.episodes)
            ]

        if ep_indices:
            gamma = self.config.immune.penalty_backprop_gamma
            # update_episode_tau already takes _lock + does its own bounds
            # check, so we don't need to wrap this loop.
            for ep_id in ep_indices:
                self.memory.update_episode_tau(ep_id, delta_v * gamma)
            if not success:
                logging.info(
                    f"[Penalty Backprop] Skill '{rule.get('pattern')}' penalty "
                    f"distributed to {len(ep_indices)} source episodes "
                    f"(of {len(source_keys) or len(rule.get('source_episode_ids', []))} originally)"
                )

        # If skill keeps failing and source episodes are toxic, add
        # suppression. Snapshot the relevant episodes UNDER THE LOCK to
        # avoid an IndexError race with the background sleep thread,
        # which may delete episodes between our index resolution and the
        # direct ``self.memory.episodes[ep_id]`` access (Devin Review on
        # commit 436b060). add_suppression_rule itself takes the lock so
        # we release before calling it.
        if not success and rule.get('confidence', 0.5) < 0.2 and ep_indices:
            theta = self.config.immune.theta_immune
            emb_dim = self.memory.embedding_dim
            to_suppress: List[Tuple[str, str, np.ndarray]] = []
            with self.memory._lock:
                for ep_id in ep_indices:
                    if not (0 <= ep_id < len(self.memory.episodes)):
                        continue
                    ep = self.memory.episodes[ep_id]
                    if ep.get('tau', 1.0) >= theta:
                        continue
                    to_suppress.append((
                        ep.get('query', ''),
                        ep.get('solution', ''),
                        np.array(
                            ep.get('embedding', [0.0] * emb_dim),
                            dtype=np.float32,
                        ),
                    ))
            for q, s, emb in to_suppress:
                self.memory.add_suppression_rule(q, s, emb)

        # Find and update rule in memory
        for i, r in enumerate(self.memory.semantic_rules):
            if r.get('pattern') == rule.get('pattern'):
                self.memory.semantic_rules[i] = rule
                self.memory.save()
                break

