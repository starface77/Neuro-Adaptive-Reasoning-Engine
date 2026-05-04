"""Centralized configuration for NARE.

All previously-magic numbers are collected here so that they can be
tuned and ablated rather than scattered across modules. None of these
values are claimed to be optimal — they are starting points and SHOULD
be calibrated on a held-out validation set before any benchmark claim
is made.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class BootstrapConfig:
    """Cold-start / bootstrap strategy.

    When memory is empty (α ≈ 0), every query goes through expensive
    SLOW path.  Bootstrap pre-warms the cache with seed examples and
    uses simplified CoT instead of full ToT until a base pool is built.
    """

    # Number of episodes below which the system is in "cold start" mode.
    cold_start_threshold: int = 10

    # When in cold-start, use simplified CoT (n=1, no ToT) to save tokens.
    cold_start_use_simple_cot: bool = True

    # Path to a JSON file with pre-warmed seed examples.
    # Each entry: {"query": str, "solution": str, "reasoning_trace": str}
    seed_examples_path: str = ""

    # When True, bootstrap seeds are loaded during Agent initialization.
    load_seeds_on_init: bool = True


@dataclass(frozen=True)
class ImmuneSystemConfig:
    """Immune system for memory quality.

    Each episode carries a trust coefficient τ_i ∈ [0,1].  After each
    use the coefficient is updated: τ_i ← τ_i + γ·ΔV.  Episodes whose
    τ falls below θ_immune are quarantined/deleted.
    """

    # Initial trust for new episodes.
    initial_tau: float = 0.5

    # Learning rate γ for trust updates.
    tau_lr: float = 0.10

    # Deletion threshold: τ < this → episode removed.
    theta_immune: float = 0.15

    # Penalty backpropagation discount: how much of a skill's penalty
    # propagates back to its source episodes.
    penalty_backprop_gamma: float = 0.5

    # Background validation: how many random episodes to audit per
    # sleep cycle.
    background_audit_count: int = 3

    # Suppression rules: max number of suppression entries.
    max_suppression_rules: int = 100


@dataclass(frozen=True)
class RoutingConfig:
    # Adaptive thresholds - auto-calibrate based on task domain
    tau_fast: float = 0.50  # Lowered significantly for faster hits
    tau_hybrid: float = 0.60  # Lowered to trigger delta reasoning easier
    tau_reflex: float = 0.50
    tau_min: float = 0.40
    tau_max: float = 0.85
    calibration_lr: float = 0.02

    # Domain-specific overrides (auto-detected)
    tau_fast_code: float = 0.65  # Code tasks - lowered from 0.80
    tau_fast_pattern: float = 0.50
    tau_fast_reasoning: float = 0.60
    skill_min_confidence: float = 0.40
    semantic_inject_min_sim: float = 0.85
    semantic_inject_min_conf: float = 0.70

    # Adaptive thresholds: auto-adjust tau_fast based on model performance
    # When enabled, system lowers thresholds for weaker models
    adaptive_thresholds: bool = False
    adaptive_target_accuracy: float = 0.70  # Target accuracy for calibration
    adaptive_adjustment_rate: float = 0.05  # How much to adjust per cycle


@dataclass(frozen=True)
class SleepConfig:
    """Sleep / consolidation phase parameters.

    NOTE: previous default was "trigger sleep if ANY pair of episodes has
    similarity > 0.6". For 3072-dim Gemini embeddings this fires on almost
    any 3 paraphrased queries, leading to constant background work and
    spurious skill creation. The new default requires a denser cluster.
    """

    # Enable/disable sleep phase entirely
    enabled: bool = True

    # Hard size trigger: if more than this many episodes accumulate, run sleep.
    max_episodes_before_sleep: int = 200

    # Density trigger: a single point must have at least this many neighbours
    # above the similarity threshold for the cluster to be considered "dense".
    cluster_density_threshold: int = 3
    cluster_similarity_threshold: float = 0.65

    # Threshold for "this rule already exists, refine instead of recreate".
    existing_rule_match_threshold: float = 0.70

    # Threshold below which a rule is considered weak and prunable after
    # multiple sleep cycles.
    weak_rule_global_score: float = 0.40
    weak_rule_min_cycles: int = 2

    # Episode dedup threshold (cosine).
    episode_dedup_threshold: float = 0.95
    semantic_dedup_threshold: float = 0.90
    fact_dedup_threshold: float = 0.92

    # Forgetting curve (Ebbinghaus-inspired): retention = exp(-t / (s * 24h)).
    # Below this retention, the episode is dropped at the next sleep cycle.
    fading_retention_threshold: float = 0.05

    # Secondary cluster gate via deterministic query structural
    # fingerprint (numbers, operations, intent tags). When enabled, a
    # candidate dense embedding cluster only fires the sleep trigger
    # if the average pairwise multiset Jaccard over its members'
    # query fingerprints also exceeds ``query_fingerprint_threshold``.
    # Off by default to preserve previous behaviour; opt in via
    # NareConfig override.
    use_query_fingerprint_gate: bool = False
    query_fingerprint_threshold: float = 0.50

    # Phase 6: held-out validation for newly-crystallized skills.
    # Split the cluster into ``cluster_size - holdout_n`` train + ``holdout_n``
    # held-out episodes. Induct the skill on train; validate on held-out
    # via the rule's own ``execute()`` + the same oracle the validator
    # uses. If the held-out execute_accuracy < ``holdout_min_accuracy``
    # the rule is REJECTED — episodes stay alive and crystallization is
    # retried on the next sleep cycle (possibly with a larger cluster).
    #
    # Active only when cluster_size >= ``holdout_min_cluster_size``.
    # With cluster_size below that, the held-out gate is silently
    # skipped (you can't hold one out of a 2-episode cluster).
    use_holdout_validation: bool = True
    holdout_n: int = 1
    holdout_min_cluster_size: int = 3
    holdout_min_accuracy: float = 0.5


@dataclass(frozen=True)
class CriticConfig:
    """Hybrid critic weights and Elo settings."""

    w_llm: float = 0.6              # weight of pairwise-judge Elo score
    w_rule: float = 0.4             # weight of rule-based heuristic score
    w_neural: float = 0.0           # reserved for future neural critic

    elo_k_factor: float = 32.0
    elo_initial_rating: float = 1200.0


@dataclass(frozen=True)
class SkillLifecycleConfig:
    """Maturity / promotion / shadow-check lifecycle for executable reflexes."""

    shadow_check_until_maturity: int = 3   # below this maturity, verify each call
    success_streak_for_maturity: int = 5
    history_window: int = 20

    # Penalty applied to confidence when shadow-check rejects.
    shadow_reject_penalty: float = 0.30

    # Repair-loop: how many times to ask the LLM to fix a failing skill.
    repair_max_attempts: int = 2

    # Hysteresis: maximum bump that GRADED REFINEMENT may apply to a rule
    # which has been REM-penalised at least once. Without this clamp, a
    # rule that fails REM (conf 0.45 -> 0.23) gets fully reset back to its
    # fresh-validation score on the next sleep cycle, oscillating forever
    # — see the benchmark log "Email Extraction" 0.45 ↔ 0.23 cycle.
    skill_refinement_max_bump_after_penalty: float = 0.10

    # After this many REM-penalty events with peak confidence still below
    # ``skill_quarantine_peak_threshold``, the rule is quarantined: it is
    # excluded from REFLEX matching and refinement. This stops the system
    # from spending sleep budget on structurally unfixable skills.
    skill_quarantine_after_penalties: int = 3
    skill_quarantine_peak_threshold: float = 0.50

    # Population size during initial extract_heuristic_rule attempts.
    extract_population_temps: Tuple[float, ...] = (0.2, 0.8)
    extract_max_outer_attempts: int = 3

    # REM: cached-replay parameters. REM first runs the skill against
    # every cached episode whose ``trigger()`` fires; if at least
    # ``rem_min_replay_episodes`` episodes triggered, the cached-replay
    # score is used as the primary REM signal (instead of LLM-generated
    # stress tests, whose POSITIVE labels are self-referential).
    rem_min_replay_episodes: int = 2
    # Below this replay score (fraction of triggered cached episodes
    # whose output matches the verified solution under the strict
    # oracle), the skill is considered to have regressed and goes
    # through the repair loop.
    rem_replay_pass_threshold: float = 0.80


@dataclass(frozen=True)
class RetrievalConfig:
    """RL-retriever and graph-memory knobs."""

    rl_lr: float = 0.01
    rl_discount: float = 0.95
    rl_epsilon_initial: float = 0.10
    rl_epsilon_decay: float = 0.999
    rl_epsilon_floor: float = 0.01
    rl_weight_clip_norm: float = 10.0

    # Score blending for re-rank: combined = w_sim * sim + w_rl * sigmoid(rl_value)
    # NOTE: previous code used a raw additive blend without normalization,
    # which is unsound (sim is bounded in [0,1], rl_value was unbounded).
    rerank_w_sim: float = 0.6
    rerank_w_rl: float = 0.4

    graph_default_edge_weight: float = 1.0
    graph_min_traversal_weight: float = 0.20
    graph_decay_per_rem: float = 0.02


@dataclass(frozen=True)
class SkillValidationConfig:
    """Weights and policy for ``llm._validate_skill``.

    The previous default mixed an LLM-judged stress-test signal (30% of
    ``overall``) with two real-ground-truth signals (trigger correctness
    on labelled episodes, execute correctness against verified
    solutions). When the same model writes the skill, the stress tests,
    AND the labels, that 30% is self-referential.

    The new defaults:

      * Trigger accuracy (real, labelled originals): 0.35
      * Execute accuracy (real, oracle-checked against verified
        solutions): 0.55
      * NEGATIVE-trap accuracy (still real signal: must NOT trigger
        on adversarial off-distribution queries): 0.10
      * POSITIVE LLM-judged stress accuracy: 0.0 by default. It is
        still computed and surfaced as ``positive_no_crash_rate`` for
        diagnostics, but no longer biases the ``overall`` score that
        gates promotion.

    A user with an external oracle can flip ``include_positive_stress``
    on and the POSITIVE stress signal will be consumed only when the
    oracle agrees \u2014 not when only the model agrees with itself.
    """

    w_trigger: float = 0.35
    w_execute: float = 0.55
    w_negative_trap: float = 0.10
    w_positive_stress: float = 0.0

    # If True, POSITIVE stress tests are checked through the oracle (if
    # one is provided) and contribute ``w_positive_stress`` to overall.
    # If False (default), POSITIVE stress is computed only as a no-crash
    # diagnostic and excluded from overall.
    include_positive_stress: bool = False

    # Hard threshold below which a generated skill is rejected outright
    # regardless of stress signal.
    minimum_trigger_accuracy: float = 0.50
    minimum_execute_accuracy: float = 0.40

    # When neither an episode oracle_spec nor a caller-provided oracle
    # is supplied, the validator needs *some* fallback. Default since
    # this PR: ``cached_episode_oracle`` — strict normalized-exact /
    # strict numeric-set match against the cached solution. The old
    # ``heuristic_overlap_oracle`` (20% numeric overlap) is leaky and
    # is now opt-in only.
    use_heuristic_overlap_fallback: bool = False


@dataclass(frozen=True)
class AmortizationConfig:
    """Formal amortization metrics.

    α_t = 1 - exp(-κ·|M_t|)   — coverage ratio
    C_t = (1-α_t)·C_LLM + α_t·C_mem  — blended cost
    """

    # κ — coverage growth rate.  Higher → faster saturation.
    kappa: float = 0.05

    # Average cost of a full System-2 inference (tokens).
    c_llm: float = 2000.0

    # Average cost of a memory retrieval (tokens equivalent).
    c_mem: float = 0.0


@dataclass(frozen=True)
class SynthesisConfig:
    max_attempts: int = 8
    max_attempts_hard: int = 12
    use_subprocess: bool = True

    # Number of candidate solutions to generate (best-of-N sampling)
    # Higher values improve accuracy for weaker models but cost more tokens
    slow_path_breadth: int = 3


@dataclass(frozen=True)
class NareConfig:
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    skill: SkillLifecycleConfig = field(default_factory=SkillLifecycleConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    skill_validation: SkillValidationConfig = field(
        default_factory=SkillValidationConfig
    )
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    immune: ImmuneSystemConfig = field(default_factory=ImmuneSystemConfig)
    amortization: AmortizationConfig = field(default_factory=AmortizationConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)


# Singleton — modules import this directly. Override per-test by passing
# a custom NareConfig into the public constructors.
DEFAULT_CONFIG = NareConfig()
