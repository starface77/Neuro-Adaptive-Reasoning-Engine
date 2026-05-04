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

    cold_start_threshold: int = 10

    cold_start_use_simple_cot: bool = True

    seed_examples_path: str = ""

    load_seeds_on_init: bool = True

@dataclass(frozen=True)
class ImmuneSystemConfig:
    """Immune system for memory quality.

    Each episode carries a trust coefficient τ_i ∈ [0,1].  After each
    use the coefficient is updated: τ_i ← τ_i + γ·ΔV.  Episodes whose
    τ falls below θ_immune are quarantined/deleted.
    """

    initial_tau: float = 0.5

    tau_lr: float = 0.10

    theta_immune: float = 0.15

    penalty_backprop_gamma: float = 0.5

    background_audit_count: int = 3

    max_suppression_rules: int = 100

@dataclass(frozen=True)
class RoutingConfig:

    tau_fast: float = 0.75
    tau_hybrid: float = 0.65
    tau_reflex: float = 0.70
    tau_min: float = 0.60
    tau_max: float = 0.90
    calibration_lr: float = 0.02

    tau_fast_code: float = 0.75
    tau_fast_pattern: float = 0.70
    tau_fast_reasoning: float = 0.75
    skill_min_confidence: float = 0.70
    semantic_inject_min_sim: float = 0.85
    semantic_inject_min_conf: float = 0.70

    adaptive_thresholds: bool = False
    adaptive_target_accuracy: float = 0.70
    adaptive_adjustment_rate: float = 0.05

@dataclass(frozen=True)
class SleepConfig:
    """Sleep / consolidation phase parameters.

    NOTE: previous default was "trigger sleep if ANY pair of episodes has
    similarity > 0.6". For 3072-dim Gemini embeddings this fires on almost
    any 3 paraphrased queries, leading to constant background work and
    spurious skill creation. The new default requires a denser cluster.
    """

    enabled: bool = True

    max_episodes_before_sleep: int = 20

    cluster_density_threshold: int = 2
    cluster_similarity_threshold: float = 0.60

    existing_rule_match_threshold: float = 0.70

    weak_rule_global_score: float = 0.40
    weak_rule_min_cycles: int = 2

    episode_dedup_threshold: float = 0.95
    semantic_dedup_threshold: float = 0.90
    fact_dedup_threshold: float = 0.92

    fading_retention_threshold: float = 0.05

    use_query_fingerprint_gate: bool = False
    query_fingerprint_threshold: float = 0.50

    use_holdout_validation: bool = True
    holdout_n: int = 1
    holdout_min_cluster_size: int = 3
    holdout_min_accuracy: float = 0.5

@dataclass(frozen=True)
class CriticConfig:
    """Hybrid critic weights and Elo settings."""

    w_llm: float = 0.6
    w_rule: float = 0.4
    w_neural: float = 0.0

    elo_k_factor: float = 32.0
    elo_initial_rating: float = 1200.0

@dataclass(frozen=True)
class SkillLifecycleConfig:
    """Maturity / promotion / shadow-check lifecycle for executable reflexes."""

    shadow_check_until_maturity: int = 3
    success_streak_for_maturity: int = 5
    history_window: int = 20

    shadow_reject_penalty: float = 0.30

    repair_max_attempts: int = 2

    skill_refinement_max_bump_after_penalty: float = 0.10

    skill_quarantine_after_penalties: int = 3
    skill_quarantine_peak_threshold: float = 0.50

    extract_population_temps: Tuple[float, ...] = (0.2, 0.8)
    extract_max_outer_attempts: int = 3

    rem_min_replay_episodes: int = 2

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

    include_positive_stress: bool = False

    minimum_trigger_accuracy: float = 0.50
    minimum_execute_accuracy: float = 0.40

    use_heuristic_overlap_fallback: bool = False

@dataclass(frozen=True)
class AmortizationConfig:
    """Formal amortization metrics.

    α_t = 1 - exp(-κ·|M_t|)   — coverage ratio
    C_t = (1-α_t)·C_LLM + α_t·C_mem  — blended cost
    """

    kappa: float = 0.05

    c_llm: float = 2000.0

    c_mem: float = 0.0

@dataclass(frozen=True)
class SynthesisConfig:
    max_attempts: int = 8
    max_attempts_hard: int = 12
    use_subprocess: bool = True

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

DEFAULT_CONFIG = NareConfig()
