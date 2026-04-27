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
class RoutingConfig:
    """Thresholds for the 4-way router (FAST / HYBRID / SLOW / REFLEX)."""

    # Cosine-similarity thresholds against the episodic FAISS index.
    tau_fast: float = 0.98          # >= this -> FAST path (return cached solution)
    tau_hybrid: float = 0.75        # >= this -> HYBRID path (delta-reasoning)
    tau_min: float = 0.95           # lower bound for tau_fast during calibration
    tau_max: float = 0.99           # upper bound for tau_fast during calibration

    # Calibration step: how aggressively tau_fast moves on feedback.
    calibration_lr: float = 0.02

    # Per-rule confidence floor: rules below this are not even considered.
    skill_min_confidence: float = 0.40

    # Semantic-rule similarity / confidence gate when injecting into context.
    semantic_inject_min_sim: float = 0.85
    semantic_inject_min_conf: float = 0.70


@dataclass(frozen=True)
class SleepConfig:
    """Sleep / consolidation phase parameters.

    NOTE: previous default was "trigger sleep if ANY pair of episodes has
    similarity > 0.6". For 3072-dim Gemini embeddings this fires on almost
    any 3 paraphrased queries, leading to constant background work and
    spurious skill creation. The new default requires a denser cluster.
    """

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

    # Population size during initial extract_heuristic_rule attempts.
    extract_population_temps: Tuple[float, ...] = (0.2, 0.8)
    extract_max_outer_attempts: int = 3


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


# Singleton — modules import this directly. Override per-test by passing
# a custom NareConfig into the public constructors.
DEFAULT_CONFIG = NareConfig()
