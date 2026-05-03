"""Centralized configuration for VARE (Verified Amortized Reasoning Engine).

Three independent components:
  M_cache  — HNSW episodic memory with activation-based forgetting
  G_θ      — Fixed-weight LLM generator with self-refinement
  V_sandbox — Formal verifier (AST + subprocess sandbox)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoutingConfig:
    """Two-way routing: FAST (cache hit) or VERIFIED_RETRY (synthesis)."""

    # Cosine-similarity threshold: >= tau_fast → return cached answer.
    tau_fast: float = 0.92

    # Calibration bounds (adjusted by outcome feedback).
    tau_min: float = 0.88
    tau_max: float = 0.96
    calibration_lr: float = 0.02


@dataclass(frozen=True)
class SynthesisConfig:
    """Verified Code Synthesis loop parameters."""

    # Maximum attempts before declaring failure.
    max_retries: int = 5

    # Temperature for initial generation vs refinement.
    initial_temperature: float = 0.8
    refinement_temperature: float = 0.4


@dataclass(frozen=True)
class MemoryConfig:
    """M_cache parameters: HNSW index + activation-based forgetting."""

    # Episode dedup threshold (cosine).
    dedup_threshold: float = 0.97

    # Activation-based forgetting (Ebbinghaus-inspired).
    # activation = activation * exp(-dt/S) + boost_on_use
    # Prune if activation < tau_prune.
    strength_decay_constant: float = 86400.0  # S in seconds (1 day)
    strength_boost_on_use: float = 1.0
    tau_prune: float = 0.05

    # Maximum memory size before forced pruning.
    max_episodes: int = 5000


@dataclass(frozen=True)
class LibraryLearningConfig:
    """Background skill compilation (Library Learning)."""

    # Cluster detection for consolidation.
    cluster_density_threshold: int = 3
    cluster_similarity_threshold: float = 0.65

    # Minimum confidence for a compiled skill to be stored.
    min_skill_confidence: float = 0.80

    # Skill validation: must pass all cluster tasks in sandbox.
    skill_dedup_threshold: float = 0.90


@dataclass(frozen=True)
class SkillValidationConfig:
    """Weights and policy for skill validation.

    Retained from previous architecture for backward compatibility
    with oracle-based tests.
    """

    w_trigger: float = 0.35
    w_execute: float = 0.55
    w_negative_trap: float = 0.10
    w_positive_stress: float = 0.0

    include_positive_stress: bool = False

    minimum_trigger_accuracy: float = 0.50
    minimum_execute_accuracy: float = 0.40


@dataclass(frozen=True)
class VareConfig:
    """Top-level VARE configuration."""

    routing: RoutingConfig = field(default_factory=RoutingConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    library: LibraryLearningConfig = field(default_factory=LibraryLearningConfig)
    skill_validation: SkillValidationConfig = field(
        default_factory=SkillValidationConfig
    )


# Backward-compatible aliases
NareConfig = VareConfig
DEFAULT_CONFIG = VareConfig()
