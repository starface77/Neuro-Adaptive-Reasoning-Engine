"""Centralized configuration for VARE (Verified Amortized Reasoning Engine).

Three components, two routes, one goal: amortize LLM reasoning through
verified episodic memory and compiled skills.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoutingConfig:
    """Two-route threshold: FAST (cache hit) vs VERIFIED_RETRY (synthesis)."""
    tau_fast: float = 0.92
    tau_min: float = 0.88
    tau_max: float = 0.96
    calibration_lr: float = 0.02


@dataclass(frozen=True)
class SynthesisConfig:
    """Verified synthesis loop parameters."""
    max_retries: int = 5
    initial_temperature: float = 0.8
    refinement_temperature: float = 0.4


@dataclass(frozen=True)
class MemoryConfig:
    """Episodic memory: HNSW index, activation decay, dedup, pruning."""
    dedup_threshold: float = 0.97
    strength_decay_constant: float = 86400.0  # 1 day in seconds
    strength_boost_on_use: float = 1.0
    tau_prune: float = 0.05
    max_episodes: int = 5000


@dataclass(frozen=True)
class LibraryLearningConfig:
    """Background skill compilation from episode clusters."""
    cluster_density_threshold: int = 3
    cluster_similarity_threshold: float = 0.65
    min_skill_confidence: float = 0.80
    skill_dedup_threshold: float = 0.90


@dataclass(frozen=True)
class SkillValidationConfig:
    """Weights and policy for ``llm._validate_skill``.

    Kept for backward compatibility with existing tests.
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
