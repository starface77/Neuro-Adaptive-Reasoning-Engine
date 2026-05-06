"""Configuration module for NARE.

Includes:
- config: Main configuration classes (NareConfig, DEFAULT_CONFIG, etc.)
- api_keys: API key management for multiple LLM providers
"""

# Import from config.py
from .config import (
    BootstrapConfig,
    ImmuneSystemConfig,
    RoutingConfig,
    SleepConfig,
    CriticConfig,
    SkillLifecycleConfig,
    RetrievalConfig,
    SkillValidationConfig,
    AmortizationConfig,
    SynthesisConfig,
    NareConfig,
    DEFAULT_CONFIG,
)

# Import from api_keys.py
from .api_keys import (
    APIKeyManager,
    get_api_key_manager,
    ensure_api_key,
)

__all__ = [
    # Config classes
    "BootstrapConfig",
    "ImmuneSystemConfig",
    "RoutingConfig",
    "SleepConfig",
    "CriticConfig",
    "SkillLifecycleConfig",
    "RetrievalConfig",
    "SkillValidationConfig",
    "AmortizationConfig",
    "SynthesisConfig",
    "NareConfig",
    "DEFAULT_CONFIG",
    # API keys
    "APIKeyManager",
    "get_api_key_manager",
    "ensure_api_key",
]

