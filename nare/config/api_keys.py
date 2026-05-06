"""API Key Management System

Manages API keys for multiple LLM providers:
- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)

Keys are stored in ~/.nare/config.json or environment variables.
"""

import os
import json
from nare.utils.logger import get_logger
from pathlib import Path
from typing import Optional, Dict

log = get_logger("nare.config.api_keys")


class APIKeyManager:
    """Manage API keys for multiple LLM providers."""

    SUPPORTED_PROVIDERS = {
        "anthropic": {
            "name": "Anthropic (Claude)",
            "env_var": "ANTHROPIC_API_KEY",
            "url": "https://console.anthropic.com/settings/keys",
            "models": [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229"
            ]
        },
        "openai": {
            "name": "OpenAI (GPT)",
            "env_var": "OPENAI_API_KEY",
            "url": "https://platform.openai.com/api-keys",
            "models": [
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo"
            ]
        },
        "google": {
            "name": "Google (Gemini)",
            "env_var": "GOOGLE_API_KEY",
            "url": "https://makersuite.google.com/app/apikey",
            "models": [
                "gemini-pro",
                "gemini-pro-vision"
            ]
        }
    }

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize API key manager.

        Args:
            config_dir: Directory for config file (default: ~/.nare)
        """
        if config_dir is None:
            config_dir = Path.home() / ".nare"

        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "config.json"
        self._keys: Dict[str, str] = {}
        self._models: Dict[str, str] = {}

        # Create config directory if needed
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load keys
        self._load_keys()

    def _load_keys(self):
        """Load API keys from config file and environment variables."""
        # Load from config file
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self._keys = config.get('api_keys', {})
                    self._models = config.get('models', {})
                    log.info(f"[APIKeys] Loaded {len(self._keys)} keys from config")
            except Exception as e:
                log.warning(f"[APIKeys] Failed to load config: {e}")

        # Override with environment variables
        for provider, info in self.SUPPORTED_PROVIDERS.items():
            env_key = os.getenv(info['env_var'])
            if env_key:
                self._keys[provider] = env_key
                log.info(f"[APIKeys] Loaded {provider} key from environment")

    def get_key(self, provider: str) -> Optional[str]:
        """Get API key for provider.

        Args:
            provider: Provider name (anthropic, openai, google)

        Returns:
            API key or None if not found
        """
        return self._keys.get(provider)

    def set_key(self, provider: str, key: str, save: bool = True):
        """Set API key for provider.

        Args:
            provider: Provider name
            key: API key
            save: Whether to save to config file
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        self._keys[provider] = key
        log.info(f"[APIKeys] Set {provider} key")

        if save:
            self._save_keys()

    def _save_keys(self):
        """Save API keys to config file."""
        try:
            config = {}
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)

            config['api_keys'] = self._keys
            config['models'] = self._models

            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)

            log.info(f"[APIKeys] Saved {len(self._keys)} keys to config")
        except Exception as e:
            log.error(f"[APIKeys] Failed to save config: {e}")

    def get_model(self, provider: str) -> Optional[str]:
        """Get selected model for provider.

        Args:
            provider: Provider name

        Returns:
            Model name or None if not set
        """
        return self._models.get(provider)

    def set_model(self, provider: str, model: str, save: bool = True):
        """Set model for provider.

        Args:
            provider: Provider name
            model: Model name
            save: Whether to save to config file
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        self._models[provider] = model
        log.info(f"[APIKeys] Set {provider} model to {model}")

        if save:
            self._save_keys()

    def has_any_key(self) -> bool:
        """Check if any API key is configured."""
        return len(self._keys) > 0

    def get_missing_providers(self) -> list:
        """Get list of providers without API keys."""
        return [
            provider for provider in self.SUPPORTED_PROVIDERS.keys()
            if provider not in self._keys
        ]

    def prompt_for_keys(self, required_provider: Optional[str] = None) -> bool:
        """Prompt user to enter API keys via CLI.

        Args:
            required_provider: If specified, only prompt for this provider

        Returns:
            True if at least one key was entered
        """
        from nare.cli.display import ui

        ui.console.print()
        ui.console.print("  [#D77757]⚠ API Key Required[/]")
        ui.console.print()

        if required_provider:
            providers = [required_provider]
        else:
            providers = self.get_missing_providers()

        if not providers:
            return True

        ui.console.print("  NARE needs an API key to function.")
        ui.console.print("  You can use any of these providers:")
        ui.console.print()

        for i, provider in enumerate(self.SUPPORTED_PROVIDERS.keys(), 1):
            info = self.SUPPORTED_PROVIDERS[provider]
            status = "✓" if provider in self._keys else " "
            ui.console.print(f"  [{status}] {i}. {info['name']}")
            ui.console.print(f"      Get key: {info['url']}")
            ui.console.print()

        ui.console.print("  Enter API key for any provider (or press Enter to skip):")
        ui.console.print()

        for provider in providers:
            info = self.SUPPORTED_PROVIDERS[provider]
            ui.console.print(f"  [#999999]{info['name']}:[/]")
            key = input("  API Key: ").strip()

            if key:
                self.set_key(provider, key, save=True)
                ui.console.print(f"  [#4EBA65]✓ {info['name']} key saved[/]")
                ui.console.print()

                # Prompt for model selection
                ui.console.print(f"  [#999999]Select model (or press Enter for default):[/]")
                models = info.get('models', [])
                for i, model in enumerate(models, 1):
                    ui.console.print(f"  {i}. {model}")
                ui.console.print()

                model_choice = input("  Model number: ").strip()
                if model_choice.isdigit():
                    idx = int(model_choice) - 1
                    if 0 <= idx < len(models):
                        selected_model = models[idx]
                        self.set_model(provider, selected_model, save=True)
                        ui.console.print(f"  [#4EBA65]✓ Model set to {selected_model}[/]")
                        ui.console.print()

                return True

        return False


# Global instance
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def ensure_api_key(provider: str = "anthropic") -> Optional[str]:
    """Ensure API key is available, prompt if not.

    Args:
        provider: Preferred provider

    Returns:
        API key or None if user declined
    """
    manager = get_api_key_manager()

    # Check if key exists
    key = manager.get_key(provider)
    if key:
        return key

    # Prompt for key
    if manager.prompt_for_keys(required_provider=provider):
        return manager.get_key(provider)

    return None
