"""Default config factory."""

from mecha_agent_cli.config.schema import AppConfig


def default_config() -> AppConfig:
    """Return a fresh default app configuration."""
    return AppConfig()
