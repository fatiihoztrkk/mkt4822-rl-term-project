"""YAML configuration loader."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from mecha_agent_cli.config.defaults import default_config
from mecha_agent_cli.config.schema import AppConfig, ModelsConfig, SecurityConfig, ValidationConfig
from mecha_agent_cli.core.errors import ConfigError


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw_data: object = yaml.safe_load(handle) or {}
    if not isinstance(raw_data, Mapping):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    mapping = cast(Mapping[object, Any], raw_data)
    return {str(key): value for key, value in mapping.items()}


def load_config(repo_root: Path | None = None) -> AppConfig:
    """Load configuration from project defaults and local ``configs`` files."""
    config = default_config()
    if repo_root is None:
        return config
    config_dir = repo_root / "configs"
    default_data = _read_yaml(config_dir / "default.yaml")
    models_data = _read_yaml(config_dir / "models.yaml")
    validation_data = _read_yaml(config_dir / "validation.yaml")
    security_data = _read_yaml(config_dir / "security.yaml")

    merged = config.model_dump()
    for key, value in default_data.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    if models_data:
        merged["models"] = ModelsConfig.model_validate(models_data).model_dump()
    if validation_data:
        merged["validation"].update(ValidationConfig.model_validate(validation_data).model_dump())
    if security_data:
        merged["security"].update(SecurityConfig.model_validate(security_data).model_dump())
    return AppConfig.model_validate(merged)


def dump_config(config: AppConfig) -> str:
    """Serialize config to YAML for CLI display."""
    return yaml.safe_dump(config.model_dump(), sort_keys=False)
