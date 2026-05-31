"""Placeholder arm registry for course starter repositories.

This module intentionally provides only a minimal, import-safe action
registry. Students are expected to design and implement the real action
space as part of the learning assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mecha_agent_cli.config.schema import ModelProfile


@dataclass(frozen=True)
class Arm:
    """Minimal action descriptor used by placeholder learning modules."""

    arm_id: str
    profile_name: str
    overrides: dict[str, Any] = field(default_factory=dict[str, Any])
    description: str = ""

    def apply(self, base: ModelProfile) -> ModelProfile:
        """Return ``base`` unchanged in placeholder mode."""
        del self
        return base


ARM_REGISTRY: tuple[Arm, ...] = (
    Arm(
        arm_id="direct.baseline",
        profile_name="direct",
        overrides={},
        description="Placeholder baseline arm.",
    ),
)

_BY_ID: dict[str, Arm] = {arm.arm_id: arm for arm in ARM_REGISTRY}


def get_arm(arm_id: str) -> Arm:
    """Return the arm with id ``arm_id`` or raise :class:`KeyError`."""
    if arm_id not in _BY_ID:
        msg = f"Unknown arm_id: {arm_id!r}"
        raise KeyError(msg)
    return _BY_ID[arm_id]


def list_arm_ids() -> list[str]:
    """Return all registered arm ids in registration order."""
    return [arm.arm_id for arm in ARM_REGISTRY]


__all__ = ["ARM_REGISTRY", "Arm", "get_arm", "list_arm_ids"]
