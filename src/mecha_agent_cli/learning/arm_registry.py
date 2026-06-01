"""Model-profile action registry for the contextual bandit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mecha_agent_cli.config.schema import ModelProfile

_BOUNDED_DIRECT: dict[str, Any] = {
    "think": False,
    "num_ctx": 4096,
    "num_predict": 4096,
}


def _bounded(**overrides: Any) -> dict[str, Any]:
    """Return stable low-VRAM direct-generation options plus arm overrides."""
    return {**_BOUNDED_DIRECT, **overrides}


@dataclass(frozen=True)
class Arm:
    """One model-profile override the contextual bandit may select."""

    arm_id: str
    profile_name: str
    overrides: dict[str, Any] = field(default_factory=dict[str, Any])
    description: str = ""

    def apply(self, base: ModelProfile) -> ModelProfile:
        """Return a copied profile with this arm's validated overrides."""
        unknown = set(self.overrides) - set(type(base).model_fields)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown ModelProfile field(s): {names}")
        return base.model_copy(update=self.overrides)


ARM_REGISTRY: tuple[Arm, ...] = (
    Arm("direct.baseline", "direct", _bounded(), "Bounded low-VRAM direct-generation baseline."),
    Arm("direct.cool", "direct", _bounded(temperature=0.2, top_p=0.85), "Lower-temperature generation."),
    Arm("direct.cold", "direct", _bounded(temperature=0.1, top_p=0.75), "Highly conservative generation."),
    Arm("direct.warm", "direct", _bounded(temperature=0.6, top_p=0.92), "Moderately exploratory generation."),
    Arm("direct.hot", "direct", _bounded(temperature=0.8, top_p=0.95), "Exploratory generation."),
    Arm("direct.tight_topk", "direct", _bounded(top_k=20), "Restrict token sampling breadth."),
    Arm("direct.broad_topk", "direct", _bounded(top_k=80), "Expand token sampling breadth."),
    Arm("direct.high_repeat", "direct", _bounded(repeat_penalty=1.2), "Discourage repetitive output."),
    Arm("direct.no_think", "direct", _bounded(num_predict=3072), "Use a shorter bounded output budget."),
    Arm("direct.fixed_seed", "direct", _bounded(seed=42), "Use deterministic model sampling."),
)

_BY_ID: dict[str, Arm] = {arm.arm_id: arm for arm in ARM_REGISTRY}


def get_arm(arm_id: str) -> Arm:
    """Return the registered arm with ``arm_id``."""
    try:
        return _BY_ID[arm_id]
    except KeyError as exc:
        raise KeyError(f"Unknown arm_id: {arm_id!r}") from exc


def list_arm_ids() -> list[str]:
    """Return registered arm identifiers in selection order."""
    return [arm.arm_id for arm in ARM_REGISTRY]


__all__ = ["ARM_REGISTRY", "Arm", "get_arm", "list_arm_ids"]
