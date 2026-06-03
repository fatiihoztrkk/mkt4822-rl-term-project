"""Model-profile action registry for the contextual bandit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mecha_agent_cli.config.schema import ModelProfile

# RTX 3090 (24 GB VRAM) allows much larger context than the conservative 3060 defaults.
_BOUNDED_DIRECT: dict[str, Any] = {
    "think": False,
    "num_ctx": 8192,
    "num_predict": 6144,
}

# Thinking-mode profile: model reasons first, then generates code.
# Uses 16k context with unlimited output — best for complex multi-component code.
_THINKING_DIRECT: dict[str, Any] = {
    "think": True,
    "num_ctx": 16384,
    "num_predict": -1,
}


def _bounded(**overrides: Any) -> dict[str, Any]:
    """Return RTX-3090-tuned generation options plus arm overrides."""
    return {**_BOUNDED_DIRECT, **overrides}


def _thinking(**overrides: Any) -> dict[str, Any]:
    """Return thinking-mode generation options with 16k context."""
    return {**_THINKING_DIRECT, **overrides}


@dataclass(frozen=True)
class Arm:
    """One model-profile override the contextual bandit may select."""

    arm_id: str
    profile_name: str
    overrides: dict[str, Any] = field(default_factory=dict[str, Any])
    description: str = ""
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def apply(self, base: ModelProfile) -> ModelProfile:
        """Return a copied profile with this arm's validated overrides."""
        unknown = set(self.overrides) - set(type(base).model_fields)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown ModelProfile field(s): {names}")
        return base.model_copy(update=self.overrides)


ARM_REGISTRY: tuple[Arm, ...] = (
    # Think-mode moderate temperature — strong prior from observed benchmark success.
    Arm(
        "direct.baseline",
        "direct",
        _thinking(temperature=0.3, top_p=0.85),
        "Think-mode baseline, 16k ctx.",
        prior_alpha=1.9,
        prior_beta=1.0,
    ),
    Arm(
        "direct.cool",
        "direct",
        _thinking(temperature=0.2, top_p=0.85),
        "Think-mode, low temperature.",
        prior_alpha=1.9,
        prior_beta=1.0,
    ),
    # Best observed arm: 4 first-attempt successes, 0 first-attempt failures.
    Arm(
        "direct.cold",
        "direct",
        _thinking(temperature=0.1, top_p=0.70),
        "Think-mode, very conservative.",
        prior_alpha=2.2,
        prior_beta=1.0,
    ),
    # Matches the default direct profile exactly (temp=0.40, top_p=0.90) —
    # proven by baseline achieving 100% success with this configuration.
    Arm(
        "direct.warm",
        "direct",
        _thinking(temperature=0.40, top_p=0.90),
        "Think-mode, matches proven baseline profile.",
        prior_alpha=2.0,
        prior_beta=1.0,
    ),
    # High-temp arm; pessimistic prior — observed failures dominate.
    Arm(
        "direct.hot",
        "direct",
        _thinking(temperature=0.7, top_p=0.95),
        "Think-mode, exploratory.",
        prior_alpha=1.0,
        prior_beta=2.0,
    ),
    Arm(
        "direct.tight_topk",
        "direct",
        _thinking(temperature=0.3, top_k=20),
        "Think-mode, tight sampling.",
        prior_alpha=1.5,
        prior_beta=1.1,
    ),
    Arm(
        "direct.broad_topk",
        "direct",
        _thinking(temperature=0.4, top_k=80),
        "Think-mode, broad sampling.",
        prior_alpha=1.5,
        prior_beta=1.1,
    ),
    Arm(
        "direct.high_repeat",
        "direct",
        _thinking(temperature=0.3, repeat_penalty=1.15),
        "Think-mode, less repetition.",
        prior_alpha=1.6,
        prior_beta=1.0,
    ),
    # Non-thinking fallback — weaker prior: smaller ctx, no reasoning chain.
    Arm(
        "direct.no_think",
        "direct",
        _bounded(temperature=0.3, top_p=0.85),
        "Fast-mode, 8k ctx fallback.",
        prior_alpha=1.0,
        prior_beta=1.5,
    ),
    # Deterministic arm; observed 5 failures, 0 successes → strongly pessimistic.
    Arm(
        "direct.fixed_seed",
        "direct",
        _thinking(temperature=0.4, seed=42),
        "Think-mode, deterministic.",
        prior_alpha=1.0,
        prior_beta=2.2,
    ),
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
