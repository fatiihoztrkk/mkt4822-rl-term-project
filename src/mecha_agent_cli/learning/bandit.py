"""Placeholder Thompson module for course starter repositories.

This file preserves imports and method signatures while avoiding full
algorithmic implementation details.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from mecha_agent_cli.config.schema import LearningConfig
from mecha_agent_cli.learning.arm_registry import ARM_REGISTRY, Arm, get_arm


@dataclass(frozen=True)
class ArmStat:
    """Minimal bookkeeping for one ``(context, arm)`` pair."""

    context_key: str
    arm_id: str
    alpha: float
    beta: float
    pulls: int
    cumulative_reward: float
    last_reward: float
    last_success: bool

    @property
    def mean(self) -> float:
        """Return a stable pseudo-mean for placeholder mode."""
        return self.alpha / (self.alpha + self.beta)


class BanditStore:
    """In-memory placeholder storage facade.

    ``db_path`` is retained for API compatibility only.
    """

    def __init__(self, db_path: Path | None) -> None:
        self.db_path = db_path
        self._rows: dict[tuple[str, str], ArmStat] = {}

    def fetch(self, context_key: str) -> dict[str, ArmStat]:
        """Return placeholder rows for ``context_key`` keyed by ``arm_id``."""
        out: dict[str, ArmStat] = {}
        for (ctx, arm_id), stat in self._rows.items():
            if ctx == context_key:
                out[arm_id] = stat
        return out

    def upsert(self, stat: ArmStat) -> None:
        """Insert or update one placeholder row."""
        self._rows[(stat.context_key, stat.arm_id)] = stat

    def all_rows(self) -> list[ArmStat]:
        """Return all placeholder rows."""
        return list(self._rows.values())


class ThompsonBandit:
    """Placeholder Thompson controller.

    Selection returns baseline arm; updates only maintain simple counters.
    """

    BASELINE_ARM_ID = "direct.baseline"

    def __init__(
        self,
        store: BanditStore,
        cfg: LearningConfig,
        *,
        arms: tuple[Arm, ...] = ARM_REGISTRY,
        rng: random.Random | None = None,
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.arms = arms
        self._arm_ids = tuple(a.arm_id for a in arms)
        self.rng = rng or random.Random()

    # -- selection -------------------------------------------------------

    def select(self, context_key: str) -> Arm:
        """Return baseline arm in placeholder mode."""
        del context_key
        return get_arm(self.BASELINE_ARM_ID)

    # -- update ----------------------------------------------------------

    def update(self, *, context_key: str, arm_id: str, reward: float, success: bool) -> ArmStat:
        """Record a minimal placeholder update and return stat."""
        prior = self.store.fetch(context_key).get(arm_id, _zero_stat(context_key, arm_id))
        new = ArmStat(
            context_key=context_key,
            arm_id=arm_id,
            alpha=prior.alpha,
            beta=prior.beta,
            pulls=prior.pulls + 1,
            cumulative_reward=prior.cumulative_reward + reward,
            last_reward=reward,
            last_success=success,
        )
        self.store.upsert(new)
        return new


def _zero_stat(context_key: str, arm_id: str) -> ArmStat:
    """Return a default stat for an unseen ``(context_key, arm_id)``."""
    return ArmStat(
        context_key=context_key,
        arm_id=arm_id,
        alpha=1.0,
        beta=1.0,
        pulls=0,
        cumulative_reward=0.0,
        last_reward=0.0,
        last_success=False,
    )


__all__ = ["ArmStat", "BanditStore", "ThompsonBandit"]
