"""Tabular Q-learning policy for model-profile selection.

The coding-agent state is a compact task-context key and each action is a
registered model-profile arm. Rewards come from validator outcomes. This module
does not contain generated-algorithm solutions or task-specific prompt code.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclasses_field


@dataclass(frozen=True)
class QValue:
    """Stored tabular Q estimate for one context and profile arm."""

    context_key: str
    arm_id: str
    value: float = 0.0
    updates: int = 0


class QLearningPolicy:
    """Choose profile arms with epsilon-greedy tabular Q-learning."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.25,
        discount_factor: float = 0.90,
        epsilon: float = 0.10,
        rng: random.Random | None = None,
    ) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("discount_factor must be in [0, 1]")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.rng = rng or random.Random()

    def select(self, arm_ids: Sequence[str], values: Mapping[str, QValue]) -> str:
        """Return an arm identifier using epsilon-greedy action selection."""
        if not arm_ids:
            raise ValueError("arm_ids must not be empty")
        if self.rng.random() < self.epsilon:
            return self.rng.choice(arm_ids)
        best_value = max(values.get(arm_id, QValue("", arm_id)).value for arm_id in arm_ids)
        best_arms = [arm_id for arm_id in arm_ids if values.get(arm_id, QValue("", arm_id)).value == best_value]
        return self.rng.choice(best_arms)

    def update(
        self,
        *,
        context_key: str,
        arm_id: str,
        reward: float,
        prior: QValue | None = None,
        next_best_value: float = 0.0,
    ) -> QValue:
        """Apply one tabular Q-learning update and return the new estimate."""
        previous = prior or QValue(context_key=context_key, arm_id=arm_id)
        target = reward + self.discount_factor * next_best_value
        value = previous.value + self.learning_rate * (target - previous.value)
        return QValue(
            context_key=context_key,
            arm_id=arm_id,
            value=value,
            updates=previous.updates + 1,
        )


@dataclass
class QLearner:
    """Convenience Q-table agent with internal state management.

    Wraps QLearningPolicy and maintains the full (state, action) → QValue
    table internally so callers only need state/action strings.
    """

    alpha: float = 0.25
    gamma: float = 0.90
    epsilon: float = 0.10
    _table: dict[tuple[str, str], QValue] = dataclasses_field(default_factory=dict, repr=False)
    _policy: QLearningPolicy = dataclasses_field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._policy = QLearningPolicy(
            learning_rate=self.alpha,
            discount_factor=self.gamma,
            epsilon=self.epsilon,
        )

    def q_value(self, state: str, action: str) -> float:
        """Return the current Q-value for a (state, action) pair."""
        return self._table.get((state, action), QValue(state, action)).value

    def select(self, state: str, actions: list[str], rng: random.Random | None = None) -> str:
        """Epsilon-greedy arm selection for ``state``."""
        if not actions:
            raise ValueError("actions must not be empty")
        if rng is not None:
            self._policy.rng = rng
        values = {a: self._table.get((state, a), QValue(state, a)) for a in actions}
        return self._policy.select(actions, values)

    def update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        next_actions: list[str],
    ) -> float:
        """Apply Q-learning update and return the new Q-value."""
        prior = self._table.get((state, action))
        next_best = max((self.q_value(next_state, a) for a in next_actions), default=0.0)
        new_qv = self._policy.update(
            context_key=state,
            arm_id=action,
            reward=reward,
            prior=prior,
            next_best_value=next_best,
        )
        self._table[(state, action)] = new_qv
        return new_qv.value

    def reset(self) -> None:
        """Clear all learned Q-values."""
        self._table.clear()


# ---------------------------------------------------------------------------
# Backward-compatible placeholder exports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QLearningPlaceholder:
    """Import-safe marker retained for backward compatibility."""

    name: str = "q_learner"
    status: str = "implemented"


def placeholder_status() -> str:
    """Return module status string."""
    return "q_learning module: QLearner + QLearningPolicy tabular Q-learning implemented"


__all__ = ["QLearner", "QLearningPlaceholder", "QLearningPolicy", "QValue", "placeholder_status"]
