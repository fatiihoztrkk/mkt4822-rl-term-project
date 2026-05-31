"""Placeholder reward helpers for course starter repositories.

Students can replace these stubs with assignment-specific reward shaping.
"""

from __future__ import annotations

from mecha_agent_cli.agent.result import AgentRunResult
from mecha_agent_cli.config.schema import LearningConfig


def episode_reward(result: AgentRunResult, cfg: LearningConfig) -> float:
    """Return a minimal placeholder reward in ``[-1.0, 1.0]``."""
    del cfg
    return 1.0 if result.status == "success" else -0.1


def reward_to_beta_update(reward: float) -> float:
    """Map reward in ``[-1, 1]`` to a pseudo-success in ``[0, 1]``."""
    return max(0.0, min(1.0, (reward + 1.0) / 2.0))


__all__ = ["episode_reward", "reward_to_beta_update"]
