"""Reward shaping for contextual model-profile selection."""

from __future__ import annotations

from mecha_agent_cli.agent.result import AgentRunResult
from mecha_agent_cli.config.schema import LearningConfig


def episode_reward(result: AgentRunResult, cfg: LearningConfig) -> float:
    """Return shaped episode reward in ``[-1.0, 1.0]``."""
    success = result.status == "success"
    reward = cfg.success_reward if success else 0.0
    reward -= max(0, result.total_attempts - 1) * cfg.attempt_penalty
    if cfg.latency_penalty > 0.0 and cfg.latency_horizon_sec > 0.0:
        reward -= cfg.latency_penalty * min(1.0, result.duration_sec / cfg.latency_horizon_sec)

    if not success:
        reward += sum(1 for check in result.validation_report.results if check.passed) * cfg.progress_bonus
        failed_names = {check.name for check in result.validation_report.results if not check.passed}
        if "extract" in failed_names:
            reward -= cfg.extract_failure_penalty
        if "behavior" in failed_names:
            reward -= cfg.behavior_failure_penalty
    return max(-1.0, min(1.0, reward))


def reward_to_beta_update(reward: float) -> float:
    """Map reward from ``[-1, 1]`` to pseudo-success weight in ``[0, 1]``.

    Uses a cubic s-curve: -1→0, 0→0.5, 1→1 exactly (boundary-safe).
    The nonlinear shape amplifies strong signals more than a linear map.
    """
    clamped = max(-1.0, min(1.0, float(reward)))
    scaled = (3.0 * clamped - clamped**3) / 2.0
    return (scaled + 1.0) / 2.0


__all__ = ["episode_reward", "reward_to_beta_update"]
