"""Unit tests for the contextual Thompson bandit."""

from __future__ import annotations

import random

from mecha_agent_cli.config.schema import LearningConfig, ModelProfile
from mecha_agent_cli.learning.arm_registry import ARM_REGISTRY, get_arm, list_arm_ids
from mecha_agent_cli.learning.bandit import BanditStore, ThompsonBandit
from mecha_agent_cli.learning.context import build_context_key, task_family


def test_arm_apply_does_not_mutate_base_profile() -> None:
    base = ModelProfile(temperature=0.4, top_p=0.9)
    arm = get_arm("direct.cool")
    out = arm.apply(base)
    assert base.temperature == 0.4
    assert out.temperature == 0.2
    assert out.top_p == 0.85


def test_arm_apply_rejects_unknown_field() -> None:
    from mecha_agent_cli.learning.arm_registry import Arm

    bad = Arm(arm_id="bad", profile_name="direct", overrides={"not_a_field": 1})
    try:
        bad.apply(ModelProfile())
    except ValueError as exc:
        assert "not_a_field" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_all_arms_in_registry_apply_cleanly() -> None:
    base = ModelProfile()
    seen = set()
    for arm in ARM_REGISTRY:
        out = arm.apply(base)
        assert isinstance(out, ModelProfile)
        assert arm.arm_id not in seen
        seen.add(arm.arm_id)
    assert "direct.baseline" in seen


def test_context_key_is_deterministic_and_lowercase() -> None:
    a = build_context_key(user_request="Sort the list please", model_name="qwen3:4b", has_baseline=False)
    b = build_context_key(user_request="sort the list please", model_name="qwen3:4b", has_baseline=False)
    assert a == b
    assert a == "sort|qwen3:4b|new"


def test_task_family_matches_known_families() -> None:
    assert task_family("Compute fibonacci(20)") == "fibonacci_dp"
    assert task_family("Implement merge sort") == "sort"
    assert task_family("Write a moving average filter") == "signal_filter"
    assert task_family("Solve linear system numerically") == "numerical"
    assert task_family("Random unrelated request") == "generic"


def test_bandit_off_mode_returns_baseline() -> None:
    cfg = LearningConfig(enabled=True, mode="off", persist=False)
    bandit = ThompsonBandit(BanditStore(None), cfg, rng=random.Random(0))
    arm = bandit.select("anything|qwen3:4b|new")
    assert arm.arm_id == "direct.baseline"


def test_bandit_force_pulls_under_pulled_arms() -> None:
    cfg = LearningConfig(enabled=True, mode="thompson", min_pulls_before_exploit=2, persist=False)
    bandit = ThompsonBandit(BanditStore(None), cfg, rng=random.Random(123))
    ctx = "sort|qwen3:4b|new"
    seen: set[str] = set()
    # 2 * len(arms) selections with rewards must touch every arm at least once
    # because the under-pulled set is non-empty until all arms reach min_pulls.
    n = 2 * len(ARM_REGISTRY)
    for _ in range(n):
        arm = bandit.select(ctx)
        seen.add(arm.arm_id)
        bandit.update(context_key=ctx, arm_id=arm.arm_id, reward=0.0, success=False)
    assert seen == set(list_arm_ids())


def test_thompson_converges_to_dominant_arm() -> None:
    """Synthetic two-arm MDP: arm A has p_success=0.9, arm B has p_success=0.1.

    After enough episodes Thompson sampling should pull A far more often than B.
    """
    cfg = LearningConfig(enabled=True, mode="thompson", min_pulls_before_exploit=1, persist=False)
    rng = random.Random(7)
    bandit = ThompsonBandit(BanditStore(None), cfg, rng=rng)
    ctx = "sort|qwen3:4b|new"
    pulls = {"direct.baseline": 0, "direct.cool": 0}
    for _ in range(200):
        arm = bandit.select(ctx)
        # Restrict to the two arms we are scoring; ignore others by NOT updating them.
        if arm.arm_id not in pulls:
            # treat as neutral exploration noise; tiny update so they don't dominate
            bandit.update(context_key=ctx, arm_id=arm.arm_id, reward=0.0, success=False)
            continue
        # arm A = baseline (good, p=0.9), arm B = cool (bad, p=0.1)
        success = rng.random() < (0.9 if arm.arm_id == "direct.baseline" else 0.1)
        reward = 1.0 if success else -1.0
        bandit.update(context_key=ctx, arm_id=arm.arm_id, reward=reward, success=success)
        pulls[arm.arm_id] += 1
    assert pulls["direct.baseline"] > pulls["direct.cool"] * 2


def test_bandit_persists_across_store_instances(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = LearningConfig(enabled=True, mode="thompson", min_pulls_before_exploit=0, persist=True)
    db = tmp_path / "rl.sqlite"
    a = ThompsonBandit(BanditStore(db), cfg, rng=random.Random(0))
    a.update(context_key="x|qwen3:4b|new", arm_id="direct.baseline", reward=1.0, success=True)
    a.update(context_key="x|qwen3:4b|new", arm_id="direct.baseline", reward=1.0, success=True)
    # Re-open, must see the prior pulls
    b = ThompsonBandit(BanditStore(db), cfg, rng=random.Random(0))
    rows = b.store.fetch("x|qwen3:4b|new")
    assert "direct.baseline" in rows
    assert rows["direct.baseline"].pulls == 2
    assert rows["direct.baseline"].alpha > 1.0


def test_reward_to_beta_update_clamps() -> None:
    from mecha_agent_cli.learning.reward import reward_to_beta_update

    assert reward_to_beta_update(-1.0) == 0.0
    assert reward_to_beta_update(0.0) == 0.5
    assert reward_to_beta_update(1.0) == 1.0
    assert reward_to_beta_update(-2.0) == 0.0
    assert reward_to_beta_update(2.0) == 1.0


def test_decay_factor_shrinks_prior_evidence() -> None:
    """With decay_factor < 1, repeated negative updates revert toward Beta(1,1)."""
    cfg = LearningConfig(
        enabled=True,
        mode="thompson",
        min_pulls_before_exploit=0,
        decay_factor=0.5,
        persist=False,
    )
    bandit = ThompsonBandit(BanditStore(None), cfg, rng=random.Random(0))
    ctx = "sort|qwen3:4b|new"
    # Build alpha up to its stationary point (1 + 1/(1-0.5) = 3 for p=1).
    for _ in range(15):
        bandit.update(context_key=ctx, arm_id="direct.baseline", reward=1.0, success=True)
    rows_after_pos = bandit.store.fetch(ctx)
    alpha_high = rows_after_pos["direct.baseline"].alpha
    assert alpha_high > 2.5  # converges to ~3.0 under decay=0.5
    # Now feed strong negatives — alpha must shrink toward 1.0.
    for _ in range(15):
        bandit.update(context_key=ctx, arm_id="direct.baseline", reward=-1.0, success=False)
    rows_after_neg = bandit.store.fetch(ctx)
    assert rows_after_neg["direct.baseline"].alpha < alpha_high
    # Compare to the no-decay reference: alpha would be ~16 under exact arithmetic.
    cfg_nodecay = LearningConfig(enabled=True, mode="thompson", min_pulls_before_exploit=0, persist=False)
    ref = ThompsonBandit(BanditStore(None), cfg_nodecay, rng=random.Random(0))
    for _ in range(15):
        ref.update(context_key=ctx, arm_id="direct.baseline", reward=1.0, success=True)
    assert ref.store.fetch(ctx)["direct.baseline"].alpha > alpha_high * 2


def test_decay_factor_default_is_lossless() -> None:
    """decay_factor == 1.0 must preserve exact Beta-Bernoulli arithmetic."""
    cfg = LearningConfig(enabled=True, mode="thompson", min_pulls_before_exploit=0, persist=False)
    assert cfg.decay_factor == 1.0
    bandit = ThompsonBandit(BanditStore(None), cfg, rng=random.Random(0))
    ctx = "sort|qwen3:4b|new"
    for _ in range(5):
        bandit.update(context_key=ctx, arm_id="direct.baseline", reward=1.0, success=True)
    stat = bandit.store.fetch(ctx)["direct.baseline"]
    # 5 successes at p=1.0 → alpha = 1 + 5*1 = 6, beta = 1 + 5*0 = 1
    assert stat.alpha == 6.0
    assert stat.beta == 1.0


def test_ucb1_mode_picks_high_mean_arm() -> None:
    """UCB1 with low ucb_c and saturated pulls must exploit the best-mean arm."""
    cfg = LearningConfig(
        enabled=True,
        mode="ucb1",
        ucb_c=0.0,  # turn off exploration bonus → pure greedy on posterior mean
        min_pulls_before_exploit=1,
        persist=False,
    )
    bandit = ThompsonBandit(BanditStore(None), cfg, rng=random.Random(0))
    ctx = "sort|qwen3:4b|new"
    # Seed arms: baseline gets many wins; cool gets many losses; rest one neutral.
    for _ in range(20):
        bandit.update(context_key=ctx, arm_id="direct.baseline", reward=1.0, success=True)
    for _ in range(20):
        bandit.update(context_key=ctx, arm_id="direct.cool", reward=-1.0, success=False)
    for arm_id in (
        "direct.cold",
        "direct.warm",
        "direct.hot",
        "direct.tight_topk",
        "direct.broad_topk",
        "direct.high_repeat",
        "direct.no_think",
        "direct.fixed_seed",
    ):
        bandit.update(context_key=ctx, arm_id=arm_id, reward=0.0, success=False)
    chosen = [bandit.select(ctx).arm_id for _ in range(20)]
    assert chosen.count("direct.baseline") >= 18  # near-deterministic with c=0


def test_latency_penalty_reduces_reward_for_slow_episodes() -> None:
    """Long durations must reduce reward below the no-latency baseline."""
    from mecha_agent_cli.agent.result import AgentRunResult
    from mecha_agent_cli.core.types import FailureType
    from mecha_agent_cli.learning.reward import episode_reward
    from mecha_agent_cli.validation.report import ValidationReport, ValidationResult

    report = ValidationReport(
        results=[
            ValidationResult(
                name="syntax",
                command=[],
                exit_code=0,
                stdout_excerpt="",
                stderr_excerpt="",
                passed=True,
                duration_sec=0.0,
                failure_type=FailureType.UNKNOWN,
            )
        ],
        semantic_score=1.0,
        total_score=1.0,
        primary_failure=FailureType.UNKNOWN,
    )
    fast = AgentRunResult(
        task_id=1,
        status="success",
        changed_files=[],
        validation_report=report,
        review_summary="",
        total_attempts=1,
        duration_sec=1.0,
    )
    slow = AgentRunResult(
        task_id=2,
        status="success",
        changed_files=[],
        validation_report=report,
        review_summary="",
        total_attempts=1,
        duration_sec=300.0,  # past the 120s horizon → full penalty
    )
    cfg_no_latency = LearningConfig(latency_penalty=0.0)
    cfg_latency = LearningConfig(latency_penalty=0.3, latency_horizon_sec=120.0)
    # Without latency shaping both episodes are identical.
    assert episode_reward(fast, cfg_no_latency) == episode_reward(slow, cfg_no_latency)
    # With latency shaping the slow episode must score strictly lower.
    assert episode_reward(slow, cfg_latency) < episode_reward(fast, cfg_latency)
    # Penalty caps at latency_penalty even past the horizon.
    expected_diff = cfg_latency.latency_penalty * (1.0 - 1.0 / cfg_latency.latency_horizon_sec)
    assert abs((episode_reward(fast, cfg_latency) - episode_reward(slow, cfg_latency)) - expected_diff) < 1e-9
