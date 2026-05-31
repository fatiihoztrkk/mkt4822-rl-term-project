"""Tests for shaped reward derivation from AgentRunResult."""

from __future__ import annotations

from mecha_agent_cli.agent.result import AgentRunResult
from mecha_agent_cli.config.schema import LearningConfig
from mecha_agent_cli.core.types import FailureType
from mecha_agent_cli.learning.reward import episode_reward
from mecha_agent_cli.validation.report import ValidationReport, ValidationResult


def _passed(name: str) -> ValidationResult:
    return ValidationResult(
        name=name,
        command=[],
        exit_code=0,
        stdout_excerpt="",
        stderr_excerpt="",
        passed=True,
        duration_sec=0.01,
        failure_type=FailureType.UNKNOWN,
    )


def _failed(name: str, ftype: FailureType) -> ValidationResult:
    return ValidationResult(
        name=name,
        command=[],
        exit_code=1,
        stdout_excerpt="",
        stderr_excerpt="boom",
        passed=False,
        duration_sec=0.01,
        failure_type=ftype,
    )


def _result(*, status: str, attempts: int, results: list[ValidationResult], primary: FailureType) -> AgentRunResult:
    report = ValidationReport(
        results=results,
        semantic_score=1.0 if primary is FailureType.UNKNOWN else 0.0,
        total_score=1.0 if primary is FailureType.UNKNOWN else 0.0,
        primary_failure=primary,
    )
    return AgentRunResult(
        task_id=1,
        status=status,
        changed_files=[],
        validation_report=report,
        review_summary="",
        total_attempts=attempts,
        max_attempts=10,
    )


def test_success_first_attempt_yields_full_reward() -> None:
    cfg = LearningConfig()
    result = _result(
        status="success",
        attempts=1,
        results=[_passed("syntax"), _passed("import"), _passed("runtime")],
        primary=FailureType.UNKNOWN,
    )
    assert episode_reward(result, cfg) == 1.0


def test_success_with_attempts_pays_attempt_penalty() -> None:
    cfg = LearningConfig(success_reward=1.0, attempt_penalty=0.05)
    result = _result(
        status="success",
        attempts=4,
        results=[_passed("syntax"), _passed("import"), _passed("runtime")],
        primary=FailureType.UNKNOWN,
    )
    assert abs(episode_reward(result, cfg) - (1.0 - 0.05 * 3)) < 1e-9


def test_failure_with_partial_progress_credits_passed_checks() -> None:
    cfg = LearningConfig(progress_bonus=0.15, attempt_penalty=0.05)
    # syntax and import passed, runtime failed
    result = _result(
        status="attempt_budget_exhausted",
        attempts=3,
        results=[_passed("syntax"), _passed("import"), _failed("runtime", FailureType.RUNTIME)],
        primary=FailureType.RUNTIME,
    )
    expected = 0.15 * 2 - 0.05 * 2
    assert abs(episode_reward(result, cfg) - expected) < 1e-9


def test_extract_failure_adds_extra_penalty() -> None:
    cfg = LearningConfig(extract_failure_penalty=0.10, attempt_penalty=0.05)
    result = _result(
        status="attempt_budget_exhausted",
        attempts=2,
        results=[_failed("extract", FailureType.SEMANTIC)],
        primary=FailureType.SEMANTIC,
    )
    expected = 0.0 - 0.05 * 1 - 0.10
    assert abs(episode_reward(result, cfg) - expected) < 1e-9


def test_reward_is_clamped_to_unit_interval() -> None:
    cfg = LearningConfig(attempt_penalty=1.0)
    bad = _result(
        status="attempt_budget_exhausted",
        attempts=50,
        results=[_failed("syntax", FailureType.SYNTAX)],
        primary=FailureType.SYNTAX,
    )
    assert episode_reward(bad, cfg) >= -1.0


def test_behavior_failure_adds_extra_penalty() -> None:
    cfg = LearningConfig(
        progress_bonus=0.15,
        attempt_penalty=0.05,
        behavior_failure_penalty=0.20,
    )
    result = _result(
        status="attempt_budget_exhausted",
        attempts=2,
        results=[
            _passed("syntax"),
            _passed("spec_contract"),
            _passed("import"),
            _passed("runtime"),
            _failed("behavior", FailureType.SEMANTIC),
        ],
        primary=FailureType.SEMANTIC,
    )
    expected = 0.15 * 4 - 0.05 * 1 - 0.20
    assert abs(episode_reward(result, cfg) - expected) < 1e-9
