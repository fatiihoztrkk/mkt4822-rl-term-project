from pathlib import Path

from mecha_agent_cli.agent.observer import AgentProgressEvent
from mecha_agent_cli.app.commands import init_command, run_command


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[AgentProgressEvent] = []

    def __call__(self, event: AgentProgressEvent) -> None:
        self.events.append(event)


def _disable_slow_validation(repo_root: Path) -> None:
    validation_config = (
        "target_file: algorithm.py\nmax_repairs: 3\nrun_pytest: false\nrun_ruff: false\nrun_pyright: false\n"
    )
    (repo_root / "configs" / "validation.yaml").write_text(validation_config)


def test_continuous_loop_repairs_and_reports_progress(tmp_path: Path) -> None:
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    observer = RecordingObserver()

    result = run_command(
        tmp_path,
        "Implement a stable merge sort in algorithm.py",
        backend="fake",
        scenario="syntax_then_repair",
        max_attempts=5,
        observer=observer,
    )

    assert result.status == "success"
    assert result.total_attempts >= 2
    assert result.attempt_snapshots
    stages = [event.stage for event in observer.events]
    assert "generate" in stages
    assert "regenerate" in stages
    assert "validate" in stages
    assert "merge sort" in (tmp_path / "algorithm.py").read_text().lower()


def test_no_code_block_first_recovers_on_next_attempt(tmp_path: Path) -> None:
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    observer = RecordingObserver()

    result = run_command(
        tmp_path,
        "Implement a stable merge sort in algorithm.py",
        backend="fake",
        scenario="no_block_first",
        max_attempts=4,
        observer=observer,
    )

    assert result.status == "success"
    assert result.total_attempts >= 2
    assert any(event.stage == "extract" and event.status == "fail" for event in observer.events)
    assert "merge sort" in (tmp_path / "algorithm.py").read_text().lower()
