from pathlib import Path

from mecha_agent_cli.app.commands import init_command, run_command


def _disable_slow_validation(repo_root: Path) -> None:
    validation_config = (
        "target_file: algorithm.py\nmax_repairs: 3\nrun_pytest: false\nrun_ruff: false\nrun_pyright: false\n"
    )
    (repo_root / "configs" / "validation.yaml").write_text(validation_config)


def test_run_with_fake_model_writes_algorithm(tmp_path: Path) -> None:
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    result = run_command(tmp_path, "Implement a stable merge sort in algorithm.py", backend="fake")
    assert result.changed_files == ["algorithm.py"]
    assert result.attempt_snapshots
    assert (tmp_path / result.attempt_snapshots[-1]).exists()
    assert "merge sort" in (tmp_path / "algorithm.py").read_text().lower()
    syntax = result.validation_report.by_name("syntax")
    assert syntax is not None and syntax.passed
