from pathlib import Path

from mecha_agent_cli.app.commands import init_command, run_command


def _disable_slow_validation(repo_root: Path) -> None:
    validation_config = (
        "target_file: algorithm.py\nmax_repairs: 3\nrun_pytest: false\nrun_ruff: false\nrun_pyright: false\n"
    )
    (repo_root / "configs" / "validation.yaml").write_text(validation_config)


def test_sort_numbers_golden(tmp_path: Path) -> None:
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    result = run_command(tmp_path, "Implement a stable merge sort", backend="fake")
    assert result.status == "success"
