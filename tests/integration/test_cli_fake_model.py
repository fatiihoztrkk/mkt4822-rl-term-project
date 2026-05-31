from pathlib import Path

from typer.testing import CliRunner

from mecha_agent_cli.app.cli import app
from mecha_agent_cli.app.commands import validate_command


def test_cli_init_and_models_list(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(app, ["init", str(tmp_path)])
    assert init_result.exit_code == 0
    assert (tmp_path / "algorithm.py").exists()
    models_result = runner.invoke(app, ["models", "list"])
    assert models_result.exit_code == 0
    assert "qwen3:4b" in models_result.stdout


def test_init_workspace_produces_clean_baseline(tmp_path: Path) -> None:
    """Bare ``mecha-agent init`` should leave a syntactically clean,
    importable, lint-clean baseline. The placeholder intentionally exposes
    no ``solve`` symbol, so the semantic checklist is not yet satisfied —
    the LLM is responsible for producing it on a subsequent ``run``.
    """

    runner = CliRunner()
    init_result = runner.invoke(app, ["init", str(tmp_path)])
    assert init_result.exit_code == 0

    report = validate_command(tmp_path)

    syntax = next(r for r in report.results if r.name == "syntax")
    assert syntax.exit_code == 0, syntax.stderr or syntax.stdout
    import_check = next(r for r in report.results if r.name == "import")
    assert import_check.exit_code == 0, import_check.stderr or import_check.stdout


def test_cli_judge_command_writes_report(tmp_path: Path) -> None:
    runner = CliRunner()
    init_result = runner.invoke(app, ["init", str(tmp_path)])
    assert init_result.exit_code == 0

    run_result = runner.invoke(
        app,
        [
            "run",
            "Implement sort",
            "--backend",
            "fake",
            "--path",
            str(tmp_path),
            "--max-attempts",
            "1",
        ],
    )
    assert run_result.exit_code == 0

    judge_result = runner.invoke(
        app,
        [
            "judge",
            "Implement sort",
            "--backend",
            "fake",
            "--path",
            str(tmp_path),
            "--out",
            ".mecha-agent/judge/test.json",
        ],
    )
    assert judge_result.exit_code == 0
    assert "report_path" in judge_result.stdout
    assert (tmp_path / ".mecha-agent" / "judge" / "test.json").exists()
