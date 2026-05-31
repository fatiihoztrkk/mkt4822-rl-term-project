"""Tests for the simplified syntax+import validation pipeline."""

from __future__ import annotations

from pathlib import Path

from mecha_agent_cli.sandbox.local_runner import LocalRunner
from mecha_agent_cli.validation.pipeline import ValidationPipeline


def test_pipeline_catches_syntax_failure(tmp_path: Path) -> None:
    (tmp_path / "algorithm.py").write_text("def solve(:\n")
    report = ValidationPipeline().run(tmp_path)
    assert report.primary_failure.value == "SYNTAX"
    assert not report.passed


def test_pipeline_passes_basic_module(tmp_path: Path) -> None:
    source = "def solve(data: list[int]) -> list[int]:\n    return list(data)\n"
    (tmp_path / "algorithm.py").write_text(source)
    report = ValidationPipeline().run(tmp_path)
    assert report.by_name("syntax") is not None
    assert report.by_name("import") is not None
    assert report.passed


def test_pipeline_fails_behavioral_criterion_when_prompt_demands_numeric_threshold(tmp_path: Path) -> None:
    source = (
        "def solve(n, alpha, gamma, epsilon, seed):\n"
        "    q = [[0.0, 0.0] for _ in range(5)]\n"
        "    q[3][1] = 0.1\n"
        "    return q\n"
    )
    (tmp_path / "algorithm.py").write_text(source)
    prompt = "Q[3][1] must be >= 0.5 after solve(2000, 0.5, 0.95, 0.1, 0)."
    report = ValidationPipeline().run(tmp_path, user_request=prompt)
    behavior = report.by_name("behavior")
    assert behavior is not None
    assert not behavior.passed
    assert report.primary_failure.value == "SEMANTIC"


def test_pipeline_allows_torch_import_signature(tmp_path: Path) -> None:
    """The model is free to import torch / numpy / anything: no forbidden-import gate."""
    source = (
        "from __future__ import annotations\n\n"
        "def solve(data: list[float]) -> list[float]:\n"
        '    """Echo data; signature is allowed to look like a torch tensor pipeline."""\n'
        "    return [float(x) for x in data]\n"
    )
    (tmp_path / "algorithm.py").write_text(source)
    report = ValidationPipeline().run(tmp_path)
    assert report.passed


def test_local_runner_uses_isolated_stable_subprocess_commands() -> None:
    runner = LocalRunner()
    import_command = runner._execution_command(["python", "-I", "-c", "import algorithm"])
    pytest_command = runner._execution_command(["pytest", "-q"])

    assert "-S" in import_command
    assert "-B" in import_command
    assert "-S" in pytest_command
    assert "-B" in pytest_command
    assert "-I" not in import_command
    assert "-I" not in pytest_command
