"""Python syntax validation."""

from __future__ import annotations

from pathlib import Path

from mecha_agent_cli.core.types import FailureType
from mecha_agent_cli.sandbox.local_runner import LocalRunner
from mecha_agent_cli.validation.base import ValidationCheck
from mecha_agent_cli.validation.report import ValidationResult


class SyntaxCheck(ValidationCheck):
    """Run a bytecode-free syntax check on the target file."""

    name = "syntax"

    def __init__(self, target_file: str = "algorithm.py", runner: LocalRunner | None = None) -> None:
        self.target_file = target_file
        self.runner = runner or LocalRunner()

    def run(self, repo_root: Path) -> ValidationResult:
        """Run syntax validation without creating ``__pycache__`` files."""
        path = repo_root / self.target_file
        command = ["python", "-m", "py_compile", self.target_file]
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            return ValidationResult(
                name=self.name,
                command=command,
                exit_code=1,
                stdout_excerpt="",
                stderr_excerpt=f"{exc.__class__.__name__}: {exc}",
                passed=False,
                duration_sec=0.0,
                failure_type=FailureType.SYNTAX,
            )
        except OSError as exc:
            return ValidationResult(
                name=self.name,
                command=command,
                exit_code=1,
                stdout_excerpt="",
                stderr_excerpt=str(exc),
                passed=False,
                duration_sec=0.0,
                failure_type=FailureType.SYNTAX,
            )
        return ValidationResult(
            name=self.name,
            command=command,
            exit_code=0,
            stdout_excerpt="syntax ok",
            stderr_excerpt="",
            passed=True,
            duration_sec=0.0,
        )
