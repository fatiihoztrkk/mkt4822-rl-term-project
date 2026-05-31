"""Import smoke validation."""

from __future__ import annotations

from pathlib import Path

from mecha_agent_cli.core.types import FailureType
from mecha_agent_cli.sandbox.local_runner import LocalRunner
from mecha_agent_cli.validation.base import ValidationCheck
from mecha_agent_cli.validation.report import ValidationResult

IMPORT_SNIPPET = (
    "import importlib.util; "
    "spec=importlib.util.spec_from_file_location('algorithm','algorithm.py'); "
    "mod=importlib.util.module_from_spec(spec); "
    "assert spec.loader is not None; "
    "spec.loader.exec_module(mod); print('ok')"
)


class ImportCheck(ValidationCheck):
    """Run an isolated import smoke check using importlib file loading."""

    name = "import"

    def __init__(self, runner: LocalRunner | None = None, timeout_sec: float = 30.0) -> None:
        self.runner = runner or LocalRunner()
        self.timeout_sec = timeout_sec

    def run(self, repo_root: Path) -> ValidationResult:
        """Run import smoke validation."""
        command = ["python", "-I", "-c", IMPORT_SNIPPET]
        result = self.runner.run(command, repo_root, timeout_sec=self.timeout_sec)
        return ValidationResult(
            name=self.name,
            command=command,
            exit_code=result.exit_code,
            stdout_excerpt=result.stdout,
            stderr_excerpt=result.stderr,
            passed=result.passed,
            duration_sec=result.duration_sec,
            failure_type=FailureType.IMPORT if not result.passed else FailureType.UNKNOWN,
        )
