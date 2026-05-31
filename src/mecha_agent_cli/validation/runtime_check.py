"""Runtime smoke validation: execute the target file as ``__main__``.

The import-check only loads the module; module-level code runs but anything
guarded by ``if __name__ == "__main__":`` does not. A surprising amount of
generated code crashes only inside ``main()`` with shape mismatches, division
by zero, ``IndexError``, etc. ``RuntimeCheck`` runs the file with
``runpy.run_path(..., run_name="__main__")`` so those failures surface and
get fed back to the model.
"""

from __future__ import annotations

from pathlib import Path

from mecha_agent_cli.core.types import FailureType
from mecha_agent_cli.sandbox.local_runner import LocalRunner
from mecha_agent_cli.validation.base import ValidationCheck
from mecha_agent_cli.validation.report import ValidationResult

RUNTIME_SNIPPET = "import runpy, sys; sys.argv=['algorithm.py']; runpy.run_path('algorithm.py', run_name='__main__')"


class RuntimeCheck(ValidationCheck):
    """Execute ``algorithm.py`` as ``__main__`` and capture failures."""

    name = "runtime"

    def __init__(self, runner: LocalRunner | None = None, timeout_sec: float = 60.0) -> None:
        self.runner = runner or LocalRunner()
        self.timeout_sec = timeout_sec

    def run(self, repo_root: Path) -> ValidationResult:
        """Run the file and treat any non-zero exit as a runtime failure."""
        command = ["python", "-I", "-c", RUNTIME_SNIPPET]
        result = self.runner.run(command, repo_root, timeout_sec=self.timeout_sec)
        return ValidationResult(
            name=self.name,
            command=command,
            exit_code=result.exit_code,
            stdout_excerpt=result.stdout,
            stderr_excerpt=result.stderr,
            passed=result.passed,
            duration_sec=result.duration_sec,
            failure_type=FailureType.RUNTIME if not result.passed else FailureType.UNKNOWN,
        )
