"""Validation pipeline for generated ``algorithm.py`` candidates."""

from __future__ import annotations

from pathlib import Path

from mecha_agent_cli.config.schema import ValidationConfig
from mecha_agent_cli.core.types import FailureType
from mecha_agent_cli.sandbox.local_runner import LocalRunner
from mecha_agent_cli.validation.behavior_check import BehaviorCheck
from mecha_agent_cli.validation.import_check import ImportCheck
from mecha_agent_cli.validation.report import ValidationReport, ValidationResult
from mecha_agent_cli.validation.runtime_check import RuntimeCheck
from mecha_agent_cli.validation.spec_contract import SpecContractCheck
from mecha_agent_cli.validation.syntax_check import SyntaxCheck
from mecha_agent_cli.validation.undefined_name_check import UndefinedNameCheck


class ValidationPipeline:
    """Run syntax → undefined-name → spec-contract → import → runtime → behavior."""

    def __init__(
        self,
        config: ValidationConfig | None = None,
        runner: LocalRunner | None = None,
    ) -> None:
        self.config = config or ValidationConfig()
        self.runner = runner or LocalRunner()

    def run(self, repo_root: Path, *, user_request: str = "") -> ValidationReport:
        """Run validation checks in fail-fast order."""
        results: list[ValidationResult] = []
        syntax = SyntaxCheck(self.config.target_file, self.runner).run(repo_root)
        results.append(syntax)
        if not syntax.passed:
            return self._report(results)
        if self.config.run_pyright:
            undefined_name = UndefinedNameCheck(self.config.target_file).run(repo_root)
            results.append(undefined_name)
            if not undefined_name.passed and not undefined_name.skipped:
                return self._report(results)
        if self.config.run_spec_contract and user_request:
            contract = SpecContractCheck(
                target_file=self.config.target_file,
                user_request=user_request,
            ).run(repo_root)
            results.append(contract)
            if not contract.passed:
                return self._report(results)
        import_result = ImportCheck(self.runner, timeout_sec=self.config.import_timeout_sec).run(repo_root)
        results.append(import_result)
        if not import_result.passed:
            return self._report(results)
        if self.config.run_runtime_check:
            runtime_result = RuntimeCheck(self.runner, timeout_sec=self.config.runtime_timeout_sec).run(repo_root)
            results.append(runtime_result)
            if not runtime_result.passed:
                return self._report(results)
        if self.config.run_behavior_check and user_request:
            behavior_result = BehaviorCheck(
                user_request=user_request,
                runner=self.runner,
                timeout_sec=self.config.behavior_timeout_sec,
            ).run(repo_root)
            results.append(behavior_result)
            if not behavior_result.passed and not behavior_result.skipped:
                return self._report(results)
        return self._report(results)

    def _report(self, results: list[ValidationResult]) -> ValidationReport:
        primary = FailureType.UNKNOWN
        for result in results:
            if not result.passed and not result.skipped:
                primary = result.failure_type
                break
        passed = primary == FailureType.UNKNOWN
        total = 1.0 if passed else 0.0
        return ValidationReport(
            results=results,
            semantic_score=1.0 if passed else 0.0,
            total_score=total,
            primary_failure=primary,
        )
