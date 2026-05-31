"""Validation report models."""

from __future__ import annotations

from dataclasses import dataclass, field

from mecha_agent_cli.core.types import FailureType, ValidationStatus


@dataclass(frozen=True)
class ValidationResult:
    """One validation command/check result."""

    name: str
    command: list[str]
    exit_code: int
    stdout_excerpt: str
    stderr_excerpt: str
    passed: bool
    duration_sec: float
    failure_type: FailureType = FailureType.UNKNOWN
    skipped: bool = False

    @property
    def status(self) -> ValidationStatus:
        """Return pass/fail/skip status."""
        if self.skipped:
            return ValidationStatus.SKIP
        return ValidationStatus.PASS if self.passed else ValidationStatus.FAIL


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate validation report."""

    results: list[ValidationResult] = field(default_factory=list[ValidationResult])
    semantic_score: float = 0.0
    total_score: float = 0.0
    primary_failure: FailureType = FailureType.UNKNOWN

    @property
    def passed(self) -> bool:
        """Return whether all non-skipped checks passed and semantic score is acceptable."""
        return all(result.passed or result.skipped for result in self.results) and self.primary_failure in {
            FailureType.UNKNOWN
        }

    def by_name(self, name: str) -> ValidationResult | None:
        """Return a validation result by name."""
        for result in self.results:
            if result.name == name:
                return result
        return None

    def compact_summary(self, max_lines: int = 120) -> str:
        """Return compact failure-oriented summary."""
        lines: list[str] = []
        for result in self.results:
            status = result.status.value
            lines.append(f"[{status}] {result.name}: exit={result.exit_code}")
            output = "\n".join(part for part in [result.stdout_excerpt.strip(), result.stderr_excerpt.strip()] if part)
            if output and not result.passed:
                lines.extend(output.splitlines()[:20])
        lines.append(f"semantic_score={self.semantic_score:.2f}")
        lines.append(f"total_score={self.total_score:.2f}")
        lines.append(f"primary_failure={self.primary_failure.value}")
        return "\n".join(lines[:max_lines])
