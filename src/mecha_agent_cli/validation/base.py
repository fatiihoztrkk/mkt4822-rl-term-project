"""Validation check interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from mecha_agent_cli.validation.report import ValidationResult


class ValidationCheck(ABC):
    """Abstract validation check."""

    name: str

    @abstractmethod
    def run(self, repo_root: Path) -> ValidationResult:
        """Run the check."""
