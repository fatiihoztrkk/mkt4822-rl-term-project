"""Environment runner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    """Subprocess result with bounded output."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float

    @property
    def passed(self) -> bool:
        """Return whether exit code is zero."""
        return self.exit_code == 0


class EnvironmentRunner(ABC):
    """Abstract validation/runtime command runner."""

    @abstractmethod
    def run(self, command: list[str], cwd: Path, *, timeout_sec: float) -> CommandResult:
        """Run a command safely."""
