"""Command allowlist for validation runners."""

from __future__ import annotations

from dataclasses import dataclass

from mecha_agent_cli.core.errors import SecurityError


@dataclass(frozen=True)
class CommandDecision:
    """Outcome of command authorization."""

    allowed: bool
    reason: str


class CommandPolicy:
    """Exact/prefix command allowlist; no shell commands are accepted."""

    def __init__(self) -> None:
        self.allowed_prefixes: tuple[tuple[str, ...], ...] = (
            ("python", "-m", "py_compile"),
            ("python", "-I", "-c"),
            ("ruff", "check"),
            ("ruff", "format", "--check"),
            ("pyright",),
            ("pytest", "-q"),
            ("git", "diff", "--"),
            ("git", "status", "--short"),
        )
        self.forbidden = {"rm", "curl", "wget", "ssh", "scp", "sudo", "chmod", "chown", "pip", "uv"}

    def check(self, command: list[str]) -> CommandDecision:
        """Check a command represented as argv tokens."""
        if not command:
            return CommandDecision(False, "empty command")
        if any(token in self.forbidden for token in command):
            return CommandDecision(False, "forbidden executable or token")
        for prefix in self.allowed_prefixes:
            if tuple(command[: len(prefix)]) == prefix:
                return CommandDecision(True, "allowed")
        return CommandDecision(False, f"command not allowlisted: {' '.join(command)}")

    def require(self, command: list[str]) -> None:
        """Raise SecurityError if command is not allowed."""
        decision = self.check(command)
        if not decision.allowed:
            raise SecurityError(decision.reason)
