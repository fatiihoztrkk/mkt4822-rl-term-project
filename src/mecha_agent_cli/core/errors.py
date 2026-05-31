"""Typed exceptions used by the agent runtime."""


class MechaError(Exception):
    """Base exception for mecha-agent-cli."""


class ConfigError(MechaError):
    """Raised when configuration cannot be loaded or validated."""


class ModelError(MechaError):
    """Raised when model calls fail."""


class OutputParseError(MechaError):
    """Raised when structured model output cannot be parsed."""


class PatchError(MechaError):
    """Raised for malformed or non-applicable patches."""


class SecurityError(MechaError):
    """Raised when a path, command, prompt, or patch violates policy."""


class ValidationError(MechaError):
    """Raised for validation pipeline failures when fail-fast mode is enabled."""
