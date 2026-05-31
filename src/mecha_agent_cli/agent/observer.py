"""Agent progress event model and observer protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class AgentProgressEvent:
    """One user-visible progress event emitted by the agent loop."""

    stage: str
    status: str
    detail: str = ""
    attempt_index: int | None = None
    total_attempts: int | None = None
    max_attempts: int | None = None
    metadata: dict[str, object] = field(default_factory=dict[str, object])


class AgentObserver(Protocol):
    """Callable observer for terminal, logs, or tests."""

    def __call__(self, event: AgentProgressEvent) -> None:
        """Handle one progress event."""


class NullObserver:
    """Observer that deliberately ignores all events."""

    def __call__(self, event: AgentProgressEvent) -> None:
        """Ignore one progress event."""
        _ = event
