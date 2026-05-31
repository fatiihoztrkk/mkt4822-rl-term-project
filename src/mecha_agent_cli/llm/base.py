"""Model client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from mecha_agent_cli.config.schema import ModelProfile

ChatMessage = dict[str, str]
StreamCallback = Callable[[str], None]


class ModelClient(ABC):
    """Abstract chat client used by the direct-generation loop."""

    @abstractmethod
    def chat_text(
        self,
        *,
        messages: Sequence[ChatMessage],
        profile: ModelProfile,
        on_thinking: StreamCallback | None = None,
        on_content: StreamCallback | None = None,
    ) -> str:
        """Run a multi-turn chat and return the raw assistant message body.

        ``messages`` is an ordered list of ``{"role": ..., "content": ...}``
        dicts with roles ``system``, ``user``, or ``assistant``. The returned
        string is the assistant response; callers handle code extraction.

        ``on_thinking`` and ``on_content`` are optional callbacks invoked with
        text chunks as they arrive when the backend supports streaming.
        Implementations that do not stream may invoke them once at the end or
        skip them entirely.
        """

    @abstractmethod
    def available_models(self) -> list[str]:
        """Return locally available model names, if the backend supports it."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name for logs."""
