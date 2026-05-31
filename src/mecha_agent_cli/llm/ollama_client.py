"""Native Ollama HTTP client for ``/api/chat`` with optional streaming."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import cast

import httpx

from mecha_agent_cli.config.schema import ModelProfile, ModelsConfig
from mecha_agent_cli.core.constants import DEFAULT_OLLAMA_BASE_URL
from mecha_agent_cli.core.errors import ModelError, OutputParseError
from mecha_agent_cli.llm.base import ChatMessage, ModelClient, StreamCallback


class OllamaClient(ModelClient):
    """HTTP client for Ollama ``/api/chat`` (multi-turn).

    Streams NDJSON when callers pass ``on_thinking``/``on_content`` callbacks
    OR when the active profile has ``think=True``; otherwise sends a single
    non-streaming request.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_sec: float = 600.0,
        *,
        models_config: ModelsConfig | None = None,
        connect_retries: int = 2,
        retry_delay_sec: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.models_config = models_config
        self.connect_retries = connect_retries
        self.retry_delay_sec = retry_delay_sec

    @property
    def name(self) -> str:
        """Return backend name."""
        return "ollama"

    def available_models(self) -> list[str]:
        """Return model names from ``/api/tags``."""
        data = self._get_json("/api/tags")
        if not isinstance(data, Mapping):
            return []
        models = cast(Mapping[str, object], data).get("models", [])
        if not isinstance(models, list):
            return []
        names: list[str] = []
        for item in cast(list[object], models):
            if isinstance(item, Mapping):
                name = cast(Mapping[str, object], item).get("name")
                if isinstance(name, str):
                    names.append(name)
        return names

    def running_models(self) -> list[dict[str, object]]:
        """Return models currently loaded in Ollama memory from ``/api/ps``."""
        data = self._get_json("/api/ps")
        if not isinstance(data, Mapping):
            return []
        models = cast(Mapping[str, object], data).get("models", [])
        if not isinstance(models, list):
            return []
        return [
            dict(cast(Mapping[str, object], item)) for item in cast(list[object], models) if isinstance(item, Mapping)
        ]

    def chat_text(
        self,
        *,
        messages: Sequence[ChatMessage],
        profile: ModelProfile,
        on_thinking: StreamCallback | None = None,
        on_content: StreamCallback | None = None,
    ) -> str:
        """Run a multi-turn chat and return the assistant content.

        Streams thinking/content chunks to the supplied callbacks when either
        is provided or when ``profile.think`` is True.
        """
        stream = on_thinking is not None or on_content is not None or bool(profile.think)
        payload: dict[str, object] = {
            "model": profile.model,
            "messages": [dict(m) for m in messages],
            "stream": stream,
            "options": profile.to_options(),
        }
        if profile.think is not None:
            payload["think"] = profile.think
        if profile.keep_alive is not None:
            payload["keep_alive"] = profile.keep_alive
        if stream:
            return self._post_chat_stream(payload, on_thinking=on_thinking, on_content=on_content)
        return self._post_chat_blocking(payload)

    # -- HTTP helpers ----------------------------------------------------

    def _get_json(self, endpoint: str) -> object:
        try:
            response = httpx.get(f"{self.base_url}{endpoint}", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc
        return response.json()

    def _timeout(self) -> httpx.Timeout:
        # Generous total timeout for slow first-token latency on small local
        # GPUs; tight 10 s connect cap so a missing daemon fails fast.
        return httpx.Timeout(self.timeout_sec, connect=min(self.timeout_sec, 10.0))

    def _post_chat_blocking(self, payload: dict[str, object]) -> str:
        for attempt in range(self.connect_retries + 1):
            try:
                response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self._timeout())
                response.raise_for_status()
                return _extract_message_content(response.json())
            except httpx.ConnectError as exc:
                if attempt >= self.connect_retries:
                    raise ModelError(
                        f"Ollama chat failed at {self.base_url}: {exc}. Ensure `ollama serve` is running and reachable."
                    ) from exc
                time.sleep(self.retry_delay_sec)
            except httpx.HTTPError as exc:
                raise ModelError(f"Ollama chat failed at {self.base_url}: {exc}") from exc
        raise RuntimeError("unreachable")

    def _post_chat_stream(
        self,
        payload: dict[str, object],
        *,
        on_thinking: StreamCallback | None,
        on_content: StreamCallback | None,
    ) -> str:
        """Stream NDJSON chat chunks, fire callbacks, and return aggregated content."""
        for attempt in range(self.connect_retries + 1):
            try:
                content_buf: list[str] = []
                thinking_buf: list[str] = []
                with httpx.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self._timeout(),
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(event, Mapping):
                            continue
                        message = cast(Mapping[str, object], event).get("message")
                        if isinstance(message, Mapping):
                            body = cast(Mapping[str, object], message)
                            thinking = body.get("thinking")
                            if isinstance(thinking, str) and thinking:
                                thinking_buf.append(thinking)
                                if on_thinking is not None:
                                    on_thinking(thinking)
                            content = body.get("content")
                            if isinstance(content, str) and content:
                                content_buf.append(content)
                                if on_content is not None:
                                    on_content(content)
                        if cast(Mapping[str, object], event).get("done"):
                            break
                joined = "".join(content_buf)
                if not joined.strip():
                    thinking_joined = "".join(thinking_buf)
                    if thinking_joined.strip():
                        payload_blocking = dict(payload)
                        payload_blocking["stream"] = False
                        try:
                            return self._post_chat_blocking(payload_blocking)
                        except OutputParseError:
                            return thinking_joined
                    raise OutputParseError("Ollama stream produced no usable message content")
                return joined
            except httpx.ConnectError as exc:
                if attempt >= self.connect_retries:
                    raise ModelError(
                        f"Ollama chat failed at {self.base_url}: {exc}. Ensure `ollama serve` is running and reachable."
                    ) from exc
                time.sleep(self.retry_delay_sec)
            except httpx.HTTPError as exc:
                raise ModelError(f"Ollama chat failed at {self.base_url}: {exc}") from exc
        raise RuntimeError("unreachable")


def _extract_message_content(data: object) -> str:
    if not isinstance(data, Mapping):
        raise OutputParseError("Ollama response body was not a JSON object")
    message = cast(Mapping[str, object], data).get("message")
    if not isinstance(message, Mapping):
        raise OutputParseError("Ollama response did not contain a message object")
    body = cast(Mapping[str, object], message)
    content = body.get("content")
    if isinstance(content, str) and content.strip():
        return content
    thinking = body.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        raise OutputParseError(
            "Ollama returned thinking text without final content; set think=false for the active profile"
        )
    raise OutputParseError("Ollama response did not contain usable message content")
