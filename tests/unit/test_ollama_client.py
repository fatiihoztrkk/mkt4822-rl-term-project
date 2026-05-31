"""Tests for the Ollama HTTP chat client."""

from __future__ import annotations

import httpx

from mecha_agent_cli.config.schema import ModelProfile, ModelsConfig
from mecha_agent_cli.core.errors import ModelError, OutputParseError
from mecha_agent_cli.llm.ollama_client import OllamaClient


class _DummyResponse:
    def __init__(self, content: str, *, thinking: str | None = None) -> None:
        self._content = content
        self._thinking = thinking

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        message: dict[str, object] = {"content": self._content}
        if self._thinking is not None:
            message["thinking"] = self._thinking
        return {"message": message}


def _make_client(**overrides: object) -> OllamaClient:
    return OllamaClient(
        base_url="http://example.test",
        models_config=ModelsConfig(),
        connect_retries=overrides.pop("connect_retries", 0),  # type: ignore[arg-type]
        retry_delay_sec=overrides.pop("retry_delay_sec", 0.0),  # type: ignore[arg-type]
    )


def test_chat_text_retries_connect_error_then_succeeds(monkeypatch) -> None:
    responses: list[object] = [
        httpx.ConnectError("connection refused"),
        _DummyResponse("hello"),
    ]

    def fake_post(*_args: object, **_kwargs: object) -> _DummyResponse:
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(httpx, "post", fake_post)
    client = _make_client(connect_retries=1)

    result = client.chat_text(
        messages=[{"role": "user", "content": "hi"}],
        profile=ModelProfile(model="test-model"),
    )

    assert result == "hello"


def test_chat_text_reports_unreachable_service(monkeypatch) -> None:
    def fake_post(*_args: object, **_kwargs: object) -> _DummyResponse:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)
    client = _make_client()

    try:
        client.chat_text(
            messages=[{"role": "user", "content": "hi"}],
            profile=ModelProfile(model="test-model"),
        )
    except ModelError as exc:
        assert "ollama serve" in str(exc).lower()
        assert "http://example.test" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ModelError")


def test_chat_text_rejects_thinking_only_response(monkeypatch) -> None:
    def fake_post(*_args: object, **_kwargs: object) -> _DummyResponse:
        return _DummyResponse("", thinking="planning ...")

    monkeypatch.setattr(httpx, "post", fake_post)
    client = _make_client()

    try:
        client.chat_text(
            messages=[{"role": "user", "content": "hi"}],
            profile=ModelProfile(model="test-model", think=False),
        )
    except OutputParseError as exc:
        assert "thinking" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected OutputParseError")


def test_chat_text_propagates_top_level_think_flag(monkeypatch) -> None:
    observed: list[dict[str, object]] = []
    chunks: list[str] = []

    class _FakeStream:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            observed.append(kwargs["json"])  # type: ignore[index]

        def __enter__(self) -> _FakeStream:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            yield '{"message":{"thinking":"reasoning step"}}'
            yield '{"message":{"content":"final"},"done":true}'

    monkeypatch.setattr(httpx, "stream", _FakeStream)
    client = _make_client()

    result = client.chat_text(
        messages=[{"role": "user", "content": "hi"}],
        profile=ModelProfile(model="test-model", think=True),
        on_thinking=chunks.append,
    )

    assert observed[0]["think"] is True
    assert observed[0]["stream"] is True
    assert chunks == ["reasoning step"]
    assert result == "final"


def test_chat_text_stream_falls_back_to_thinking_when_content_missing(monkeypatch) -> None:
    chunks: list[str] = []

    class _FakeStream:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def __enter__(self) -> _FakeStream:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            yield '{"message":{"thinking":"only-thought"}}'
            yield '{"done":true}'

    monkeypatch.setattr(httpx, "stream", _FakeStream)
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: _DummyResponse("final-from-blocking"))
    client = _make_client()

    result = client.chat_text(
        messages=[{"role": "user", "content": "hi"}],
        profile=ModelProfile(model="test-model", think=True),
        on_thinking=chunks.append,
    )

    assert chunks == ["only-thought"]
    assert result == "final-from-blocking"
