from __future__ import annotations

import subprocess

import httpx
import pytest

from mecha_agent_cli.llm.ollama_service import OllamaServiceManager, managed_ollama_service


class _DummyHealthyResponse:
    def raise_for_status(self) -> None:
        return None


class _DummyProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float) -> int:
        self.wait_calls.append(timeout)
        return 0 if self.returncode is None else self.returncode


def test_managed_ollama_service_starts_and_stops_local_process(monkeypatch) -> None:
    process = _DummyProcess()
    popen_calls: list[list[str]] = []
    health_checks = iter([httpx.ConnectError("refused"), _DummyHealthyResponse()])

    def fake_get(*args, **kwargs) -> _DummyHealthyResponse:
        result = next(health_checks)
        if isinstance(result, Exception):
            raise result
        return result

    popen_envs: list[dict[str, str]] = []

    def fake_popen(args: list[str], **kwargs) -> _DummyProcess:
        popen_calls.append(args)
        popen_envs.append(kwargs.get("env", {}))
        return process

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ollama")

    with managed_ollama_service("http://localhost:11434"):
        assert process.terminated is False

    assert popen_calls == [["/usr/bin/ollama", "serve"]]
    assert process.terminated is True
    assert process.wait_calls == [5.0]
    assert popen_envs[0]["OLLAMA_NUM_PARALLEL"] == "1"
    assert popen_envs[0]["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert popen_envs[0]["OLLAMA_KV_CACHE_TYPE"] == "q8_0"


def test_managed_ollama_service_does_not_stop_existing_server(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _DummyHealthyResponse())
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Popen should not be called"))

    with managed_ollama_service("http://localhost:11434"):
        pass


def test_ollama_service_manager_skips_remote_base_url(monkeypatch) -> None:
    manager = OllamaServiceManager("http://example.com:11434")
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: pytest.fail("health check should not run"))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Popen should not be called"))

    manager.start()
