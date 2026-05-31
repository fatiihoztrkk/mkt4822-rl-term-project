"""Local Ollama service lifecycle helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager
from urllib.parse import urlparse

import httpx

from mecha_agent_cli.core.constants import DEFAULT_OLLAMA_BASE_URL
from mecha_agent_cli.core.errors import ModelError

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


class OllamaServiceManager:
    """Start a local Ollama service on demand and stop it if this process launched it."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        *,
        startup_timeout_sec: float = 20.0,
        poll_interval_sec: float = 0.25,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.startup_timeout_sec = startup_timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self._process: subprocess.Popen[bytes] | None = None
        self._started_by_us = False

    def start(self) -> None:
        """Start ``ollama serve`` once if the local service is not already healthy."""
        if not _is_local_base_url(self.base_url):
            return
        if self._is_healthy():
            return
        executable = shutil.which("ollama")
        if executable is None:
            raise ModelError("Ollama executable not found on PATH; cannot start local ollama service.")
        try:
            self._process = subprocess.Popen(
                [executable, "serve"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=_stable_ollama_env(),
            )
        except OSError as exc:
            raise ModelError(f"Failed to start local ollama service: {exc}") from exc
        self._started_by_us = True
        deadline = time.monotonic() + self.startup_timeout_sec
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                code = self._process.returncode
                self._reset_process_state()
                raise ModelError(f"Local ollama service exited before becoming ready (exit={code}).")
            if self._is_healthy():
                return
            time.sleep(self.poll_interval_sec)
        self.stop()
        raise ModelError(f"Timed out waiting for local ollama service at {self.base_url} to become ready.")

    def stop(self) -> None:
        """Stop the managed service if this process launched it."""
        if not self._started_by_us or self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        self._reset_process_state()

    def _is_healthy(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=1.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    def _reset_process_state(self) -> None:
        self._process = None
        self._started_by_us = False


@contextmanager
def managed_ollama_service(base_url: str, *, enabled: bool = True) -> Generator[None]:
    """Ensure a local Ollama service exists for the duration of one command or session."""
    if not enabled:
        yield
        return
    manager = OllamaServiceManager(base_url)
    manager.start()
    try:
        yield
    finally:
        manager.stop()


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    hostname = parsed.hostname
    return hostname in _LOCAL_HOSTS


def _stable_ollama_env() -> dict[str, str]:
    """Return conservative Ollama server defaults for single-GPU 6 GB VRAM setups."""
    env = os.environ.copy()
    env.setdefault("OLLAMA_NUM_PARALLEL", "1")
    env.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
    env.setdefault("OLLAMA_MAX_QUEUE", "16")
    env.setdefault("OLLAMA_FLASH_ATTENTION", "1")
    env.setdefault("OLLAMA_KV_CACHE_TYPE", "q8_0")
    return env
