"""Local validation command runner."""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

from mecha_agent_cli.sandbox.base import CommandResult, EnvironmentRunner
from mecha_agent_cli.sandbox.command_policy import CommandPolicy


class LocalRunner(EnvironmentRunner):
    """Run allowlisted commands locally without shell access."""

    def __init__(self, policy: CommandPolicy | None = None, max_output_chars: int = 12000) -> None:
        self.policy = policy or CommandPolicy()
        self.max_output_chars = max_output_chars

    def run(self, command: list[str], cwd: Path, *, timeout_sec: float) -> CommandResult:
        """Run an allowlisted command using ``subprocess.run`` with ``shell=False``."""
        self.policy.require(command)
        start = time.monotonic()
        exec_command = self._execution_command(command)
        try:
            env = self._execution_env(command, cwd)
            with (
                tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file,
                tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file,
            ):
                completed = subprocess.run(
                    exec_command,
                    cwd=cwd,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    check=False,
                    timeout=timeout_sec,
                    env=env,
                    close_fds=True,
                )
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout = stdout_file.read()[-self.max_output_chars :]
                stderr = stderr_file.read()[-self.max_output_chars :]
            code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = _coerce_timeout_stream(exc.stdout)
            stderr = _coerce_timeout_stream(exc.stderr)
            stderr = f"{stderr}\nCommand timed out after {timeout_sec:.1f}s"
            code = 124
        return CommandResult(
            command=command,
            exit_code=code,
            stdout=stdout,
            stderr=stderr,
            duration_sec=time.monotonic() - start,
        )

    def _execution_command(self, command: list[str]) -> list[str]:
        """Map policy commands to stable interpreter invocations."""
        if command[:3] == ["python", "-m", "py_compile"]:
            return [sys.executable, "-S", "-B", "-m", "py_compile", *command[3:]]
        if command[:3] == ["python", "-I", "-c"]:
            # Use -S with an explicit PYTHONPATH instead of -I so validation avoids
            # user-site startup hooks while still being able to import installed
            # tools from site-packages when the environment is clean.
            return [sys.executable, "-S", "-B", "-c", *command[3:]]
        if command[:2] == ["pytest", "-q"]:
            # Running pytest with -S avoids third-party auto-start hooks that can
            # keep the interpreter alive after test completion in some environments.
            return [sys.executable, "-S", "-B", "-m", "pytest", "-q", *command[2:]]
        return command

    def _execution_env(self, command: list[str], cwd: Path) -> dict[str, str]:
        """Return a deterministic subprocess environment."""
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("PYTEST_CURRENT_TEST", None)
        if command[:3] == ["python", "-I", "-c"] or command[:2] == ["pytest", "-q"]:
            if command[:2] == ["pytest", "-q"]:
                env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
                env["PYTEST_ADDOPTS"] = ""
            purelib = sysconfig.get_paths().get("purelib", "")
            platlib = sysconfig.get_paths().get("platlib", "")
            path_entries = [str(cwd)]
            for entry in (purelib, platlib):
                if entry and entry not in path_entries:
                    path_entries.append(entry)
            existing = env.get("PYTHONPATH", "")
            if existing:
                path_entries.append(existing)
            env["PYTHONPATH"] = os.pathsep.join(path_entries)
        return env


def _coerce_timeout_stream(value: str | bytes | None) -> str:
    """Normalize subprocess timeout output to text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
