"""Interactive REPL."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from mecha_agent_cli.app.commands import run_command
from mecha_agent_cli.config.loader import load_config
from mecha_agent_cli.llm.ollama_service import managed_ollama_service


def start_repl(repo_root: Path, *, backend: str = "ollama") -> None:
    """Start a minimal interactive task loop."""
    console = Console()
    console.print("[bold]mecha-agent chat[/bold] — type 'exit' to quit")
    config = load_config(repo_root)
    with managed_ollama_service(config.models.base_url, enabled=backend == "ollama"):
        while True:
            task = console.input("[cyan]task> [/cyan]").strip()
            if task.lower() in {"exit", "quit"}:
                return
            if not task:
                continue
            result = run_command(repo_root, task, backend=backend, manage_service=False)
            console.print(f"status={result.status} reward={result.reward} changed={result.changed_files}")
