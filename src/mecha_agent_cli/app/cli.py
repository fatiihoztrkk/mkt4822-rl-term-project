"""Typer CLI entrypoint."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import typer
from rich.console import Console

from mecha_agent_cli import __version__
from mecha_agent_cli.agent.direct_loop import MissingPackagePrompt
from mecha_agent_cli.app.commands import doctor_command, init_command, judge_command, run_command, validate_command
from mecha_agent_cli.app.interactive import start_repl
from mecha_agent_cli.config.loader import dump_config, load_config
from mecha_agent_cli.core.constants import DEFAULT_MODEL, FALLBACK_MODEL
from mecha_agent_cli.memory.sqlite_store import SQLiteStore
from mecha_agent_cli.repo.workspace import discover_workspace
from mecha_agent_cli.ui.progress import RichAgentObserver
from mecha_agent_cli.ui.tables import key_value_table

app = typer.Typer(help="Local Qwen3 4B direct-generation coding agent CLI.")
config_app = typer.Typer(help="Configuration commands.")
models_app = typer.Typer(help="Model diagnostics.")
memory_app = typer.Typer(help="Memory commands.")
app.add_typer(config_app, name="config")
app.add_typer(models_app, name="models")
app.add_typer(memory_app, name="memory")
console = Console()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    """Initialize the CLI."""
    if version:
        console.print(__version__)
        raise typer.Exit()


@app.command("init")
def init(
    path: Path = typer.Argument(Path.cwd(), help="Workspace path."),
    force: bool = typer.Option(False, "--force", help="Overwrite generated workspace files."),
) -> None:
    """Initialize a mecha-agent workspace."""
    init_command(path.resolve(), force=force)
    console.print(f"[green]Initialized mecha-agent workspace at {path.resolve()}[/green]")


@app.command("run")
def run(
    task: str = typer.Argument(..., help="Natural-language coding task."),
    backend: str = typer.Option("ollama", "--backend", help="ollama or fake."),
    fake_scenario: str = typer.Option("default", "--fake-scenario", help="Fake model scenario."),
    path: Path = typer.Option(Path.cwd(), "--path", help="Workspace path."),
    max_attempts: int = typer.Option(
        0,
        "--max-attempts",
        help="Maximum direct-generation attempts. 0 uses config default.",
    ),
    until_success: bool = typer.Option(
        False,
        "--until-success",
        help="Keep generating until validators pass or the user interrupts.",
    ),
    auto_install: bool = typer.Option(
        False,
        "--auto-install",
        help="Install missing third-party packages without prompting (use only in trusted shells).",
    ),
    no_install: bool = typer.Option(
        False,
        "--no-install",
        help="Never install missing packages; always feed the import error back to the model.",
    ),
    judge_final: bool = typer.Option(
        False,
        "--judge-final",
        help="After run completes, invoke a short-form LLM judge over task+logs+runtime output.",
    ),
) -> None:
    """Run direct generation: each attempt writes algorithm.py and feeds errors back.

    The cumulative chat history merges all prior assistant outputs and validator
    feedbacks so the model builds on its own previous code on every retry.
    """
    if auto_install and no_install:
        raise typer.BadParameter("--auto-install and --no-install are mutually exclusive.")

    observer = RichAgentObserver(console)
    prompt = _build_install_prompt(auto_install=auto_install, no_install=no_install, observer=observer)
    result = run_command(
        path.resolve(),
        task,
        backend=backend,
        scenario=fake_scenario,
        max_attempts=max_attempts if max_attempts > 0 else None,
        until_success=until_success if until_success else None,
        observer=observer,
        missing_package_prompt=prompt,
    )
    primary_failure_value = result.validation_report.primary_failure.value
    primary_failure_display = (
        "NONE" if result.status == "success" or primary_failure_value == "UNKNOWN" else primary_failure_value
    )
    console.print(
        key_value_table(
            "Run result",
            {
                "task_id": result.task_id,
                "status": result.status,
                "changed_files": ", ".join(result.changed_files) or "none",
                "attempt_snapshots": ", ".join(result.attempt_snapshots) or "none",
                "total_attempts": result.total_attempts,
                "max_attempts": "infinite" if result.max_attempts <= 0 else result.max_attempts,
                "primary_failure": primary_failure_display,
            },
        )
    )
    console.print(result.validation_report.compact_summary(max_lines=60))
    if judge_final:
        judge = judge_command(path.resolve(), task, backend=backend, scenario=fake_scenario)
        console.print(
            key_value_table(
                "Judge",
                {
                    "verdict": judge.verdict,
                    "confidence": f"{judge.confidence:.2f}",
                    "suspected_false_positive": judge.suspected_false_positive,
                    "primary_failure_guess": judge.primary_failure_guess,
                    "report_path": judge.report_path,
                },
            )
        )
        if judge.reasons:
            console.print("Judge reasons:")
            for reason in judge.reasons:
                console.print(f"- {reason}")


@app.command("judge")
def judge(
    task: str = typer.Argument(..., help="Natural-language task text used for adjudication context."),
    backend: str = typer.Option("ollama", "--backend", help="ollama or fake."),
    fake_scenario: str = typer.Option("default", "--fake-scenario", help="Fake model scenario."),
    path: Path = typer.Option(Path.cwd(), "--path", help="Workspace path."),
    out: Path = typer.Option(Path(".mecha-agent/judge/latest.json"), "--out", help="Output JSON report path."),
) -> None:
    """Run short-form post-run LLM adjudication for verification."""
    report = judge_command(
        path.resolve(),
        task,
        backend=backend,
        scenario=fake_scenario,
        out_path=(path.resolve() / out),
    )
    console.print(
        key_value_table(
            "Judge",
            {
                "verdict": report.verdict,
                "confidence": f"{report.confidence:.2f}",
                "suspected_false_positive": report.suspected_false_positive,
                "primary_failure_guess": report.primary_failure_guess,
                "report_path": report.report_path,
            },
        )
    )
    if report.reasons:
        console.print("Judge reasons:")
        for reason in report.reasons:
            console.print(f"- {reason}")


@app.command("chat")
def chat(
    backend: str = typer.Option("ollama", "--backend", help="ollama or fake."),
    path: Path = typer.Option(Path.cwd(), "--path", help="Workspace path."),
) -> None:
    """Start interactive REPL."""
    start_repl(path.resolve(), backend=backend)


@app.command("validate")
def validate(
    path: Path = typer.Option(Path.cwd(), "--path", help="Workspace path."),
    user_request: str = typer.Option("", "--task", help="Optional task text (unused; reserved for future hooks)."),
) -> None:
    """Run validation pipeline (syntax + import)."""
    report = validate_command(path.resolve(), user_request=user_request)
    console.print(report.compact_summary())
    raise typer.Exit(0 if report.primary_failure.value == "UNKNOWN" else 1)


@app.command("review")
def review(path: Path = typer.Option(Path.cwd(), "--path", help="Workspace path.")) -> None:
    """Review current validation status without changing files."""
    report = validate_command(path.resolve())
    console.print(report.compact_summary())


@app.command("repair")
def repair(
    task: str = typer.Option("Fix latest validation failure", "--task", help="Repair task."),
    backend: str = typer.Option("ollama", "--backend", help="ollama or fake."),
    path: Path = typer.Option(Path.cwd(), "--path", help="Workspace path."),
) -> None:
    """Run a repair task against the current workspace."""
    result = run_command(path.resolve(), task, backend=backend)
    console.print(result.validation_report.compact_summary())


@app.command("doctor")
def doctor(
    path: Path = typer.Option(Path.cwd(), "--path", help="Workspace path."),
    backend: str = typer.Option("ollama", "--backend", help="ollama or fake."),
) -> None:
    """Check Python, tools, Ollama, and model availability."""
    report = doctor_command(path.resolve(), backend=backend)
    console.print(key_value_table("Doctor", report.__dict__))
    if report.message:
        console.print(report.message)


@config_app.command("show")
def config_show(path: Path = typer.Option(Path.cwd(), "--path", help="Workspace path.")) -> None:
    """Show merged configuration."""
    console.print(dump_config(load_config(path.resolve())))


@config_app.command("doctor")
def config_doctor(path: Path = typer.Option(Path.cwd(), "--path", help="Workspace path.")) -> None:
    """Validate local config files."""
    config = load_config(path.resolve())
    console.print(f"target_file={config.target_file}")
    console.print(f"default_model={config.models.default_model}")


@models_app.command("list")
def models_list() -> None:
    """Print preferred and fallback model names."""
    console.print(f"default: {DEFAULT_MODEL}", markup=False, highlight=False)
    console.print(f"fallback: {FALLBACK_MODEL}", markup=False, highlight=False)
    console.print(f"pull: ollama pull {DEFAULT_MODEL}", markup=False, highlight=False)


@models_app.command("doctor")
def models_doctor(path: Path = typer.Option(Path.cwd(), "--path", help="Workspace path.")) -> None:
    """Check Ollama model availability."""
    report = doctor_command(path.resolve())
    console.print(
        key_value_table(
            "Models",
            {
                "ollama_ok": report.ollama_ok,
                DEFAULT_MODEL: report.default_model_available,
                FALLBACK_MODEL: report.fallback_model_available,
                "fallback_instruction": f"ollama pull {FALLBACK_MODEL}",
            },
        )
    )
    if report.message:
        console.print(report.message)


@memory_app.command("list")
def memory_list(path: Path = typer.Option(Path.cwd(), "--path", help="Workspace path.")) -> None:
    """List recent memory records."""
    root = discover_workspace(path.resolve())
    store = SQLiteStore(root / ".mecha-agent" / "memory.sqlite")
    rows = store.list_tasks(limit=20)
    if not rows:
        console.print("No task memory records.")
        return
    for row in rows:
        console.print(f"#{row.id} {row.timestamp} {row.task_type} {row.status}: {row.user_request}")


@memory_app.command("clear")
def memory_clear(
    yes: bool = typer.Option(False, "--yes", help="Do not prompt for confirmation."),
    path: Path = typer.Option(Path.cwd(), "--path", help="Workspace path."),
) -> None:
    """Clear local memory database."""
    if not yes and not typer.confirm("Clear local mecha-agent memory?"):
        raise typer.Exit()
    root = discover_workspace(path.resolve())
    SQLiteStore(root / ".mecha-agent" / "memory.sqlite").clear()
    console.print("Memory cleared.")


def _build_install_prompt(
    *,
    auto_install: bool,
    no_install: bool,
    observer: RichAgentObserver | None = None,
) -> MissingPackagePrompt:
    """Return a callable that decides whether to install missing packages.

    - ``--no-install``  → always returns ``False``.
    - ``--auto-install`` → always returns ``True`` and prints what's installing.
    - otherwise         → interactive y/N confirm. If stdin is not a TTY the
                          prompt cannot be answered and we decline.

    When an ``observer`` is supplied its live spinner is paused for the
    duration of the prompt so it does not redraw over the input line.
    """
    if no_install:

        def _decline(_missing: set[str]) -> bool:
            return False

        return _decline

    def _prompt(missing: set[str]) -> bool:
        names = ", ".join(sorted(missing))
        if auto_install:
            console.print(f"[yellow]Auto-installing missing package(s):[/yellow] {names}")
            return True
        if not sys.stdin.isatty():
            console.print(
                f"[yellow]Validator missing package(s):[/yellow] {names} "
                "(stdin is not a TTY; declining install. Pass --auto-install to enable.)"
            )
            return False
        pause_cm = observer.pause() if observer is not None else nullcontext()
        with pause_cm:
            console.print(
                f"[yellow]Validator failed: missing package(s):[/yellow] {names}\n"
                f"[dim]Will install into:[/dim] {sys.executable}"
            )
            return typer.confirm(f"Install {names} now?", default=False)

    return _prompt
