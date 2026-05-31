"""Real-runtime benchmark over hard RL/control/AI tasks.

Drives the actual ``DirectAgentLoop`` (the same class the user gets when they
run ``mecha-agent run``) against the curated task set in
``project-assignment/``. Each episode:

  1. spins up a fresh temporary workspace,
  2. writes a ``configs/default.yaml`` enabling or disabling the bandit,
  3. loads the task prompt from disk,
  4. invokes ``run_command`` end-to-end (chat -> code extraction -> validator),
  5. records per-attempt validator outcomes (syntax / import / runtime),
  6. when the bandit is enabled, posterior updates persist to a shared SQLite
     database so the agent learns across the whole task suite.

Usage::

    python scripts/hard_runtime_benchmark.py --backend ollama --mode both \
        --max-attempts 4 --repeat 1 --out runs/hard_bench.jsonl

Use ``--backend fake`` for a smoke test that exercises the full plumbing
without hitting Ollama.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from mecha_agent_cli.agent.observer import AgentProgressEvent
from mecha_agent_cli.agent.result import AgentRunResult
from mecha_agent_cli.app.commands import init_command, run_command
from mecha_agent_cli.learning.bandit import BanditStore
from mecha_agent_cli.learning.context import task_family

REPO_ROOT = Path(__file__).resolve().parent.parent
HARD_TASKS_DIR = REPO_ROOT / "project-assignment"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """A loaded task prompt."""

    name: str
    family: str
    prompt: str

    @property
    def stem(self) -> str:
        """File name without extension."""
        return Path(self.name).stem


@dataclass(frozen=True)
class StageOutcome:
    """Per-stage validator pass/fail for one episode."""

    syntax_pass: bool
    import_pass: bool
    runtime_pass: bool


@dataclass
class EpisodeRecord:
    """One end-to-end run of one task under one mode."""

    task: str
    family: str
    mode: str  # "baseline" or "bandit"
    repeat_index: int
    status: str  # success / budget_exhausted / error
    success: bool
    attempts: int
    reward: float
    strategy: str
    duration_sec: float
    syntax_pass: bool
    import_pass: bool
    runtime_pass: bool
    primary_failure: str
    workspace: str | None = None
    algorithm_sha: str | None = None
    algorithm_lines: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_tasks(directory: Path = HARD_TASKS_DIR, glob: str = "*.txt") -> list[TaskSpec]:
    """Load every task prompt from ``directory`` matching ``glob``.

    ``glob`` may contain multiple comma-separated patterns; the union of
    matched files is returned (deduplicated, sorted). Each prompt is also
    tagged with the bandit task family derived from its prose.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Hard tasks directory missing: {directory}")
    patterns = [p.strip() for p in glob.split(",") if p.strip()] or ["*.txt"]
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(directory.glob(pattern))
    specs: list[TaskSpec] = []
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        specs.append(TaskSpec(name=path.name, family=task_family(text), prompt=text))
    if not specs:
        raise FileNotFoundError(f"No tasks matched glob '{glob}' in {directory}")
    return specs


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------


_DEFAULT_YAML_TEMPLATE = """memory:
  path: .mecha-agent/memory.sqlite
direct:
  request_timeout_sec: 600
learning:
  enabled: {enabled}
  mode: "{mode}"
  persist: {persist}
"""


def _write_workspace_config(workspace: Path, *, enabled: bool, persist: bool) -> None:
    """Write ``configs/default.yaml`` to drive the agent's learning settings."""
    configs = workspace / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    yaml_text = _DEFAULT_YAML_TEMPLATE.format(
        enabled=str(enabled).lower(),
        mode="thompson" if enabled else "off",
        persist=str(persist).lower(),
    )
    (configs / "default.yaml").write_text(yaml_text, encoding="utf-8")


def _link_shared_bandit_db(workspace: Path, shared_db: Path) -> None:
    """Make the workspace's memory.sqlite point at the shared bandit DB.

    The DirectAgentLoop reads ``config.memory.path`` relative to the workspace
    root. We make that file a symlink to the shared DB so all episodes write
    into one set of bandit_arms rows.
    """
    target = workspace / ".mecha-agent" / "memory.sqlite"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(shared_db.resolve())


# ---------------------------------------------------------------------------
# Validator-stage extraction
# ---------------------------------------------------------------------------


_SYNTAX_NAMES = {"syntax", "syntax_check", "py_compile"}
_IMPORT_NAMES = {"import", "import_check", "import_smoke"}
_RUNTIME_NAMES = {"runtime", "runtime_check", "runtime_smoke", "exec"}


def _stage_outcomes(result: AgentRunResult) -> StageOutcome:
    """Project the validation report onto syntax / import / runtime booleans."""
    syntax = import_ = runtime = False
    for r in result.validation_report.results:
        name = r.name.lower()
        if name in _SYNTAX_NAMES:
            syntax = r.passed or r.skipped
        elif name in _IMPORT_NAMES:
            import_ = r.passed or r.skipped
        elif name in _RUNTIME_NAMES:
            runtime = r.passed or r.skipped
    return StageOutcome(syntax_pass=syntax, import_pass=import_, runtime_pass=runtime)


# ---------------------------------------------------------------------------
# Single-episode driver
# ---------------------------------------------------------------------------


class _SilentObserver:
    """Observer that drops everything (keeps benchmark output clean)."""

    def __call__(self, event: AgentProgressEvent) -> None:
        """Discard event."""
        del event


def _run_one_episode(
    task: TaskSpec,
    *,
    mode: str,
    repeat_index: int,
    backend: str,
    scenario: str,
    max_attempts: int,
    shared_db: Path | None,
    manage_service: bool,
    workspaces_dir: Path | None = None,
) -> EpisodeRecord:
    """Run one (task, mode) episode in an isolated workspace.

    When ``workspaces_dir`` is given, the workspace persists at
    ``<workspaces_dir>/<mode>/r<repeat>/<task_stem>/`` so a human can inspect
    the generated ``algorithm.py`` afterwards. Otherwise the workspace is a
    TemporaryDirectory that vanishes at the end of the call.
    """
    started = time.perf_counter()
    if workspaces_dir is not None:
        workspace = workspaces_dir / mode / f"r{repeat_index}" / task.stem
        if workspace.exists():
            import shutil as _sh

            _sh.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        return _drive_episode(
            task,
            workspace=workspace,
            mode=mode,
            repeat_index=repeat_index,
            backend=backend,
            scenario=scenario,
            max_attempts=max_attempts,
            shared_db=shared_db,
            manage_service=manage_service,
            started=started,
            persisted=True,
        )
    with TemporaryDirectory(prefix="mecha-bench-") as tmp:
        workspace = Path(tmp)
        return _drive_episode(
            task,
            workspace=workspace,
            mode=mode,
            repeat_index=repeat_index,
            backend=backend,
            scenario=scenario,
            max_attempts=max_attempts,
            shared_db=shared_db,
            manage_service=manage_service,
            started=started,
            persisted=False,
        )


def _drive_episode(
    task: TaskSpec,
    *,
    workspace: Path,
    mode: str,
    repeat_index: int,
    backend: str,
    scenario: str,
    max_attempts: int,
    shared_db: Path | None,
    manage_service: bool,
    started: float,
    persisted: bool,
) -> EpisodeRecord:
    """Inner driver that runs the agent against an already-prepared workspace."""
    use_bandit = mode == "bandit"
    error: str | None = None
    init_command(workspace, force=True)
    _write_workspace_config(workspace, enabled=use_bandit, persist=use_bandit)
    if use_bandit and shared_db is not None:
        _link_shared_bandit_db(workspace, shared_db)
    try:
        result = run_command(
            workspace,
            task.prompt,
            backend=backend,
            scenario=scenario,
            max_attempts=max_attempts,
            until_success=False,
            observer=_SilentObserver(),
            manage_service=manage_service,
        )
    except Exception as exc:  # benchmark must not crash on a single task
        error = f"{type(exc).__name__}: {exc}"
        return EpisodeRecord(
            task=task.stem,
            family=task.family,
            mode=mode,
            repeat_index=repeat_index,
            status="error",
            success=False,
            attempts=0,
            reward=-1.0,
            strategy="error",
            duration_sec=time.perf_counter() - started,
            syntax_pass=False,
            import_pass=False,
            runtime_pass=False,
            primary_failure="ERROR",
            workspace=str(workspace) if persisted else None,
            algorithm_sha=None,
            algorithm_lines=0,
            error=error,
        )
    stages = _stage_outcomes(result)
    algo_path = workspace / "algorithm.py"
    sha = None
    line_count = 0
    if algo_path.exists():
        text = algo_path.read_text(encoding="utf-8", errors="replace")
        line_count = text.count("\n") + (0 if text.endswith("\n") else 1)
        import hashlib as _hl

        sha = _hl.sha256(text.encode("utf-8")).hexdigest()[:12]
    return EpisodeRecord(
        task=task.stem,
        family=task.family,
        mode=mode,
        repeat_index=repeat_index,
        status=result.status,
        success=result.status == "success",
        attempts=result.total_attempts,
        reward=result.reward,
        strategy=result.strategy_name,
        duration_sec=time.perf_counter() - started,
        syntax_pass=stages.syntax_pass,
        import_pass=stages.import_pass,
        runtime_pass=stages.runtime_pass,
        primary_failure=result.validation_report.primary_failure.name,
        workspace=str(workspace) if persisted else None,
        algorithm_sha=sha,
        algorithm_lines=line_count,
        error=None,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _summary_for_mode(records: list[EpisodeRecord]) -> dict[str, object]:
    """Aggregate per-mode metrics."""
    if not records:
        return {"episodes": 0}
    n = len(records)
    successes = sum(1 for r in records if r.success)
    syntax_ok = sum(1 for r in records if r.syntax_pass)
    import_ok = sum(1 for r in records if r.import_pass)
    runtime_ok = sum(1 for r in records if r.runtime_pass)
    attempts = [r.attempts for r in records if r.attempts > 0]
    failure_breakdown = Counter(r.primary_failure for r in records if not r.success)
    return {
        "episodes": n,
        "success_rate": successes / n,
        "syntax_pass_rate": syntax_ok / n,
        "import_pass_rate": import_ok / n,
        "runtime_pass_rate": runtime_ok / n,
        "mean_attempts": statistics.mean(attempts) if attempts else 0.0,
        "mean_reward": statistics.mean(r.reward for r in records),
        "mean_duration_sec": statistics.mean(r.duration_sec for r in records),
        "failure_breakdown": dict(failure_breakdown),
    }


def _per_family_stats(records: Iterable[EpisodeRecord]) -> dict[str, dict[str, float]]:
    """Group success rate by family."""
    bucket: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for r in records:
        bucket[r.family].append(r)
    return {
        family: {
            "episodes": len(rs),
            "success_rate": sum(1 for r in rs if r.success) / len(rs),
            "mean_attempts": statistics.mean(r.attempts for r in rs if r.attempts > 0)
            if any(r.attempts > 0 for r in rs)
            else 0.0,
        }
        for family, rs in sorted(bucket.items())
    }


def _bandit_posterior(shared_db: Path) -> list[dict[str, object]]:
    """Dump the per-(context, arm) posterior so we can see what the bandit learned."""
    if not shared_db.exists():
        return []
    store = BanditStore(shared_db)
    rows: list[dict[str, object]] = []
    for stat in store.all_rows():
        rows.append(
            {
                "context_key": stat.context_key,
                "arm_id": stat.arm_id,
                "alpha": stat.alpha,
                "beta": stat.beta,
                "pulls": stat.pulls,
                "mean": stat.mean,
            }
        )
    rows.sort(key=lambda r: (str(r["context_key"]), -float(r["mean"])))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-runtime hard-task benchmark.")
    parser.add_argument(
        "--backend",
        choices=["ollama", "fake"],
        default="ollama",
        help="LLM backend (default: ollama).",
    )
    parser.add_argument(
        "--scenario",
        default="default",
        help="FakeModelClient scenario name (only used with --backend fake).",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "bandit", "both"],
        default="both",
        help="Run baseline (learning off), bandit (learning on), or both.",
    )
    parser.add_argument("--max-attempts", type=int, default=4, help="Per-episode attempt budget.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to run each task per mode (for variance reduction).",
    )
    parser.add_argument(
        "--task-glob",
        default="*.txt",
        help="Glob (within project-assignment) selecting which tasks to run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "runs" / "hard_bench.jsonl",
        help="JSONL output path; parent dirs are created.",
    )
    parser.add_argument(
        "--bandit-db",
        type=Path,
        default=None,
        help=(
            "Path to a persistent SQLite DB shared by every bandit episode. "
            "Default: a temp file inside the run directory."
        ),
    )
    parser.add_argument(
        "--no-manage-service",
        action="store_true",
        help="Do not auto-start ollama serve (use if you started it manually).",
    )
    parser.add_argument(
        "--workspaces-dir",
        type=Path,
        default=None,
        help=(
            "Persist each episode's workspace under "
            "<workspaces-dir>/<mode>/r<repeat>/<task>/ instead of using a temp dir. "
            "Lets you inspect the generated algorithm.py afterwards."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tasks = load_tasks(glob=args.task_glob)
    modes: list[str] = ["baseline", "bandit"] if args.mode == "both" else [args.mode]

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bandit_db = args.bandit_db or (out_path.parent / "hard_bench_bandit.sqlite")
    if "bandit" in modes and bandit_db.exists():
        bandit_db.unlink()  # start each benchmark from a fresh posterior

    print(f"Loaded {len(tasks)} tasks; modes={modes}; max_attempts={args.max_attempts}; repeat={args.repeat}")
    print(f"Backend: {args.backend}; output: {out_path}")
    if "bandit" in modes:
        print(f"Bandit DB: {bandit_db}")

    records: list[EpisodeRecord] = []
    with out_path.open("w", encoding="utf-8") as sink:
        for repeat_index in range(args.repeat):
            for task in tasks:
                for mode in modes:
                    rec = _run_one_episode(
                        task,
                        mode=mode,
                        repeat_index=repeat_index,
                        backend=args.backend,
                        scenario=args.scenario,
                        max_attempts=args.max_attempts,
                        shared_db=bandit_db if mode == "bandit" else None,
                        manage_service=not args.no_manage_service,
                        workspaces_dir=args.workspaces_dir,
                    )
                    records.append(rec)
                    sink.write(json.dumps(asdict(rec)) + "\n")
                    sink.flush()
                    flag = "OK " if rec.success else "FAIL"
                    print(
                        f"[{flag}] r={repeat_index} {mode:8s} {task.stem:42s} "
                        f"attempts={rec.attempts} status={rec.status} "
                        f"syn={int(rec.syntax_pass)} imp={int(rec.import_pass)} "
                        f"run={int(rec.runtime_pass)} dur={rec.duration_sec:.1f}s"
                    )

    print("\n=== SUMMARY ===")
    summary: dict[str, object] = {}
    for mode in modes:
        mode_records = [r for r in records if r.mode == mode]
        summary[mode] = _summary_for_mode(mode_records)
        summary[f"{mode}_per_family"] = _per_family_stats(mode_records)
    if "bandit" in modes:
        summary["bandit_posterior"] = _bandit_posterior(bandit_db)
    print(json.dumps(summary, indent=2))

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
