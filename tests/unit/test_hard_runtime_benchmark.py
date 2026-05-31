"""Tests for ``scripts/hard_runtime_benchmark.py``.

Exercises the pure-python plumbing (task loading, family tagging, summary
aggregation, CLI parsing) without hitting Ollama. A single end-to-end episode
is run with the FakeModelClient to make sure the workspace setup, validator,
and stage-extraction wiring still compose.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "hard_runtime_benchmark.py"
HARD_TASKS_DIR = REPO_ROOT / "project-assignment"


def _import_benchmark():
    """Import the benchmark module by file path (it is not a package member)."""
    spec = importlib.util.spec_from_file_location("hard_runtime_benchmark", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("hard_runtime_benchmark", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark_module():
    return _import_benchmark()


def test_hard_task_directory_has_at_least_2_files() -> None:
    files = sorted(HARD_TASKS_DIR.glob("*.txt"))
    assert len(files) >= 2, f"Expected >=2 assignment tasks, found {len(files)}"


def test_load_tasks_tags_each_with_a_family(benchmark_module) -> None:
    specs = benchmark_module.load_tasks(HARD_TASKS_DIR)
    assert len(specs) >= 2
    families = {s.family for s in specs}
    # Family resolver only emits keywords from the registry; every task must hit one.
    assert all(isinstance(s.family, str) and s.family for s in specs)
    assert len(families) >= 1


def test_load_tasks_rejects_missing_directory(benchmark_module, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        benchmark_module.load_tasks(tmp_path / "does-not-exist")


def test_load_tasks_rejects_empty_glob(benchmark_module, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError):
        benchmark_module.load_tasks(tmp_path, glob="*.nope")


def test_summary_for_mode_handles_empty_records(benchmark_module) -> None:
    assert benchmark_module._summary_for_mode([]) == {"episodes": 0}


def test_summary_for_mode_reports_rates(benchmark_module) -> None:
    cls = benchmark_module.EpisodeRecord
    records = [
        cls(
            task="t1",
            family="numerical",
            mode="bandit",
            repeat_index=0,
            status="success",
            success=True,
            attempts=1,
            reward=1.0,
            strategy="bandit:direct.cool",
            duration_sec=0.1,
            syntax_pass=True,
            import_pass=True,
            runtime_pass=True,
            primary_failure="UNKNOWN",
        ),
        cls(
            task="t2",
            family="generic",
            mode="bandit",
            repeat_index=0,
            status="budget_exhausted",
            success=False,
            attempts=4,
            reward=-0.4,
            strategy="bandit:direct.warm",
            duration_sec=0.2,
            syntax_pass=True,
            import_pass=False,
            runtime_pass=False,
            primary_failure="IMPORT_ERROR",
        ),
    ]
    s = benchmark_module._summary_for_mode(records)
    assert s["episodes"] == 2
    assert s["success_rate"] == 0.5
    assert s["syntax_pass_rate"] == 1.0
    assert s["import_pass_rate"] == 0.5
    assert s["runtime_pass_rate"] == 0.5
    assert s["failure_breakdown"] == {"IMPORT_ERROR": 1}


def test_per_family_stats_groups_correctly(benchmark_module) -> None:
    cls = benchmark_module.EpisodeRecord

    def _mk(family: str, success: bool, attempts: int) -> object:
        return cls(
            task="x",
            family=family,
            mode="baseline",
            repeat_index=0,
            status="success" if success else "budget_exhausted",
            success=success,
            attempts=attempts,
            reward=1.0 if success else -0.1,
            strategy="direct_chat_history",
            duration_sec=0.01,
            syntax_pass=True,
            import_pass=True,
            runtime_pass=success,
            primary_failure="UNKNOWN" if success else "RUNTIME_ERROR",
        )

    records = [_mk("numerical", True, 1), _mk("numerical", False, 4), _mk("generic", True, 2)]
    grouped = benchmark_module._per_family_stats(records)
    assert grouped["numerical"]["episodes"] == 2
    assert grouped["numerical"]["success_rate"] == 0.5
    assert grouped["generic"]["success_rate"] == 1.0


def test_parse_args_defaults(benchmark_module) -> None:
    ns = benchmark_module._parse_args(["--backend", "fake"])
    assert ns.backend == "fake"
    assert ns.mode == "both"
    assert ns.max_attempts == 4
    assert ns.repeat == 1


def test_parse_args_overrides(benchmark_module) -> None:
    ns = benchmark_module._parse_args(
        [
            "--backend",
            "fake",
            "--mode",
            "bandit",
            "--max-attempts",
            "2",
            "--task-glob",
            "ai_*.txt",
        ]
    )
    assert ns.mode == "bandit"
    assert ns.max_attempts == 2
    assert ns.task_glob == "ai_*.txt"


def test_run_one_episode_with_fake_backend(benchmark_module) -> None:
    """Smoke test: full plumbing runs end-to-end against the FakeModelClient.

    We deliberately do NOT assert on success/failure of the validator — the
    fake backend returns a generic envelope that does not implement the hard
    task. We only require that the run completes without crashing and that
    the record has all the expected fields populated.
    """
    specs = benchmark_module.load_tasks(HARD_TASKS_DIR)
    task = specs[0]
    with TemporaryDirectory() as out_dir:
        rec = benchmark_module._run_one_episode(
            task,
            mode="baseline",
            repeat_index=0,
            backend="fake",
            scenario="default",
            max_attempts=1,
            shared_db=None,
            manage_service=False,
        )
        assert rec.task == task.stem
        assert rec.mode == "baseline"
        assert rec.status in {"success", "attempt_budget_exhausted", "budget_exhausted", "error"}
        assert isinstance(rec.attempts, int)
        assert isinstance(rec.duration_sec, float)
        assert rec.duration_sec >= 0.0
        # smoke: file was created so syntax/import flags are observable
        assert isinstance(rec.syntax_pass, bool)
        assert Path(out_dir).exists()
