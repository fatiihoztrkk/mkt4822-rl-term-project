"""SQLite-backed task, validation, reflection, and strategy memory."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from mecha_agent_cli.memory.migrations import SCHEMA_SQL
from mecha_agent_cli.memory.models import TaskRecord, now_iso
from mecha_agent_cli.validation.report import ValidationReport


def sha256_text(text: str) -> str:
    """Return SHA-256 hex digest for text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SQLiteStore:
    """SQLite persistence layer for replayable agent state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        """Open a connection with row access by name."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        """Create all tables if absent."""
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def create_task(
        self,
        *,
        repo_id: str,
        session_id: str,
        user_request: str,
        task_type: str,
        target_files: list[str],
        strategy_name: str,
        model_name: str,
        status: str = "started",
    ) -> int:
        """Insert and return a task id."""
        self.initialize()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tasks(repo_id, session_id, timestamp, user_request, task_type,
                                  target_files_json, strategy_name, model_name, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    repo_id,
                    session_id,
                    now_iso(),
                    user_request,
                    task_type,
                    json.dumps(target_files),
                    strategy_name,
                    model_name,
                    status,
                ),
            )
            task_id = cursor.lastrowid
            if task_id is None:
                raise RuntimeError("SQLite did not return a task id")
            conn.commit()
            return int(task_id)

    def update_task_status(self, task_id: int, status: str) -> None:
        """Update status of a task."""
        with self.connect() as conn:
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
            conn.commit()

    def add_artifact(self, *, task_id: int, file_path: str, before: str, after: str, diff_text: str) -> None:
        """Store a changed file artifact."""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts(task_id, file_path, before_hash, after_hash, diff_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task_id, file_path, sha256_text(before), sha256_text(after), diff_text, now_iso()),
            )
            conn.commit()

    def add_validation_run(self, *, task_id: int, attempt_index: int, report: ValidationReport) -> None:
        """Store an aggregate validation report."""

        def ok(name: str) -> int:
            result = report.by_name(name)
            return 1 if result is None or result.skipped or result.passed else 0

        duration = sum(result.duration_sec for result in report.results)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO validation_runs(task_id, attempt_index, syntax_ok, import_ok, ruff_ok,
                  pyright_ok, pytest_ok, semantic_score, total_score, failure_type, failure_summary,
                  duration_sec, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    attempt_index,
                    ok("syntax"),
                    ok("import"),
                    ok("ruff"),
                    ok("pyright"),
                    ok("pytest"),
                    report.semantic_score,
                    report.total_score,
                    report.primary_failure.value,
                    report.compact_summary(),
                    duration,
                    now_iso(),
                ),
            )
            conn.commit()

    def add_reflection(
        self,
        *,
        task_id: int,
        failure_type: str,
        root_cause: str,
        lesson: str,
        future_rule: str,
        applicable_task_types: list[str],
        applicable_files: list[str],
    ) -> None:
        """Store a reflection memory."""
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO reflections(task_id, failure_type, root_cause, lesson, future_rule,
                  applicable_task_types_json, applicable_files_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    failure_type,
                    root_cause,
                    lesson,
                    future_rule,
                    json.dumps(applicable_task_types),
                    json.dumps(applicable_files),
                    now_iso(),
                ),
            )
            conn.commit()

    def update_strategy_stats(
        self,
        *,
        strategy_name: str,
        task_type: str,
        model_name: str,
        reward: float,
        success: bool,
    ) -> None:
        """Update epsilon-greedy strategy statistics incrementally."""
        self.initialize()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM strategy_stats
                WHERE strategy_name = ? AND task_type = ? AND model_name = ?
                """,
                (strategy_name, task_type, model_name),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO strategy_stats(strategy_name, task_type, model_name, attempts, successes,
                      mean_reward, last_reward, last_used_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        strategy_name,
                        task_type,
                        model_name,
                        1,
                        1 if success else 0,
                        reward,
                        reward,
                        now_iso(),
                    ),
                )
            else:
                attempts = int(row["attempts"]) + 1
                successes = int(row["successes"]) + (1 if success else 0)
                old_mean = float(row["mean_reward"])
                mean_reward = old_mean + (reward - old_mean) / attempts
                conn.execute(
                    """
                    UPDATE strategy_stats
                    SET attempts = ?, successes = ?, mean_reward = ?, last_reward = ?, last_used_at = ?
                    WHERE id = ?
                    """,
                    (attempts, successes, mean_reward, reward, now_iso(), int(row["id"])),
                )
            conn.commit()

    def list_tasks(self, limit: int = 20) -> list[TaskRecord]:
        """Return recent tasks."""
        self.initialize()
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM tasks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [TaskRecord(**dict(row)) for row in rows]

    def list_reflections(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent reflection rows as dictionaries."""
        self.initialize()
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM reflections ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def strategy_rows(self, task_type: str, model_name: str) -> list[dict[str, Any]]:
        """Return strategy stats for a task/model pair."""
        self.initialize()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM strategy_stats WHERE task_type = ? AND model_name = ?",
                (task_type, model_name),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear(self) -> None:
        """Delete the memory database."""
        if self.db_path.exists():
            self.db_path.unlink()
