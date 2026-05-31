"""Memory value models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class TaskRecord:
    """Task memory record."""

    id: int
    repo_id: str
    session_id: str
    timestamp: str
    user_request: str
    task_type: str
    target_files_json: str
    strategy_name: str
    model_name: str
    status: str


@dataclass(frozen=True)
class StrategyStat:
    """Strategy reward statistics."""

    strategy_name: str
    task_type: str
    model_name: str
    attempts: int
    successes: int
    mean_reward: float
    last_reward: float
    last_used_at: str


def now_iso() -> str:
    """Return current UTC timestamp."""
    return datetime.now(UTC).isoformat()
