"""Result dataclass returned by ``DirectAgentLoop.run``."""

from __future__ import annotations

from dataclasses import dataclass, field

from mecha_agent_cli.validation.report import ValidationReport


@dataclass(frozen=True)
class AgentRunResult:
    """Outcome of one ``mecha-agent run`` invocation."""

    task_id: int
    status: str
    changed_files: list[str]
    validation_report: ValidationReport
    review_summary: str
    reward: float = 0.0
    repair_attempts: int = 0
    strategy_name: str = "direct_chat_history"
    attempt_snapshots: list[str] = field(default_factory=list[str])
    total_attempts: int = 0
    max_attempts: int = 0
    duration_sec: float = 0.0
