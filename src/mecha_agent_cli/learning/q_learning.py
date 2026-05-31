"""Minimal placeholder module for future Q-learning integration.

This file intentionally does not provide a ready-to-run Q-learning
implementation. It exists so instructor workspaces can import a stable
module path while students implement approved RL components later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QLearningPlaceholder:
    """Simple import-safe marker object for course scaffolding."""

    name: str = "q_learning_placeholder"
    status: str = "not_implemented"


def placeholder_status() -> str:
    """Return a short status string for diagnostics and tests."""

    return "q_learning module is a placeholder for course setup"


__all__ = ["QLearningPlaceholder", "placeholder_status"]
