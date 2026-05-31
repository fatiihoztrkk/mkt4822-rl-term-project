"""Placeholder context helpers for course starter repositories.

This module intentionally avoids task-specific bucketing logic. Students can
replace these stubs with a real context strategy during the assignment.
"""

from __future__ import annotations


def task_family(user_request: str) -> str:
    """Return a constant placeholder task family label."""
    del user_request
    return "generic"


def build_context_key(*, user_request: str, model_name: str, has_baseline: bool) -> str:
    """Build a deterministic placeholder context key."""
    family = task_family(user_request)
    mode = "edit" if has_baseline else "new"
    safe_model = (model_name or "unknown").strip().lower() or "unknown"
    return f"{family}|{safe_model}|{mode}"


__all__ = ["build_context_key", "task_family"]
