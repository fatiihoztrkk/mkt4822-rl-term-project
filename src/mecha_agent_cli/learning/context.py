"""Task-context helpers for the model-profile bandit."""

from __future__ import annotations


def task_family(user_request: str) -> str:
    """Classify a request into a compact, deterministic task family."""
    request = user_request.lower()
    if "fibonacci" in request:
        return "fibonacci_dp"
    if "sort" in request:
        return "sort"
    if "signal" in request or "filter" in request or "moving average" in request:
        return "signal_filter"
    if "numerical" in request or "matrix" in request or "linear system" in request:
        return "numerical"
    if "actor-critic" in request or "actor critic" in request or "reinforcement learning" in request:
        return "rl_actor_critic"
    if "graph" in request or "bfs" in request or "shortest path" in request:
        return "graph"
    if "string" in request or "word" in request or "text" in request:
        return "string"
    if "json" in request or "csv" in request or "serializer" in request:
        return "io"
    return "generic"


def build_context_key(*, user_request: str, model_name: str, has_baseline: bool) -> str:
    """Build the stable context key used for posterior storage."""
    family = task_family(user_request)
    mode = "edit" if has_baseline else "new"
    safe_model = (model_name or "unknown").strip().lower() or "unknown"
    return f"{family}|{safe_model}|{mode}"


__all__ = ["build_context_key", "task_family"]
