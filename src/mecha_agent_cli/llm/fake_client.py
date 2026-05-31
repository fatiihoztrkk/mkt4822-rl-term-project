"""Deterministic fake model backend for tests and demos."""

from __future__ import annotations

from collections.abc import Sequence

from mecha_agent_cli.config.schema import ModelProfile
from mecha_agent_cli.llm.base import ChatMessage, ModelClient, StreamCallback

_SORT_CODE = '''"""Generated algorithm implementation."""

from __future__ import annotations


def solve(data: list[int]) -> list[int]:
    """Return a stable sorted copy of ``data`` using merge sort."""
    values = list(data)
    if len(values) <= 1:
        return values
    midpoint = len(values) // 2
    left = solve(values[:midpoint])
    right = solve(values[midpoint:])
    merged: list[int] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged
'''

_FIBONACCI_CODE = '''"""Generated algorithm implementation."""

from __future__ import annotations


def solve(n: int) -> int:
    """Return the nth Fibonacci number for ``n >= 0``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a = 0
    b = 1
    for _ in range(n):
        a, b = b, a + b
    return a
'''

_MOVING_AVERAGE_CODE = '''"""Generated algorithm implementation."""

from __future__ import annotations


def solve(data: list[float], window: int) -> list[float]:
    """Return a causal moving-average control signal filter for ``data``."""
    if window <= 0:
        raise ValueError("window must be positive")
    output: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(data):
        running_sum += value
        if index >= window:
            running_sum -= data[index - window]
            divisor = window
        else:
            divisor = index + 1
        output.append(running_sum / divisor)
    return output
'''

_GENERIC_CODE = '''"""Generated algorithm implementation."""

from __future__ import annotations


def solve(request: dict[str, object]) -> dict[str, object]:
    """Return a deterministic result envelope for a generic request."""
    inputs = request.get("inputs", request)
    return {"outputs": inputs, "diagnostics": {"status": "ok"}}
'''


def _select_code(prompt: str) -> str:
    lower = prompt.lower()
    if "fibonacci" in lower:
        return _FIBONACCI_CODE
    if "moving" in lower or "filter" in lower:
        return _MOVING_AVERAGE_CODE
    if "sort" in lower:
        return _SORT_CODE
    return _GENERIC_CODE


class FakeModelClient(ModelClient):
    """A deterministic ``ModelClient`` returning fenced code based on the task."""

    def __init__(self, scenario: str = "default") -> None:
        self.scenario = scenario
        self._chat_text_calls = 0

    @property
    def name(self) -> str:
        """Return backend name."""
        return "fake"

    def available_models(self) -> list[str]:
        """Return fake model names."""
        return ["qwen3:4b"]

    def chat_text(
        self,
        *,
        messages: Sequence[ChatMessage],
        profile: ModelProfile,
        on_thinking: StreamCallback | None = None,
        on_content: StreamCallback | None = None,
    ) -> str:
        """Return a deterministic fenced-code response based on the first user turn."""
        del profile, on_thinking, on_content
        self._chat_text_calls += 1
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")

        if self.scenario == "syntax_then_repair" and self._chat_text_calls == 1:
            return "Here you go:\n```python\ndef solve(data: list[int]) -> list[int]:\n    return sorted(data\n```\n"
        if self.scenario == "no_block_first" and self._chat_text_calls == 1:
            return "Sorry, here is just a description without a code block."

        code = _select_code(first_user)
        return f"Sure, here is algorithm.py:\n```python\n{code}```\n"
