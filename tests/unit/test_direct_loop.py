"""Unit tests for the direct generation loop's history-merge behavior."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mecha_agent_cli.agent.direct_loop import DirectAgentLoop
from mecha_agent_cli.app.commands import init_command
from mecha_agent_cli.config.loader import load_config
from mecha_agent_cli.config.schema import ModelProfile
from mecha_agent_cli.llm.base import ChatMessage, ModelClient

_GOOD_CODE = '''"""solver."""

from __future__ import annotations


def solve(n: int) -> int:
    """Return n + 1."""
    return n + 1
'''

_BROKEN_CODE = "def solve(n: int) -> int:\n    return n +\n"


class _ScriptedClient(ModelClient):
    """A ModelClient that returns scripted chat_text responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.captured_messages: list[list[ChatMessage]] = []

    @property
    def name(self) -> str:
        return "scripted"

    def available_models(self) -> list[str]:
        return ["qwen3:4b"]

    def chat_text(
        self,
        *,
        messages: Sequence[ChatMessage],
        profile: ModelProfile,
        on_thinking: object | None = None,
        on_content: object | None = None,
    ) -> str:
        _ = on_thinking, on_content
        _ = profile
        self.captured_messages.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("Scripted client ran out of responses")
        return self.responses.pop(0)


def _disable_slow_validation(repo_root: Path) -> None:
    cfg = "target_file: algorithm.py\nrun_pytest: false\nrun_ruff: false\nrun_pyright: false\n"
    (repo_root / "configs" / "validation.yaml").write_text(cfg)


def test_direct_loop_writes_algorithm_py_on_first_attempt(tmp_path: Path) -> None:
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    config = load_config(tmp_path)
    client = _ScriptedClient([f"```python\n{_GOOD_CODE}```\n"])
    loop = DirectAgentLoop(repo_root=tmp_path, config=config, client=client, max_attempts=3)
    result = loop.run("Implement solve(n: int) -> int that returns n + 1")
    assert result.status == "success"
    assert result.total_attempts == 1
    assert (tmp_path / "algorithm.py").read_text().startswith('"""solver."""')


def test_direct_loop_accumulates_history_across_attempts(tmp_path: Path) -> None:
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    config = load_config(tmp_path)
    client = _ScriptedClient(
        [
            f"```python\n{_BROKEN_CODE}```\n",
            f"```python\n{_GOOD_CODE}```\n",
        ]
    )
    loop = DirectAgentLoop(repo_root=tmp_path, config=config, client=client, max_attempts=3)
    result = loop.run("Implement solve(n: int) -> int that returns n + 1")
    assert result.status == "success"
    assert result.total_attempts == 2
    # Second call must include the first assistant turn AND a follow-up user turn
    second_messages = client.captured_messages[1]
    roles = [m["role"] for m in second_messages]
    assert roles[0] == "system"
    assert roles[1] == "user"  # original task
    assert roles[2] == "assistant"  # broken draft echoed back
    assert roles[3] == "user"  # validator feedback
    assert any("VALIDATOR REPORT" in m["content"] for m in second_messages if m["role"] == "user")


def test_direct_loop_does_not_block_arbitrary_imports(tmp_path: Path) -> None:
    """No AST safety gate: a module with `import subprocess` (or torch, etc.) is fine
    as long as it parses and imports cleanly.
    """
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    config = load_config(tmp_path)
    code = (
        "from __future__ import annotations\n\n"
        "import json  # arbitrary imports allowed\n\n"
        "def solve(data: list[int]) -> list[int]:\n"
        '    """Echo input."""\n'
        "    return list(data)\n"
    )
    client = _ScriptedClient([f"```python\n{code}```\n"])
    loop = DirectAgentLoop(repo_root=tmp_path, config=config, client=client, max_attempts=3)
    result = loop.run("Implement solve")
    assert result.status == "success"
    assert result.total_attempts == 1


def test_direct_loop_truncates_history_when_max_chars_exceeded(tmp_path: Path) -> None:
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    config = load_config(tmp_path)
    config.direct.max_history_chars = 1500  # very small to force truncation
    bulky = "# " + ("filler " * 200) + "\n" + _BROKEN_CODE
    responses = [f"```python\n{bulky}```\n"] * 5 + [f"```python\n{_GOOD_CODE}```\n"]
    client = _ScriptedClient(responses)
    loop = DirectAgentLoop(repo_root=tmp_path, config=config, client=client, max_attempts=10)
    result = loop.run("Implement solve(n: int) -> int that returns n + 1")
    assert result.status == "success"
    # Collected message count must be bounded once truncation starts kicking in;
    # the head (system + first user) is always preserved, so messages list never
    # grows beyond the head plus a small recent tail of assistant/user pairs.
    longest_history_len = max(len(m) for m in client.captured_messages)
    # 2 head + at most 4 trailing turns retained (2 assistant + 2 user pairs).
    assert longest_history_len <= 2 + 4 + 2


def test_direct_loop_autonomous_judge_can_trigger_extra_attempt(tmp_path: Path) -> None:
    """After validator pass, judge FAIL should force another generation attempt."""
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    config = load_config(tmp_path)
    config.direct.auto_judge_after_validation = True
    config.direct.judge_max_repairs = 2
    config.direct.judge_min_confidence = 0.25

    judge_fail = (
        '{"verdict":"FAIL","confidence":0.9,"suspected_false_positive":false,'
        '"primary_failure_guess":"SEMANTIC","reasons":["metrics contradict threshold"]}'
    )
    judge_pass = (
        '{"verdict":"PASS","confidence":0.9,"suspected_false_positive":false,'
        '"primary_failure_guess":"UNKNOWN","reasons":["looks good"]}'
    )
    client = _ScriptedClient(
        [
            f"```python\n{_GOOD_CODE}```\n",  # attempt-1 code
            judge_fail,  # judge response for attempt-1
            f"```python\n{_GOOD_CODE}```\n",  # attempt-2 code
            judge_pass,  # judge response for attempt-2
        ]
    )
    loop = DirectAgentLoop(repo_root=tmp_path, config=config, client=client, max_attempts=3)
    result = loop.run("Implement solve(n: int) -> int that returns n + 1")
    assert result.status == "success"
    assert result.total_attempts == 2


def test_direct_loop_judge_rejection_exhausts_budget(tmp_path: Path) -> None:
    """When judge keeps rejecting and no budget remains, run must fail."""
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    config = load_config(tmp_path)
    config.direct.auto_judge_after_validation = True
    config.direct.judge_max_repairs = 0

    judge_fail = (
        '{"verdict":"FAIL","confidence":0.9,"suspected_false_positive":false,'
        '"primary_failure_guess":"SEMANTIC","reasons":["semantic mismatch"]}'
    )
    client = _ScriptedClient(
        [
            f"```python\n{_GOOD_CODE}```\n",  # attempt-1 code
            judge_fail,  # judge response for attempt-1
        ]
    )
    loop = DirectAgentLoop(repo_root=tmp_path, config=config, client=client, max_attempts=1)
    result = loop.run("Implement solve(n: int) -> int that returns n + 1")
    assert result.status == "attempt_budget_exhausted"
    assert result.validation_report.primary_failure.value == "SEMANTIC"
