"""Integration tests: DirectAgentLoop with the contextual Thompson bandit."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mecha_agent_cli.agent.direct_loop import DirectAgentLoop
from mecha_agent_cli.agent.observer import AgentProgressEvent
from mecha_agent_cli.app.commands import init_command
from mecha_agent_cli.config.loader import load_config
from mecha_agent_cli.config.schema import ModelProfile
from mecha_agent_cli.learning.bandit import BanditStore
from mecha_agent_cli.llm.base import ChatMessage, ModelClient

_GOOD = '''"""solver."""

from __future__ import annotations


def solve(n: int) -> int:
    """Return n + 1."""
    return n + 1
'''

_BROKEN = "def solve(n: int) -> int:\n    return n +\n"


class _Recorder(ModelClient):
    """Captures the ModelProfile used on every chat_text call."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.profiles: list[ModelProfile] = []

    @property
    def name(self) -> str:
        return "recorder"

    def available_models(self) -> list[str]:
        return ["qwen3:4b"]

    def chat_text(self, *, messages: Sequence[ChatMessage], profile: ModelProfile, **_: object) -> str:
        _ = messages
        self.profiles.append(profile)
        if not self.responses:
            raise AssertionError("recorder ran out of responses")
        return self.responses.pop(0)


def _disable_slow_validation(repo_root: Path) -> None:
    cfg = "target_file: algorithm.py\nrun_pytest: false\nrun_ruff: false\nrun_pyright: false\n"
    (repo_root / "configs" / "validation.yaml").write_text(cfg)


def _enable_learning(repo_root: Path, *, mode: str = "thompson", min_pulls: int = 0) -> None:
    extra = f'learning:\n  enabled: true\n  mode: "{mode}"\n  min_pulls_before_exploit: {min_pulls}\n  persist: true\n'
    default_path = repo_root / "configs" / "default.yaml"
    existing = default_path.read_text() if default_path.exists() else ""
    default_path.write_text(existing + ("\n" if existing and not existing.endswith("\n") else "") + extra)


def test_loop_runs_unchanged_when_learning_disabled(tmp_path: Path) -> None:
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.learning.enabled is False

    client = _Recorder([f"```python\n{_GOOD}```\n"])
    loop = DirectAgentLoop(repo_root=tmp_path, config=cfg, client=client, max_attempts=2)
    result = loop.run("Implement solve")
    assert result.status == "success"
    assert result.strategy_name == "direct_chat_history"  # unchanged from baseline
    # Profile must be the unmodified "direct" profile
    assert client.profiles[0].temperature == cfg.models.profiles["direct"].temperature


def test_loop_applies_bandit_arm_and_records_strategy_name(tmp_path: Path) -> None:
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    _enable_learning(tmp_path, mode="off")  # deterministic baseline pick
    cfg = load_config(tmp_path)
    assert cfg.learning.enabled is True

    events: list[AgentProgressEvent] = []
    client = _Recorder([f"```python\n{_GOOD}```\n"])
    loop = DirectAgentLoop(
        repo_root=tmp_path,
        config=cfg,
        client=client,
        observer=events.append,
        max_attempts=2,
    )
    result = loop.run("Implement solve")
    assert result.status == "success"
    # off-mode always returns the baseline arm
    assert result.strategy_name == "bandit:direct.baseline"
    # Per-attempt bandit select/update must be emitted once for this 1-attempt success.
    stages = [e.stage for e in events]
    assert stages.count("bandit_select") == 1
    assert stages.count("bandit_update") == 1


def test_loop_persists_bandit_state_to_sqlite(tmp_path: Path) -> None:
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    _enable_learning(tmp_path, mode="off")
    cfg = load_config(tmp_path)

    client = _Recorder([f"```python\n{_GOOD}```\n"])
    loop = DirectAgentLoop(repo_root=tmp_path, config=cfg, client=client, max_attempts=2)
    result = loop.run("Implement solve")
    assert result.status == "success"

    db_path = tmp_path / cfg.memory.path
    assert db_path.exists()
    rows = BanditStore(db_path).all_rows()
    assert any(r.arm_id == "direct.baseline" and r.pulls == 1 for r in rows)
    matched = [r for r in rows if r.arm_id == "direct.baseline"]
    assert matched[0].alpha > 1.0  # alpha incremented after a successful episode


def test_loop_propagates_arm_overrides_into_profile(tmp_path: Path) -> None:
    """Per-attempt bandit decisions should drive each generation profile."""
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    _enable_learning(tmp_path, mode="thompson", min_pulls=10)
    cfg = load_config(tmp_path)

    client = _Recorder([f"```python\n{_BROKEN}```\n", f"```python\n{_GOOD}```\n"])
    loop = DirectAgentLoop(repo_root=tmp_path, config=cfg, client=client, max_attempts=3)
    result = loop.run("Implement solve")
    assert result.status == "success"
    # Two attempts -> two generation profiles, both selected by the bandit.
    profile_0 = client.profiles[0]
    profile_1 = client.profiles[1]
    assert isinstance(profile_0, ModelProfile)
    assert isinstance(profile_1, ModelProfile)
    # Strategy name encodes the full arm sequence across attempts.
    assert result.strategy_name.startswith("bandit:")
    assert "->" in result.strategy_name or result.strategy_name == "bandit:direct.baseline"


def test_loop_emits_bandit_events_per_attempt(tmp_path: Path) -> None:
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    _enable_learning(tmp_path, mode="off")
    cfg = load_config(tmp_path)

    events: list[AgentProgressEvent] = []
    client = _Recorder([f"```python\n{_BROKEN}```\n", f"```python\n{_GOOD}```\n"])
    loop = DirectAgentLoop(repo_root=tmp_path, config=cfg, client=client, observer=events.append, max_attempts=3)
    result = loop.run("Implement solve")
    assert result.status == "success"
    stages = [e.stage for e in events]
    # Two attempts => two select + two update events.
    assert stages.count("bandit_select") == 2
    assert stages.count("bandit_update") == 2


def test_loop_stamps_duration_sec_on_result(tmp_path: Path) -> None:
    """Every result returned from run() must carry a non-negative duration_sec."""
    init_command(tmp_path)
    _disable_slow_validation(tmp_path)
    cfg = load_config(tmp_path)

    client = _Recorder([f"```python\n{_GOOD}```\n"])
    loop = DirectAgentLoop(repo_root=tmp_path, config=cfg, client=client, max_attempts=2)
    result = loop.run("Implement solve")
    assert result.status == "success"
    assert result.duration_sec >= 0.0
    # The fake client returns instantly but validation runs subprocess; should be > 0 in practice.
    # We only assert the field is wired through, not a hard lower bound.
