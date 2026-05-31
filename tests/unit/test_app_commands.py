from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from mecha_agent_cli.agent.result import AgentRunResult
from mecha_agent_cli.app import commands
from mecha_agent_cli.config.loader import load_config
from mecha_agent_cli.config.schema import ModelProfile
from mecha_agent_cli.llm.base import ChatMessage, ModelClient
from mecha_agent_cli.validation.report import ValidationReport


def test_load_config_accepts_string_model_default_options(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "models.yaml").write_text(
        'default_options:\n  num_ctx: 4096\n  keep_alive: "30s"\n',
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.models.default_options["num_ctx"] == 4096
    assert config.models.default_options["keep_alive"] == "30s"


def test_run_command_manages_ollama_service_once(tmp_path: Path, monkeypatch) -> None:
    config = load_config(tmp_path)
    lifecycle: list[tuple[str, bool, str]] = []

    @contextmanager
    def fake_managed_service(base_url: str, *, enabled: bool = True):
        lifecycle.append((base_url, enabled, "enter"))
        try:
            yield
        finally:
            lifecycle.append((base_url, enabled, "exit"))

    class _DummyLoop:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def run(self, task: str) -> AgentRunResult:
            return AgentRunResult(
                task_id=1,
                status="ok",
                changed_files=[],
                validation_report=ValidationReport(),
                review_summary="",
                reward=0.0,
                repair_attempts=0,
                strategy_name="test",
            )

    monkeypatch.setattr(commands, "managed_ollama_service", fake_managed_service)
    monkeypatch.setattr(commands, "load_config", lambda repo_root: config)
    monkeypatch.setattr(commands, "get_client", lambda config, backend, scenario: object())
    monkeypatch.setattr(commands, "DirectAgentLoop", _DummyLoop)

    result = commands.run_command(tmp_path, "test task", backend="ollama")

    assert result.status == "ok"
    assert lifecycle == [
        (config.models.base_url, True, "enter"),
        (config.models.base_url, True, "exit"),
    ]


def test_run_command_skips_ollama_service_for_fake_backend(tmp_path: Path, monkeypatch) -> None:
    config = load_config(tmp_path)
    lifecycle: list[tuple[str, bool, str]] = []

    @contextmanager
    def fake_managed_service(base_url: str, *, enabled: bool = True):
        lifecycle.append((base_url, enabled, "enter"))
        try:
            yield
        finally:
            lifecycle.append((base_url, enabled, "exit"))

    class _DummyLoop:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def run(self, task: str) -> AgentRunResult:
            return AgentRunResult(
                task_id=1,
                status="ok",
                changed_files=[],
                validation_report=ValidationReport(),
                review_summary="",
                reward=0.0,
                repair_attempts=0,
                strategy_name="test",
            )

    monkeypatch.setattr(commands, "managed_ollama_service", fake_managed_service)
    monkeypatch.setattr(commands, "load_config", lambda repo_root: config)
    monkeypatch.setattr(commands, "get_client", lambda config, backend, scenario: object())
    monkeypatch.setattr(commands, "DirectAgentLoop", _DummyLoop)

    commands.run_command(tmp_path, "test task", backend="fake")

    assert lifecycle == [
        (config.models.base_url, False, "enter"),
        (config.models.base_url, False, "exit"),
    ]


def test_parse_judge_json_accepts_fenced_payload() -> None:
    raw = (
        "```json\n"
        '{"verdict":"PASS","confidence":0.8,'
        '"suspected_false_positive":false,'
        '"primary_failure_guess":"UNKNOWN","reasons":["ok"]}'
        "\n```"
    )
    parsed = commands._parse_judge_json(raw)
    assert parsed["verdict"] == "PASS"
    assert parsed["confidence"] == 0.8
    assert parsed["reasons"] == ["ok"]


def test_parse_judge_json_invalid_returns_unsure() -> None:
    parsed = commands._parse_judge_json("not json at all")
    assert parsed["verdict"] == "UNSURE"
    assert parsed["suspected_false_positive"] is True


def test_parse_judge_json_prose_fallback_extracts_verdict() -> None:
    raw = (
        'Analysis... verdict: "FAIL" confidence: 0.92 suspected_false_positive: true primary_failure_guess: "SEMANTIC"'
    )
    parsed = commands._parse_judge_json(raw)
    assert parsed["verdict"] == "FAIL"
    assert parsed["confidence"] == 0.92
    assert parsed["primary_failure_guess"] == "SEMANTIC"


class _JudgeClient(ModelClient):
    @property
    def name(self) -> str:
        return "judge-client"

    def available_models(self) -> list[str]:
        return ["qwen3:4b"]

    def chat_text(self, *, messages: list[ChatMessage], profile: ModelProfile, **_: object) -> str:
        _ = messages, profile
        return (
            '{"verdict":"FAIL","confidence":0.92,"suspected_false_positive":true,'
            '"primary_failure_guess":"SEMANTIC","reasons":["runtime metric contradicts threshold"]}'
        )


def test_judge_command_writes_report_file(tmp_path: Path, monkeypatch) -> None:
    # Minimal workspace layout
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "algorithm.py").write_text("def solve(x):\n    return x\n", encoding="utf-8")
    config = load_config(tmp_path)

    monkeypatch.setattr(commands, "load_config", lambda repo_root: config)
    monkeypatch.setattr(commands, "get_client", lambda config, backend, scenario: _JudgeClient())
    monkeypatch.setattr(commands, "validate_command", lambda repo_root, user_request="": ValidationReport())
    monkeypatch.setattr(commands, "_runtime_capture", lambda repo_root: ("{'success_rate': 0.0}", ""))

    out = tmp_path / ".mecha-agent" / "judge" / "custom.json"
    report = commands.judge_command(tmp_path, "task text", backend="fake", out_path=out, manage_service=False)

    assert report.verdict == "FAIL"
    assert report.suspected_false_positive is True
    assert out.exists()
