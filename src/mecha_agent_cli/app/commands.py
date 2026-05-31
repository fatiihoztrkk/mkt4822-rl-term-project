"""Programmatic command helpers used by the Typer CLI."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from mecha_agent_cli.agent.direct_loop import DirectAgentLoop, MissingPackagePrompt
from mecha_agent_cli.agent.observer import AgentObserver
from mecha_agent_cli.agent.result import AgentRunResult
from mecha_agent_cli.config.loader import load_config
from mecha_agent_cli.config.schema import AppConfig
from mecha_agent_cli.core.constants import DEFAULT_MODEL, FALLBACK_MODEL
from mecha_agent_cli.llm.base import ModelClient
from mecha_agent_cli.llm.fake_client import FakeModelClient
from mecha_agent_cli.llm.ollama_client import OllamaClient
from mecha_agent_cli.llm.ollama_service import managed_ollama_service
from mecha_agent_cli.repo.workspace import initialize_workspace
from mecha_agent_cli.validation.pipeline import ValidationPipeline
from mecha_agent_cli.validation.report import ValidationReport


@dataclass(frozen=True)
class DoctorReport:
    """Environment doctor report."""

    python_ok: bool
    uv_ok: bool
    git_ok: bool
    ruff_ok: bool
    pyright_ok: bool
    pytest_ok: bool
    ollama_ok: bool
    default_model_available: bool
    fallback_model_available: bool
    message: str


@dataclass(frozen=True)
class JudgeReport:
    """Structured post-run LLM adjudication output."""

    verdict: str
    confidence: float
    suspected_false_positive: bool
    reasons: list[str]
    primary_failure_guess: str
    report_path: str
    raw_response: str


class _JudgePayload(TypedDict):
    verdict: str
    confidence: float
    suspected_false_positive: bool
    reasons: list[str]
    primary_failure_guess: str


def get_client(config: AppConfig, backend: str = "ollama", scenario: str = "default") -> ModelClient:
    """Create a model client from a backend name."""
    if backend == "fake":
        return FakeModelClient(scenario=scenario)
    if backend == "ollama":
        return OllamaClient(
            config.models.base_url,
            timeout_sec=config.direct.request_timeout_sec,
            models_config=config.models,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def init_command(repo_root: Path, *, force: bool = False) -> None:
    """Initialize workspace."""
    initialize_workspace(repo_root, force=force)


def run_command(
    repo_root: Path,
    task: str,
    *,
    backend: str = "ollama",
    scenario: str = "default",
    max_attempts: int | None = None,
    until_success: bool | None = None,
    observer: AgentObserver | None = None,
    manage_service: bool = True,
    missing_package_prompt: MissingPackagePrompt | None = None,
) -> AgentRunResult:
    """Run one direct-generation task.

    Each attempt sends the cumulative chat history (system + task + prior
    drafts + validator feedback) to the model, writes the returned fenced
    ```python ... ``` block straight to ``algorithm.py``, and runs a syntax
    + import smoke check. On failure the validator output is appended as the
    next user turn and the model retries.
    """
    config = load_config(repo_root)
    with managed_ollama_service(config.models.base_url, enabled=manage_service and backend == "ollama"):
        client = get_client(config, backend=backend, scenario=scenario)
        return DirectAgentLoop(
            repo_root=repo_root,
            config=config,
            client=client,
            observer=observer,
            max_attempts=max_attempts,
            continue_until_success=until_success,
            missing_package_prompt=missing_package_prompt,
        ).run(task)


def judge_command(
    repo_root: Path,
    task: str,
    *,
    backend: str = "ollama",
    scenario: str = "default",
    manage_service: bool = True,
    out_path: Path | None = None,
) -> JudgeReport:
    """Run a short-form LLM self-judge over generated code and run artifacts.

    The judge is advisory (not a validator authority). It consumes compact
    evidence (task excerpt, validation summary, runtime stdout/stderr, code
    excerpt) and returns a strict JSON verdict.
    """
    config = load_config(repo_root)
    report = validate_command(repo_root, user_request=task)
    runtime_out, runtime_err = _runtime_capture(repo_root)
    target = repo_root / config.target_file
    code_text = target.read_text(encoding="utf-8") if target.exists() else ""

    prompt = _build_judge_prompt(
        task=task,
        validation_summary=report.compact_summary(max_lines=80),
        runtime_stdout=runtime_out,
        runtime_stderr=runtime_err,
        code_excerpt=_truncate_middle(code_text, max_chars=6000),
    )

    with managed_ollama_service(config.models.base_url, enabled=manage_service and backend == "ollama"):
        client = get_client(config, backend=backend, scenario=scenario)
        profile = config.models.profiles.get("critic")
        if profile is None:
            base = config.models.profile("direct")
            data = base.model_dump()
            data.update({"think": False, "temperature": 0.1, "top_p": 0.6, "num_predict": 320})
            data["num_ctx"] = min(int(data.get("num_ctx", 4096)), 4096)
            profile = type(base).model_validate(data)
        raw = client.chat_text(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict software correctness adjudicator. "
                        "Return ONLY valid JSON matching the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            profile=profile,
        )

    parsed = _parse_judge_json(raw)
    report_dir = repo_root / ".mecha-agent" / "judge"
    report_dir.mkdir(parents=True, exist_ok=True)
    validation_summary = report.compact_summary(max_lines=80)
    validator_failed = "primary_failure=UNKNOWN" not in validation_summary
    if validator_failed and parsed["verdict"] == "PASS":
        parsed["verdict"] = "UNSURE"
        parsed["suspected_false_positive"] = True
        parsed["reasons"] = [
            "Judge predicted PASS but validator summary contains failures; downgraded to UNSURE.",
            *parsed["reasons"],
        ][:8]
    save_path = out_path or (report_dir / "latest.json")
    payload = {
        "task": task,
        "validation_summary": validation_summary,
        "runtime_stdout": _truncate_middle(runtime_out, max_chars=2000),
        "runtime_stderr": _truncate_middle(runtime_err, max_chars=2000),
        "judge": parsed,
        "raw_response": raw,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return JudgeReport(
        verdict=parsed["verdict"],
        confidence=parsed["confidence"],
        suspected_false_positive=parsed["suspected_false_positive"],
        reasons=parsed["reasons"],
        primary_failure_guess=parsed["primary_failure_guess"],
        report_path=str(save_path),
        raw_response=raw,
    )


def validate_command(repo_root: Path, *, user_request: str = "") -> ValidationReport:
    """Run validation pipeline."""
    config = load_config(repo_root)
    return ValidationPipeline(config.validation).run(repo_root, user_request=user_request)


def doctor_command(repo_root: Path, *, backend: str = "ollama") -> DoctorReport:
    """Check runtime dependencies and model availability."""
    config = load_config(repo_root)
    python_ok = shutil.which("python") is not None
    uv_ok = shutil.which("uv") is not None
    git_ok = shutil.which("git") is not None
    ruff_ok = shutil.which("ruff") is not None
    pyright_ok = shutil.which("pyright") is not None
    pytest_ok = shutil.which("pytest") is not None
    ollama_ok = False
    default_available = False
    fallback_available = False
    message = ""
    if backend == "fake":
        ollama_ok = True
        default_available = True
        fallback_available = True
        message = "fake backend selected"
    else:
        try:
            with managed_ollama_service(config.models.base_url, enabled=True):
                models = OllamaClient(config.models.base_url).available_models()
                ollama_ok = True
                default_available = DEFAULT_MODEL in models or config.models.default_model in models
                fallback_available = FALLBACK_MODEL in models or config.models.fallback_model in models
                if not default_available:
                    message = f"Missing model. Run: ollama pull {DEFAULT_MODEL}"
        except Exception as exc:
            message = f"Ollama unavailable: {exc}. Run `ollama serve` and `ollama pull {DEFAULT_MODEL}`."
    return DoctorReport(
        python_ok=python_ok,
        uv_ok=uv_ok,
        git_ok=git_ok,
        ruff_ok=ruff_ok,
        pyright_ok=pyright_ok,
        pytest_ok=pytest_ok,
        ollama_ok=ollama_ok,
        default_model_available=default_available,
        fallback_model_available=fallback_available,
        message=message,
    )


def _runtime_capture(repo_root: Path) -> tuple[str, str]:
    """Capture runtime stdout/stderr by executing algorithm.py as __main__."""
    from mecha_agent_cli.sandbox.local_runner import LocalRunner

    snippet = "import runpy, sys; sys.argv=['algorithm.py']; runpy.run_path('algorithm.py', run_name='__main__')"
    result = LocalRunner(max_output_chars=12000).run(["python", "-I", "-c", snippet], repo_root, timeout_sec=60.0)
    return result.stdout, result.stderr


def _truncate_middle(text: str, *, max_chars: int) -> str:
    """Bound prompt payloads while preserving head/tail evidence."""
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + "\n...<truncated>...\n" + text[-keep:]


def _build_judge_prompt(
    *,
    task: str,
    validation_summary: str,
    runtime_stdout: str,
    runtime_stderr: str,
    code_excerpt: str,
) -> str:
    """Build compact evidence prompt for final correctness adjudication."""
    return (
        "Evaluate whether the produced algorithm likely satisfies the task semantically.\n"
        "Return ONLY JSON with this exact schema:\n"
        '{"verdict":"PASS|FAIL|UNSURE","confidence":0.0,"suspected_false_positive":false,'
        '"primary_failure_guess":"SYNTAX|IMPORT|RUNTIME|SEMANTIC|UNKNOWN",'
        '"reasons":["..."]}\n\n'
        "Guidelines:\n"
        "- Be strict about semantic correctness; passing syntax/import/runtime alone is insufficient.\n"
        "- If metrics in runtime output contradict task acceptance thresholds, verdict should be FAIL.\n"
        "- Keep reasons concise and evidence-based.\n\n"
        f"TASK:\n{_truncate_middle(task, max_chars=3000)}\n\n"
        f"VALIDATION_SUMMARY:\n{validation_summary}\n\n"
        f"RUNTIME_STDOUT:\n{_truncate_middle(runtime_stdout, max_chars=2500)}\n\n"
        f"RUNTIME_STDERR:\n{_truncate_middle(runtime_stderr, max_chars=1200)}\n\n"
        f"CODE_EXCERPT:\n{code_excerpt}\n"
    )


def _parse_judge_json(raw: str) -> _JudgePayload:
    """Parse judge JSON, tolerating fenced wrappers or leading prose."""
    text = raw.strip()
    # Strip fenced code wrappers if present.
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        parsed_any = json.loads(text)
    except Exception:
        heuristic = _parse_judge_prose_fallback(raw)
        if heuristic is not None:
            return heuristic
        return {
            "verdict": "UNSURE",
            "confidence": 0.0,
            "suspected_false_positive": True,
            "primary_failure_guess": "UNKNOWN",
            "reasons": ["Judge response was not valid JSON."],
        }
    if not isinstance(parsed_any, dict):
        return {
            "verdict": "UNSURE",
            "confidence": 0.0,
            "suspected_false_positive": True,
            "primary_failure_guess": "UNKNOWN",
            "reasons": ["Judge JSON root was not an object."],
        }
    parsed = cast(dict[str, object], parsed_any)
    # Normalize weak outputs.
    verdict = str(parsed.get("verdict", "UNSURE")).upper()
    if verdict not in {"PASS", "FAIL", "UNSURE"}:
        verdict = "UNSURE"
    confidence_value = parsed.get("confidence", 0.0)
    if isinstance(confidence_value, (int, float)):
        confidence = float(confidence_value)
    elif isinstance(confidence_value, str):
        try:
            confidence = float(confidence_value)
        except Exception:
            confidence = 0.0
    else:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reasons_value = parsed.get("reasons", [])
    if isinstance(reasons_value, list):
        raw_reasons = cast(list[object], reasons_value)
        reasons = [str(item) for item in raw_reasons][:8]
    else:
        reasons = [str(reasons_value)]
    return {
        "verdict": verdict,
        "confidence": confidence,
        "suspected_false_positive": bool(parsed.get("suspected_false_positive", False)),
        "primary_failure_guess": str(parsed.get("primary_failure_guess", "UNKNOWN")),
        "reasons": reasons,
    }


def _parse_judge_prose_fallback(raw: str) -> _JudgePayload | None:
    """Parse common prose-style judge outputs when strict JSON is not respected."""
    text = raw.strip()
    verdict_match = re.search(
        r"verdict\s*[:=]\s*\"?(PASS|FAIL|UNSURE)\"?(?!\|)(?=\s|,|\}|$)",
        text,
        flags=re.IGNORECASE,
    )
    if verdict_match is None:
        return None
    verdict = verdict_match.group(1).upper()

    confidence = 0.0
    conf_match = re.search(r"confidence\s*[:=]\s*\"?([01](?:\.\d+)?)\"?", text, flags=re.IGNORECASE)
    if conf_match is not None:
        try:
            confidence = float(conf_match.group(1))
        except Exception:
            confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    false_pos = True
    fp_match = re.search(r"suspected_false_positive\s*[:=]\s*\"?(true|false)\"?", text, flags=re.IGNORECASE)
    if fp_match is not None:
        false_pos = fp_match.group(1).lower() == "true"

    primary = "UNKNOWN"
    primary_match = re.search(
        r"primary_failure_guess\s*[:=]\s*\"?(SYNTAX|IMPORT|RUNTIME|SEMANTIC|UNKNOWN)\"?",
        text,
        flags=re.IGNORECASE,
    )
    if primary_match is not None:
        primary = primary_match.group(1).upper()

    reasons = ["Parsed prose judge response (non-JSON)."]
    if "missing" in text.lower() and "function" in text.lower():
        reasons = ["Required functions appear missing according to judge explanation."]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "suspected_false_positive": false_pos,
        "primary_failure_guess": primary,
        "reasons": reasons,
    }
