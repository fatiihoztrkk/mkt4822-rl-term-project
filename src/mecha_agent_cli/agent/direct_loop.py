"""Direct, pythalab-cortex-style code-generation loop for ``algorithm.py``.

The model is asked, in plain chat, to produce the complete contents of a single
file. Whatever comes back gets written straight to ``algorithm.py``. We do not
gate the model output behind AST safety filters, semantic checklists, ruff,
pyright, or pytest — only a syntax + import smoke check (mirroring
pythalab-cortex's validation surface).

If the smoke check fails we feed the error back as the next user turn and let
the model try again, accumulating chat history so attempt N sees attempts
1..N-1 plus their validator feedback.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast
from uuid import uuid4

from mecha_agent_cli.agent.observer import AgentObserver, AgentProgressEvent, NullObserver
from mecha_agent_cli.agent.result import AgentRunResult
from mecha_agent_cli.config.schema import AppConfig
from mecha_agent_cli.core.errors import ModelError
from mecha_agent_cli.core.types import FailureType
from mecha_agent_cli.learning.arm_registry import Arm
from mecha_agent_cli.learning.bandit import BanditStore, ThompsonBandit
from mecha_agent_cli.learning.context import build_context_key
from mecha_agent_cli.learning.reward import episode_reward
from mecha_agent_cli.llm.base import ChatMessage, ModelClient
from mecha_agent_cli.llm.code_extractor import ExtractedCode, extract_python_code
from mecha_agent_cli.validation.pipeline import ValidationPipeline
from mecha_agent_cli.validation.report import ValidationReport, ValidationResult

MissingPackagePrompt = Callable[[set[str]], bool]
"""Hook invoked when the import-check reports missing modules.

The callable receives the set of missing top-level *import* names (e.g.
``{"matplotlib", "numpy"}``) and must return ``True`` to install them into the
active Python environment, or ``False`` to leave the failure intact.
"""

# Map top-level import names to PyPI distribution names where they differ.
_IMPORT_TO_PYPI: dict[str, str] = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "skimage": "scikit-image",
}

_DIRECT_SYSTEM_PROMPT = (
    "You are a precise senior Python engineer. You write the complete contents of a "
    "single file named ``algorithm.py``.\n\n"
    "Code-quality rules (MANDATORY):\n"
    "  - Always wrap your answer in exactly ONE fenced ```python ... ``` block. No prose "
    "outside the block.\n"
    "  - Output the COMPLETE updated contents of algorithm.py, from the first line to the "
    'last. Do not write "... unchanged ..." or use ellipses.\n'
    "  - Include all required imports at the top of the file.\n"
    "  - Python 3.11+. Use modern builtin generics (``list[int]``, ``dict[str, object]``).\n"
    "  - Include type hints on public functions and concise docstrings.\n"
    "  - The code MUST be syntactically valid and importable as-is.\n"
    "  - Do not use placeholders such as ``...``, ``# TODO``, ``pass  # implement``, or "
    "``raise NotImplementedError``.\n\n"
    "Dependency policy:\n"
    "  - Always satisfy the user's request as written. If the user asks for ``matplotlib``, "
    "``numpy``, ``pandas``, ``scipy``, ``sklearn``, ``torch``, ``tensorflow``, etc., import "
    "and use them normally. Do NOT refuse, do NOT silently downgrade to a stdlib-only "
    "version, and do NOT replace plotting with CSV writes unless the user asked for that.\n"
    "  - When a third-party package is genuinely missing from the environment, the runtime "
    "will detect ``ModuleNotFoundError`` and offer to install it. Your job is to write the "
    "correct code; the runtime handles installation.\n"
    "  - Prefer the standard library only when no third-party dependency is required by "
    "the task. Pick well-known, pip-installable packages with stable import names.\n\n"
    "Iteration rules:\n"
    "  - When earlier assistant turns are present, treat them as your own prior drafts. "
    "Build on them: keep what works, fix what the validator complained about, and "
    "extend the file to satisfy the latest user instructions.\n"
    "  - Never delete previously-correct code unless the new request explicitly contradicts it.\n\n"
    "Execution model (IMPORTANT):\n"
    "  - After each generation the runtime executes ``algorithm.py`` as ``__main__`` and "
    "captures any uncaught exception. So your file MUST run end-to-end without crashing.\n"
    "  - Provide a top-level ``if __name__ == '__main__':`` block that exercises the "
    "primary entry point with realistic inputs (small but non-trivial: e.g. a 2-step "
    "simulation, a tiny example dataset). Do not leave it empty and do not call "
    "``sys.exit`` or ``input()``.\n"
    "  - Validate matrix/array shapes, units, and indices before using them. When you "
    "slice state vectors (``x[:n]``, ``x[n:]``), make sure the resulting shapes are what "
    "the consuming function expects (e.g. an inertia matrix ``M @ v`` requires ``v`` to "
    "have the matching column count).\n"
    "  - Guard divisions, square roots, ``log``, matrix inversions and similar against "
    "degenerate inputs. Prefer a small example that you have mentally executed.\n"
    "  - If a runtime error is reported, READ THE TRACEBACK: the failing line, the "
    "operator, and the operand shapes/values are the actual fix target. Do not just "
    "reshuffle code — fix the specific operation that crashed."
)


@dataclass(frozen=True)
class _AttemptOutcome:
    raw_response: str
    extracted: ExtractedCode | None
    code_written: bool
    snapshot_path: Path | None
    report: ValidationReport
    feedback_for_next_turn: str


@dataclass(frozen=True)
class _JudgeDecision:
    accepted: bool
    verdict: str
    confidence: float
    suspected_false_positive: bool
    reasons: list[str]
    feedback_for_next_turn: str


class _JudgePayload(TypedDict):
    verdict: str
    confidence: float
    suspected_false_positive: bool
    primary_failure_guess: str
    reasons: list[str]


class DirectAgentLoop:
    """Stable, serial, cumulative direct-to-algorithm.py generation loop."""

    def __init__(
        self,
        *,
        repo_root: Path,
        config: AppConfig,
        client: ModelClient,
        session_id: str | None = None,
        observer: AgentObserver | None = None,
        max_attempts: int | None = None,
        continue_until_success: bool | None = None,
        missing_package_prompt: MissingPackagePrompt | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.config = config
        self.client = client
        self.session_id = session_id or uuid4().hex
        self.observer: AgentObserver = observer or NullObserver()
        self.max_attempts_override = max_attempts
        self.continue_until_success = (
            config.agent.continue_until_success if continue_until_success is None else continue_until_success
        )
        self.pipeline = ValidationPipeline(config.validation)
        self.missing_package_prompt = missing_package_prompt
        self._task_counter = 0
        self._bandit: ThompsonBandit | None = self._build_bandit()
        # Per-episode RL state.
        self._active_arm: Arm | None = None
        self._base_context_key: str = ""
        self._previous_report: ValidationReport | None = None
        self._arm_history: list[str] = []

    def _build_bandit(self) -> ThompsonBandit | None:
        learning = self.config.learning
        if not learning.enabled:
            return None
        db_path = self.repo_root / self.config.memory.path if learning.persist else None
        return ThompsonBandit(BanditStore(db_path), learning)

    # -- Public entry point ------------------------------------------------

    def run(self, user_request: str) -> AgentRunResult:
        """Execute the direct generation loop for ``user_request`` and return a result."""
        target_file = self.repo_root / self.config.target_file
        model_name = self.config.models.default_model
        max_attempts = self._effective_max_attempts()
        task_id = self._next_task_id()
        episode_started = time.perf_counter()

        self._emit("preflight", "running", "direct generation loop starting; serial single-GPU mode")
        baseline_text = target_file.read_text(encoding="utf-8") if target_file.exists() else ""
        context_key = build_context_key(
            user_request=user_request,
            model_name=model_name,
            has_baseline=bool(baseline_text.strip()),
        )
        self._base_context_key = context_key
        self._previous_report = None
        self._arm_history = []
        messages: list[ChatMessage] = [
            {"role": "system", "content": _DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": self._initial_user_prompt(user_request, baseline_text)},
        ]
        attempt_limit = "∞" if max_attempts <= 0 else str(max_attempts)
        self._emit(
            "preflight",
            "done",
            f"target={self.config.target_file} model={model_name} max_attempts={attempt_limit}",
        )
        snapshot_paths: list[str] = []
        best_report: ValidationReport | None = None
        best_score = 0.0
        terminal_report: ValidationReport | None = None
        total_attempts = 0
        last_feedback = ""
        judge_repairs = 0

        try:
            while self._budget_available(total_attempts, max_attempts):
                stage_label = "generate" if total_attempts == 0 else "regenerate"
                self._emit(
                    stage_label,
                    "running",
                    f"requesting attempt {total_attempts + 1} from {model_name}",
                    attempt_index=total_attempts,
                    total_attempts=total_attempts,
                    max_attempts=max_attempts,
                )
                outcome = self._one_attempt(
                    task_id=task_id,
                    attempt_index=total_attempts,
                    user_request=user_request,
                    target_file=target_file,
                    messages=messages,
                )
                if outcome.snapshot_path is not None:
                    snapshot_paths.append(str(outcome.snapshot_path.relative_to(self.repo_root)))
                best_report, best_score = self._track_best(best_report, best_score, outcome.report)
                terminal_report = outcome.report
                total_attempts += 1

                if outcome.report.passed:
                    judge_decision = self._judge_after_validation_success(
                        user_request=user_request,
                        target_file=target_file,
                        report=outcome.report,
                        attempt_index=total_attempts - 1,
                    )
                    if judge_decision is not None and not judge_decision.accepted:
                        judge_repairs += 1
                        self._emit(
                            "judge",
                            "fail",
                            (
                                f"verdict={judge_decision.verdict} confidence={judge_decision.confidence:.2f} "
                                f"false_positive={judge_decision.suspected_false_positive}"
                            ),
                            attempt_index=total_attempts - 1,
                        )
                        judge_report = self._failure_report(
                            "judge_gate",
                            "Judge rejected validator-pass candidate: " + "; ".join(judge_decision.reasons[:3]),
                            FailureType.SEMANTIC,
                        )
                        terminal_report = judge_report
                        if judge_repairs > self.config.direct.judge_max_repairs:
                            return self._finalize(
                                context_key,
                                self._budget_exhausted(
                                    task_id=task_id,
                                    report=judge_report,
                                    snapshot_paths=snapshot_paths,
                                    total_attempts=total_attempts,
                                    max_attempts=max_attempts,
                                    last_feedback=judge_decision.feedback_for_next_turn,
                                ),
                                episode_started,
                            )
                        messages.append({"role": "assistant", "content": outcome.raw_response})
                        messages.append({"role": "user", "content": judge_decision.feedback_for_next_turn})
                        messages = self._truncate_history(messages, self.config.direct.max_history_chars)
                        last_feedback = judge_decision.feedback_for_next_turn
                        continue
                    return self._finalize(
                        context_key,
                        self._success(
                            task_id=task_id,
                            report=outcome.report,
                            total_attempts=total_attempts,
                            max_attempts=max_attempts,
                            snapshot_paths=snapshot_paths,
                        ),
                        episode_started,
                    )

                # Persist this round into chat history so the next turn sees the
                # model's last raw output and the validator's complaint.
                messages.append({"role": "assistant", "content": outcome.raw_response})
                messages.append({"role": "user", "content": outcome.feedback_for_next_turn})
                messages = self._truncate_history(messages, self.config.direct.max_history_chars)
                last_feedback = outcome.feedback_for_next_turn

            return self._finalize(
                context_key,
                self._budget_exhausted(
                    task_id=task_id,
                    report=terminal_report or best_report,
                    snapshot_paths=snapshot_paths,
                    total_attempts=total_attempts,
                    max_attempts=max_attempts,
                    last_feedback=last_feedback,
                ),
                episode_started,
            )
        except ModelError as exc:
            return self._finalize(
                context_key,
                self._error_result(
                    task_id=task_id,
                    exc=exc,
                    snapshot_paths=snapshot_paths,
                    total_attempts=total_attempts,
                    max_attempts=max_attempts,
                ),
                episode_started,
            )

    def _judge_after_validation_success(
        self,
        *,
        user_request: str,
        target_file: Path,
        report: ValidationReport,
        attempt_index: int,
    ) -> _JudgeDecision | None:
        """Optionally run short-form judge after validator pass.

        Returns:
        - ``None`` when auto-judge is disabled or judge call fails unexpectedly.
        - ``_JudgeDecision`` when judge completes.
        """
        if not self.config.direct.auto_judge_after_validation:
            return None
        try:
            code_text = target_file.read_text(encoding="utf-8") if target_file.exists() else ""
        except OSError:
            code_text = ""
        runtime_result = report.by_name("runtime")
        runtime_stdout = runtime_result.stdout_excerpt if runtime_result is not None else ""
        runtime_stderr = runtime_result.stderr_excerpt if runtime_result is not None else ""
        prompt = self._judge_prompt(
            task=user_request,
            validation_summary=report.compact_summary(max_lines=80),
            runtime_stdout=runtime_stdout,
            runtime_stderr=runtime_stderr,
            code_excerpt=self._truncate_middle(code_text, max_chars=self.config.direct.judge_prompt_max_chars),
        )
        base = self.config.models.profile(self.config.direct.profile_name)
        profile = self.config.models.profiles.get(self.config.direct.judge_profile_name)
        if profile is None:
            data = base.model_dump()
            data.update({"think": False, "temperature": 0.1, "top_p": 0.6, "num_predict": 320})
            data["num_ctx"] = min(int(data.get("num_ctx", 4096)), 4096)
            profile = type(base).model_validate(data)
        try:
            self._emit("judge", "running", "running short-form semantic adjudication", attempt_index=attempt_index)
            raw = self.client.chat_text(
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
        except Exception as exc:
            self._emit("judge", "skip", f"judge skipped due to model error: {exc}", attempt_index=attempt_index)
            return None
        parsed = self._parse_judge_json(raw)
        verdict = parsed["verdict"].upper()
        confidence = parsed["confidence"]
        suspected_false_positive = parsed["suspected_false_positive"]
        reasons = parsed["reasons"]
        # Only block validator-pass on an EXPLICIT high-confidence FAIL.
        # PASS / UNSURE / unparseable judge responses must not override the
        # validator pipeline (e.g. fake backends or judge timeouts).
        accepted = not (verdict == "FAIL" and confidence >= self.config.direct.judge_min_confidence)
        feedback = (
            "Post-validation judge flagged semantic issues despite passing validators. "
            "Repair the code and keep all currently passing checks green.\n\n"
            f"Judge verdict={verdict}, confidence={confidence:.2f}, "
            f"suspected_false_positive={suspected_false_positive}.\n"
            "Judge reasons:\n"
            + "\n".join(f"- {r}" for r in reasons)
            + "\n\nOutput the COMPLETE updated algorithm.py inside one ```python ... ``` block."
        )
        self._emit(
            "judge",
            "done",
            f"verdict={verdict} confidence={confidence:.2f} accepted={accepted}",
            attempt_index=attempt_index,
        )
        return _JudgeDecision(
            accepted=accepted,
            verdict=verdict,
            confidence=confidence,
            suspected_false_positive=suspected_false_positive,
            reasons=reasons,
            feedback_for_next_turn=feedback,
        )

    @staticmethod
    def _truncate_middle(text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        keep = max_chars // 2
        return text[:keep] + "\n...<truncated>...\n" + text[-keep:]

    def _judge_prompt(
        self,
        *,
        task: str,
        validation_summary: str,
        runtime_stdout: str,
        runtime_stderr: str,
        code_excerpt: str,
    ) -> str:
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
            f"TASK:\n{self._truncate_middle(task, max_chars=3000)}\n\n"
            f"VALIDATION_SUMMARY:\n{validation_summary}\n\n"
            f"RUNTIME_STDOUT:\n{self._truncate_middle(runtime_stdout, max_chars=2000)}\n\n"
            f"RUNTIME_STDERR:\n{self._truncate_middle(runtime_stderr, max_chars=1200)}\n\n"
            f"CODE_EXCERPT:\n{code_excerpt}\n"
        )

    @staticmethod
    def _parse_judge_json(raw: str) -> _JudgePayload:
        text = raw.strip()
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
            parsed_any = None
        if not isinstance(parsed_any, dict):
            return {
                "verdict": "UNSURE",
                "confidence": 0.0,
                "suspected_false_positive": True,
                "primary_failure_guess": "UNKNOWN",
                "reasons": ["Judge response was not valid JSON."],
            }
        parsed: dict[str, object] = cast(dict[str, object], parsed_any)
        verdict = str(parsed.get("verdict", "UNSURE")).upper()
        if verdict not in {"PASS", "FAIL", "UNSURE"}:
            verdict = "UNSURE"
        conf_value = parsed.get("confidence", 0.0)
        if isinstance(conf_value, (int, float)):
            confidence = float(conf_value)
        elif isinstance(conf_value, str):
            try:
                confidence = float(conf_value)
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

    # -- Single attempt ---------------------------------------------------

    def _one_attempt(
        self,
        *,
        task_id: int,
        attempt_index: int,
        user_request: str,
        target_file: Path,
        messages: list[ChatMessage],
    ) -> _AttemptOutcome:
        attempt_context = self._attempt_context_key(
            base_context_key=self._base_context_key,
            attempt_index=attempt_index,
        )
        self._active_arm = self._select_arm(attempt_context)
        if self._active_arm is not None:
            self._arm_history.append(self._active_arm.arm_id)
        base_profile = self.config.models.profile(self.config.direct.profile_name)
        profile = self._active_arm.apply(base_profile) if self._active_arm is not None else base_profile

        def _emit_thinking(chunk: str) -> None:
            self._emit("thinking", "stream", chunk, attempt_index=attempt_index)

        raw = self.client.chat_text(messages=messages, profile=profile, on_thinking=_emit_thinking)
        self._emit(
            "generate",
            "done",
            f"model returned {len(raw)} chars",
            attempt_index=attempt_index,
        )
        extracted = extract_python_code(raw)
        if extracted is None or not extracted.code.strip():
            self._emit(
                "extract",
                "fail",
                "model output had no usable ```python``` block; will retry with stronger format hint",
                attempt_index=attempt_index,
            )
            report = self._failure_report("extract", "no python code block in model response", FailureType.SEMANTIC)
            self._bandit_update_attempt(
                context_key=attempt_context,
                arm=self._active_arm,
                report=report,
                attempt_index=attempt_index,
            )
            self._previous_report = report
            return _AttemptOutcome(
                raw_response=raw,
                extracted=None,
                code_written=False,
                snapshot_path=None,
                report=report,
                feedback_for_next_turn=(
                    "Your previous response did not contain a valid ```python ... ``` code "
                    "block. Output the COMPLETE updated algorithm.py inside exactly one such "
                    "fenced block. No prose, no diff, no patches."
                ),
            )

        # Materialize directly to algorithm.py — no path policy, no safety gate.
        target_file.write_text(extracted.code, encoding="utf-8")
        snapshot = self._save_snapshot(task_id, attempt_index, extracted.code)

        self._emit(
            "validate",
            "running",
            f"wrote algorithm.py ({len(extracted.code)} chars); running validators",
            attempt_index=attempt_index,
        )
        report = self.pipeline.run(self.repo_root, user_request=user_request)
        report = self._maybe_install_missing_packages(report, user_request, attempt_index)
        self._bandit_update_attempt(
            context_key=attempt_context,
            arm=self._active_arm,
            report=report,
            attempt_index=attempt_index,
        )
        self._previous_report = report
        if report.passed:
            self._emit(
                "validate",
                "success",
                f"attempt {attempt_index + 1} passed",
                attempt_index=attempt_index,
            )
            return _AttemptOutcome(
                raw_response=raw,
                extracted=extracted,
                code_written=True,
                snapshot_path=snapshot,
                report=report,
                feedback_for_next_turn="",
            )

        self._emit(
            "validate",
            "fail",
            f"primary_failure={report.primary_failure.value}; feeding validator output back",
            attempt_index=attempt_index,
        )
        feedback = self._format_validator_feedback(report)
        return _AttemptOutcome(
            raw_response=raw,
            extracted=extracted,
            code_written=True,
            snapshot_path=snapshot,
            report=report,
            feedback_for_next_turn=feedback,
        )

    # -- Helpers ----------------------------------------------------------

    def _initial_user_prompt(self, user_request: str, current_file: str) -> str:
        if current_file.strip():
            return (
                f"Task: {user_request}\n\n"
                "Current contents of algorithm.py (treat as a starting point you may rewrite):\n"
                "```python\n"
                f"{current_file}\n"
                "```\n\n"
                "Respond with the COMPLETE updated algorithm.py inside one ```python ... ``` block."
            )
        return (
            f"Task: {user_request}\n\n"
            "algorithm.py does not exist yet. Create the COMPLETE file from scratch and return "
            "it inside one ```python ... ``` block."
        )

    def _maybe_install_missing_packages(
        self,
        report: ValidationReport,
        user_request: str,
        attempt_index: int,
    ) -> ValidationReport:
        """Optionally install missing third-party packages, then re-run validation.

        Returns the original report unchanged when no prompt is wired, when the
        primary failure is not an import error, when no missing modules can be
        parsed, when the user declines, or when the install fails.
        """
        if self.missing_package_prompt is None:
            return report
        if report.primary_failure is not FailureType.IMPORT:
            return report
        summary = report.compact_summary(max_lines=200)
        missing = _extract_missing_modules(summary)
        if not missing:
            return report
        try:
            approved = bool(self.missing_package_prompt(missing))
        except (KeyboardInterrupt, EOFError):
            approved = False
        if not approved:
            return report
        pip_names = sorted({_IMPORT_TO_PYPI.get(name, name) for name in missing})
        self._emit(
            "install",
            "running",
            f"installing {', '.join(pip_names)} into {sys.executable}",
            attempt_index=attempt_index,
        )
        installed = _pip_install(pip_names, cwd=self.repo_root)
        if not installed.ok:
            self._emit(
                "install",
                "fail",
                f"pip install failed (exit={installed.exit_code}): {installed.stderr_tail}",
                attempt_index=attempt_index,
            )
            return report
        self._emit(
            "install",
            "done",
            f"installed {', '.join(pip_names)}; re-running validators",
            attempt_index=attempt_index,
        )
        return self.pipeline.run(self.repo_root, user_request=user_request)

    def _format_validator_feedback(self, report: ValidationReport) -> str:
        max_lines = self.config.direct.error_summary_max_lines
        summary = report.compact_summary(max_lines=max_lines)
        directive = self._actionable_directive(report, summary)
        return (
            "Validators failed on your previous version of algorithm.py. "
            "Fix the issues listed below while keeping every other working part of the file. "
            "Output the COMPLETE updated algorithm.py inside one ```python ... ``` block.\n\n"
            f"{directive}"
            "VALIDATOR REPORT:\n"
            f"{summary}"
        )

    @staticmethod
    def _actionable_directive(report: ValidationReport, summary: str) -> str:
        """Return an extra instruction targeted at the dominant failure mode.

        The default validator summary is too generic for small models — they
        often rationalise and resubmit identical code. This injects a concrete
        "do this, not that" hint above the raw report.
        """
        # Spec-contract failure: required functions missing or wrong arity.
        if report.primary_failure is FailureType.SEMANTIC and any(
            r.name == "spec_contract" and not r.passed for r in report.results
        ):
            contract_line = next(
                (r.stderr_excerpt for r in report.results if r.name == "spec_contract" and not r.passed),
                "",
            )
            return (
                "REQUIRED FIX: The spec-contract check parsed the user's prompt and found "
                "that one or more REQUIRED public functions are MISSING from algorithm.py "
                "or have the wrong number of positional parameters. The runtime smoke "
                "test is NOT enough — every function the prompt declares with a "
                "``def NAME(...)`` line must exist at module scope with at least the "
                "stated arity. Define each missing function fully (no stubs, no "
                "``pass``, no ``raise NotImplementedError``); make ``main()`` actually "
                "call them. Re-emitting the previous file will fail identically.\n\n"
                f"{contract_line}\n\n"
            )
        if report.primary_failure is FailureType.IMPORT:
            missing = _extract_missing_modules(summary)
            if missing:
                joined = ", ".join(sorted(missing))
                return (
                    "REQUIRED FIX: The validator's import smoke check failed because the "
                    f"following package(s) are NOT installed: {joined}. The validator runs "
                    "in a minimal venv with ONLY the Python standard library. You MUST "
                    "remove every ``import`` and ``from`` statement that references those "
                    "packages and rewrite the affected logic using stdlib alternatives "
                    "(math, statistics, csv, json, random, ...). Re-emitting the same "
                    "imports will fail identically.\n\n"
                )
            return (
                "REQUIRED FIX: The import smoke check failed. Re-read every top-level "
                "import and remove any that refer to packages outside the Python standard "
                "library, then rewrite the affected logic with stdlib only.\n\n"
            )
        if report.primary_failure is FailureType.RUNTIME:
            tail = _extract_traceback_tail(summary)
            return (
                "REQUIRED FIX: The runtime smoke check ran ``algorithm.py`` as ``__main__`` "
                "and an uncaught exception was raised. This is NOT an environment problem — "
                "the file you produced crashes. Fix the SPECIFIC failing line shown in the "
                "traceback below: check the operator, the operand shapes/values/types, the "
                "indices and slicing. Trace what each variable contains at that point and "
                "correct the construction. Re-emitting the same buggy expression will fail "
                "identically.\n\n"
                f"TRACEBACK TAIL:\n{tail}\n\n"
            )
        return ""

    def _truncate_history(self, messages: list[ChatMessage], max_chars: int) -> list[ChatMessage]:
        if len(messages) <= 3:
            return messages
        total = sum(len(m["content"]) for m in messages)
        if total <= max_chars:
            return messages
        head = messages[:2]
        tail = list(messages[2:])
        while tail and sum(len(m["content"]) for m in head + tail) > max_chars and len(tail) > 2:
            tail.pop(0)
        return head + tail

    def _save_snapshot(self, task_id: int, attempt_index: int, code: str) -> Path | None:
        if not self.config.direct.save_attempt_snapshots:
            return None
        attempts_dir = self.repo_root / ".mecha-agent" / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        path = attempts_dir / f"task-{task_id:06d}-attempt-{attempt_index:03d}.py"
        path.write_text(code, encoding="utf-8")
        return path

    def _track_best(
        self,
        best: ValidationReport | None,
        best_score: float,
        candidate: ValidationReport,
    ) -> tuple[ValidationReport, float]:
        if best is None or candidate.total_score >= best_score:
            return candidate, candidate.total_score
        return best, best_score

    def _effective_max_attempts(self) -> int:
        if self.continue_until_success:
            return 0
        if self.max_attempts_override is not None and self.max_attempts_override > 0:
            return self.max_attempts_override
        return self.config.direct.max_attempts

    def _budget_available(self, total_attempts: int, max_attempts: int) -> bool:
        return max_attempts <= 0 or total_attempts < max_attempts

    def _next_task_id(self) -> int:
        self._task_counter += 1
        salt = int(hashlib.sha256(str(self.repo_root).encode()).hexdigest()[:8], 16) % 1000
        return salt * 1000 + self._task_counter

    def _select_arm(self, context_key: str) -> Arm | None:
        """Pick the arm for this episode and emit a ``bandit_select`` event."""
        if self._bandit is None:
            return None
        arm = self._bandit.select(context_key)
        self._emit(
            "bandit_select",
            "done",
            f"context={context_key} arm={arm.arm_id}",
        )
        return arm

    @staticmethod
    def _attempt_failure_token(report: ValidationReport | None) -> str:
        if report is None:
            return "none"
        return report.primary_failure.value.lower()

    @staticmethod
    def _report_passed_checks(report: ValidationReport | None) -> int:
        if report is None:
            return 0
        return sum(1 for r in report.results if r.passed and not r.skipped)

    @staticmethod
    def _report_duration_bucket(report: ValidationReport | None) -> str:
        if report is None:
            return "na"
        duration = sum(max(0.0, float(r.duration_sec)) for r in report.results)
        if duration < 1.0:
            return "fast"
        if duration < 3.0:
            return "mid"
        return "slow"

    def _attempt_context_key(self, *, base_context_key: str, attempt_index: int) -> str:
        """Build a compact per-attempt context key for phase-2/3 RL.

        This keeps cardinality bounded while exposing the strongest signals:
        previous failure type, previous progress, and previous validator latency.
        """
        fail = self._attempt_failure_token(self._previous_report)
        prog = min(self._report_passed_checks(self._previous_report), 5)
        dur = self._report_duration_bucket(self._previous_report)
        a_bucket = str(min(attempt_index, 3)) if attempt_index < 3 else "3p"
        return f"{base_context_key}|a{a_bucket}|pf:{fail}|p:{prog}|d:{dur}"

    def _attempt_reward(self, *, report: ValidationReport, attempt_index: int) -> float:
        """Return per-attempt reward in ``[-1, 1]`` for online bandit updates."""
        cfg = self.config.learning
        success = report.passed
        progress = self._report_passed_checks(report)
        base = cfg.success_reward if success else 0.0
        progress_credit = 0.0 if success else cfg.progress_bonus * float(progress)
        attempt_cost = cfg.attempt_penalty * float(attempt_index)
        extract_fail = report.primary_failure is FailureType.SEMANTIC and any(
            r.name == "extract" and not r.passed for r in report.results
        )
        extract_cost = cfg.extract_failure_penalty if extract_fail else 0.0
        behavior_fail = report.primary_failure is FailureType.SEMANTIC and any(
            r.name == "behavior" and not r.passed for r in report.results
        )
        behavior_cost = cfg.behavior_failure_penalty if behavior_fail else 0.0
        duration = sum(max(0.0, float(r.duration_sec)) for r in report.results)
        if cfg.latency_penalty > 0.0 and cfg.latency_horizon_sec > 0.0 and duration > 0.0:
            latency_cost = cfg.latency_penalty * min(1.0, duration / cfg.latency_horizon_sec)
        else:
            latency_cost = 0.0
        raw = base + progress_credit - attempt_cost - extract_cost - behavior_cost - latency_cost
        return max(-1.0, min(1.0, raw))

    def _bandit_update_attempt(
        self,
        *,
        context_key: str,
        arm: Arm | None,
        report: ValidationReport,
        attempt_index: int,
    ) -> None:
        if self._bandit is None or arm is None:
            return
        reward = self._attempt_reward(report=report, attempt_index=attempt_index)
        success = report.passed
        stat = self._bandit.update(
            context_key=context_key,
            arm_id=arm.arm_id,
            reward=reward,
            success=success,
        )
        self._emit(
            "bandit_update",
            "done",
            (
                f"attempt={attempt_index + 1} ctx={context_key} arm={arm.arm_id} "
                f"reward={reward:+.3f} pulls={stat.pulls} "
                f"alpha={stat.alpha:.2f} beta={stat.beta:.2f} mean={stat.mean:.3f}"
            ),
            attempt_index=attempt_index,
        )

    def _finalize(self, context_key: str, result: AgentRunResult, episode_started: float) -> AgentRunResult:
        """Stamp duration and finalize strategy metadata."""
        duration_sec = max(0.0, time.perf_counter() - episode_started)
        # Always reflect duration on the result so callers (benchmarks, UI)
        # can read it regardless of whether learning is enabled.
        result_with_duration = AgentRunResult(
            task_id=result.task_id,
            status=result.status,
            changed_files=result.changed_files,
            validation_report=result.validation_report,
            review_summary=result.review_summary,
            reward=result.reward,
            repair_attempts=result.repair_attempts,
            strategy_name=result.strategy_name,
            attempt_snapshots=result.attempt_snapshots,
            total_attempts=result.total_attempts,
            max_attempts=result.max_attempts,
            duration_sec=duration_sec,
        )
        if self._bandit is None:
            return result_with_duration
        reward = episode_reward(result_with_duration, self.config.learning)
        strategy_suffix = "->".join(self._arm_history) if self._arm_history else "none"
        return AgentRunResult(
            task_id=result_with_duration.task_id,
            status=result_with_duration.status,
            changed_files=result_with_duration.changed_files,
            validation_report=result_with_duration.validation_report,
            review_summary=result_with_duration.review_summary,
            reward=reward,
            repair_attempts=result_with_duration.repair_attempts,
            strategy_name=f"bandit:{strategy_suffix}",
            attempt_snapshots=result_with_duration.attempt_snapshots,
            total_attempts=result_with_duration.total_attempts,
            max_attempts=result_with_duration.max_attempts,
            duration_sec=duration_sec,
        )

    def _emit(
        self,
        stage: str,
        status: str,
        detail: str = "",
        *,
        attempt_index: int | None = None,
        total_attempts: int | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self.observer(
            AgentProgressEvent(
                stage=stage,
                status=status,
                detail=detail,
                attempt_index=attempt_index,
                total_attempts=total_attempts,
                max_attempts=max_attempts,
            )
        )

    def _success(
        self,
        *,
        task_id: int,
        report: ValidationReport,
        total_attempts: int,
        max_attempts: int,
        snapshot_paths: list[str],
    ) -> AgentRunResult:
        self._emit(
            "complete",
            "success",
            f"algorithm.py validated after {total_attempts} attempt(s)",
            total_attempts=total_attempts,
            max_attempts=max_attempts,
        )
        return AgentRunResult(
            task_id=task_id,
            status="success",
            changed_files=[self.config.target_file],
            validation_report=report,
            review_summary=f"Direct generation succeeded in {total_attempts} attempt(s).",
            attempt_snapshots=snapshot_paths,
            total_attempts=total_attempts,
            max_attempts=max_attempts,
            repair_attempts=max(total_attempts - 1, 0),
        )

    def _budget_exhausted(
        self,
        *,
        task_id: int,
        report: ValidationReport | None,
        snapshot_paths: list[str],
        total_attempts: int,
        max_attempts: int,
        last_feedback: str,
    ) -> AgentRunResult:
        self._emit(
            "complete",
            "failed",
            f"stopped after {total_attempts} attempt(s)",
            total_attempts=total_attempts,
            max_attempts=max_attempts,
        )
        final_report = report or self._failure_report(
            "attempt_budget", "no validation report produced", FailureType.UNKNOWN
        )
        return AgentRunResult(
            task_id=task_id,
            status="attempt_budget_exhausted",
            changed_files=[],
            validation_report=final_report,
            review_summary=last_feedback or "Attempt budget exhausted before validation passed.",
            attempt_snapshots=snapshot_paths,
            total_attempts=total_attempts,
            max_attempts=max_attempts,
            repair_attempts=max(total_attempts - 1, 0),
        )

    def _error_result(
        self,
        *,
        task_id: int,
        exc: Exception,
        snapshot_paths: list[str],
        total_attempts: int,
        max_attempts: int,
    ) -> AgentRunResult:
        report = self._failure_report("client_error", str(exc), FailureType.UNKNOWN)
        self._emit("complete", "error", str(exc), total_attempts=total_attempts, max_attempts=max_attempts)
        return AgentRunResult(
            task_id=task_id,
            status="client_error",
            changed_files=[],
            validation_report=report,
            review_summary=str(exc),
            attempt_snapshots=snapshot_paths,
            total_attempts=total_attempts,
            max_attempts=max_attempts,
        )

    def _failure_report(self, name: str, message: str, failure_type: FailureType) -> ValidationReport:
        result = ValidationResult(
            name=name,
            command=[],
            exit_code=1,
            stdout_excerpt="",
            stderr_excerpt=message,
            passed=False,
            duration_sec=0.0,
            failure_type=failure_type,
        )
        return ValidationReport(
            results=[result],
            semantic_score=0.0,
            total_score=0.0,
            primary_failure=failure_type,
        )


def reset_attempts_dir(repo_root: Path) -> None:
    """Remove cached per-attempt code snapshots."""
    attempts_dir = repo_root / ".mecha-agent" / "attempts"
    if attempts_dir.exists():
        shutil.rmtree(attempts_dir)


_MODULE_NOT_FOUND_RE = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")


@dataclass(frozen=True)
class _PipResult:
    ok: bool
    exit_code: int
    stderr_tail: str


def _pip_install(packages: list[str], *, cwd: Path, timeout_sec: float = 600.0) -> _PipResult:
    """Install ``packages`` into the running interpreter via ``pip install``.

    Uses ``sys.executable -m pip install --disable-pip-version-check`` with
    ``shell=False`` so package names are never word-split. Output is captured
    and the last 600 chars of stderr are returned for diagnostics.
    """
    if not packages:
        return _PipResult(ok=True, exit_code=0, stderr_tail="")
    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *packages]
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return _PipResult(ok=False, exit_code=124, stderr_tail=f"pip install timed out after {timeout_sec:.0f}s")
    stderr_tail = (completed.stderr or "").strip()[-600:]
    return _PipResult(ok=completed.returncode == 0, exit_code=completed.returncode, stderr_tail=stderr_tail)


def _extract_missing_modules(summary: str) -> set[str]:
    """Return the set of top-level package names reported as missing.

    Parses ``ModuleNotFoundError: No module named 'foo.bar'`` lines from the
    validator summary and keeps only the top-level package (``foo``).
    """
    missing: set[str] = set()
    for match in _MODULE_NOT_FOUND_RE.finditer(summary):
        top = match.group(1).split(".", 1)[0].strip()
        if top:
            missing.add(top)
    return missing


def _extract_traceback_tail(summary: str, max_lines: int = 30) -> str:
    """Return the last ``max_lines`` of a Python traceback block from the summary.

    Falls back to the trailing ``max_lines`` of the summary when no obvious
    traceback marker is found.
    """
    lines = summary.splitlines()
    start = 0
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("Traceback (most recent call last)"):
            start = idx
            break
    tail = lines[start:] if start else lines
    if len(tail) > max_lines:
        tail = tail[-max_lines:]
    return "\n".join(tail).strip()


__all__ = ["DirectAgentLoop", "MissingPackagePrompt", "reset_attempts_dir"]
