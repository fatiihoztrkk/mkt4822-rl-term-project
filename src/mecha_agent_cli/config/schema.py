"""Pydantic configuration schemas."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from mecha_agent_cli.core.constants import (
    DEFAULT_MAX_REPAIRS,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_TARGET_FILE,
    FALLBACK_MODEL,
)


class ModelProfile(BaseModel):
    """Runtime options for one LLM role."""

    model: str = DEFAULT_MODEL
    think: bool | None = None
    temperature: float = Field(default=0.03, ge=0.0, le=2.0)
    top_p: float = Field(default=0.65, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=200)
    repeat_penalty: float = Field(default=1.05, ge=0.0)
    num_ctx: int = Field(default=DEFAULT_NUM_CTX, ge=1024)
    num_predict: int = Field(default=1024, ge=-2)
    num_gpu: int | None = Field(
        default=None,
        ge=0,
        le=999,
        description=(
            "Number of model layers to keep on the GPU. None lets Ollama decide. "
            "On RTX 3060 6 GB the safe maximum for qwen3:4b at num_ctx=8192 is around 28-32 layers."
        ),
    )
    num_thread: int | None = Field(default=None, ge=1, le=64)
    seed: int | None = Field(
        default=None,
        description="Fixed sampler seed for reproducible generation; leave None to let Ollama pick.",
    )
    keep_alive: str | int | None = Field(default="30s")

    def to_options(self) -> dict[str, int | float]:
        """Return Ollama-compatible options."""
        options: dict[str, int | float] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        if self.num_thread is not None:
            options["num_thread"] = self.num_thread
        if self.seed is not None:
            options["seed"] = self.seed
        return options


class ModelsConfig(BaseModel):
    """Model backend configuration."""

    default_model: str = DEFAULT_MODEL
    fallback_model: str = FALLBACK_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    default_options: dict[str, str | int | float] = Field(
        default_factory=lambda: {"num_ctx": DEFAULT_NUM_CTX, "repeat_penalty": 1.05, "top_p": 0.70}
    )
    profiles: dict[str, ModelProfile] = Field(
        default_factory=lambda: {
            "planner": ModelProfile(think=False, temperature=0.05, top_p=0.65, num_predict=768),
            "patcher": ModelProfile(think=False, temperature=0.02, top_p=0.55, num_predict=1280),
            "code_unit": ModelProfile(think=False, temperature=0.02, top_p=0.55, num_predict=1280),
            "repairer": ModelProfile(think=False, temperature=0.01, top_p=0.50, num_predict=1024),
            "reviewer": ModelProfile(think=False, temperature=0.05, top_p=0.65, num_predict=768),
            "reflection": ModelProfile(think=False, temperature=0.05, top_p=0.65, num_predict=512),
            "json_repair": ModelProfile(think=False, temperature=0.0, top_p=0.40, num_ctx=4096, num_predict=768),
            # Mirrors pythalab-cortex chat options so Ollama makes the same
            # offload decision on a 6 GB GPU. With num_ctx=16384 the qwen3:4b
            # Q4 model (~2.6 GB) plus KV cache (~2.3 GB) sits near the 6 GB
            # VRAM limit, so Ollama partially offloads to CPU. That is what
            # caps GPU utilisation at ~50% on RTX 3060 (matching pythalab),
            # vs. 100% sustained when num_ctx=8192 fits fully on-GPU.
            # We deliberately do NOT send top_k / seed / num_gpu / num_thread
            # — leave them unset so Ollama picks defaults, exactly like
            # pythalab does. think=True is kept because the UI streams the
            # model's reasoning live; without it qwen3 emits no <think> chunks.
            "thinking": ModelProfile(
                think=True,
                temperature=0.4,
                top_p=0.9,
                repeat_penalty=1.1,
                num_ctx=16384,
                num_predict=-1,
                keep_alive="30m",
            ),
            # Backward-compatible alias for older configs that still reference
            # the historical direct profile name.
            "direct": ModelProfile(
                think=True,
                temperature=0.4,
                top_p=0.9,
                repeat_penalty=1.1,
                num_ctx=16384,
                num_predict=-1,
                keep_alive="30m",
            ),
            "critic": ModelProfile(
                think=False,
                temperature=0.1,
                top_p=0.6,
                repeat_penalty=1.05,
                num_ctx=4096,
                num_predict=320,
                keep_alive="10m",
            ),
        }
    )

    def profile(self, name: str) -> ModelProfile:
        """Return a named profile or a conservative fallback."""
        return self.profiles.get(name, ModelProfile(model=self.default_model))


class ValidationConfig(BaseModel):
    """Validation pipeline settings."""

    target_file: str = DEFAULT_TARGET_FILE
    max_repairs: int = Field(default=DEFAULT_MAX_REPAIRS, ge=0, le=10)
    semantic_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    run_ruff: bool = True
    run_pyright: bool = True
    run_pytest: bool = True
    run_semantic: bool = True
    run_hypothesis: bool = False
    pytest_timeout_sec: float = Field(
        default=180.0,
        ge=5.0,
        le=1800.0,
        description="Timeout for the staged/final pytest invocation; can be slow on first import.",
    )
    pyright_timeout_sec: float = Field(
        default=120.0,
        ge=5.0,
        le=900.0,
        description="Timeout for the staged/final Pyright invocation.",
    )
    ruff_timeout_sec: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Timeout for each Ruff lint/format invocation.",
    )
    import_timeout_sec: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Timeout for the import smoke check.",
    )
    runtime_timeout_sec: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Timeout for the runtime smoke check (executes the target file as __main__).",
    )
    run_runtime_check: bool = Field(
        default=True,
        description=(
            "When true, the pipeline executes the target file as __main__ after the import "
            "smoke check passes, capturing stdout/stderr and feeding any uncaught exception "
            "back to the model on the next attempt."
        ),
    )
    run_spec_contract: bool = Field(
        default=True,
        description=(
            "When true, the pipeline parses the user's prompt for ``def NAME(args)`` "
            "patterns and verifies, via AST, that every named function is actually "
            "defined in the target file with at least the required positional arity. "
            "Catches stub responses that satisfy syntax/import/runtime but ignore the "
            "user's spec. Pure-Python, sub-millisecond, no LLM calls."
        ),
    )
    run_behavior_check: bool = Field(
        default=True,
        description=(
            "When true, parse numeric acceptance criteria from prompt text and execute "
            "them in an isolated subprocess after runtime passes. This adds a lightweight "
            "semantic gate beyond compile/import/runtime."
        ),
    )
    behavior_timeout_sec: float = Field(
        default=20.0,
        ge=3.0,
        le=300.0,
        description="Timeout for behavioral validation checks parsed from prompt criteria.",
    )
    pyright_strict: bool = Field(
        default=False,
        description=(
            "When true, the staging Pyright run uses strict mode. Default is the basic mode "
            "since small local models often trip on aggressive strict-only diagnostics."
        ),
    )


class SecurityConfig(BaseModel):
    """Filesystem and command policy configuration."""

    workspace_only: bool = True
    allow_test_generation: bool = False
    allow_multi_file_edit: bool = False
    allow_third_party_imports: bool = False
    allow_data_science_imports: bool = Field(
        default=False,
        description=(
            "When true, generated code may import numpy, pandas, scipy, sklearn, matplotlib, "
            "torch, and tensorflow. Security-critical roots (subprocess, socket, os, sys, shutil, "
            "urllib, requests, httpx) remain blocked regardless of this flag."
        ),
    )
    write_allowlist: list[str] = Field(default_factory=lambda: [DEFAULT_TARGET_FILE])
    explicit_test_write_allowlist: list[str] = Field(default_factory=lambda: ["tests/test_algorithm.py"])
    deny_write_patterns: list[str] = Field(
        default_factory=lambda: [
            ".env",
            ".git/**",
            ".ssh/**",
            "**/*token*",
            "**/*secret*",
            "pyproject.toml",
            "configs/security.yaml",
        ]
    )
    forbidden_code_patterns: list[str] = Field(
        default_factory=lambda: [
            "eval(",
            "exec(",
            "subprocess",
            "socket",
            "urllib",
            "requests",
            "httpx",
        ]
    )


class AgentConfig(BaseModel):
    """Loop-level runtime limits."""

    max_steps: int = Field(default=8, ge=1, le=50)
    max_repairs: int = Field(default=DEFAULT_MAX_REPAIRS, ge=0, le=10)
    max_total_attempts: int = Field(
        default=25,
        ge=1,
        le=500,
        description="Maximum staged generate/repair/final-validation attempts before stopping.",
    )
    continue_until_success: bool = Field(
        default=False,
        description="When true, keep generating new staged candidates until validation passes or the user interrupts.",
    )
    max_duplicate_drafts: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Maximum identical staged code drafts tolerated before forcing a fresh candidate.",
    )
    max_same_failure_streak: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Maximum repeated primary failure streak before abandoning the current candidate.",
    )
    min_score_improvement: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum validation-score improvement required to keep repairing a candidate.",
    )
    default_backend: str = "ollama"


class RepoConfig(BaseModel):
    """Repository context settings."""

    target_file: str = DEFAULT_TARGET_FILE
    max_file_window_lines: int = Field(default=120, ge=20, le=400)


class MemoryConfig(BaseModel):
    """Memory store settings."""

    path: str = ".mecha-agent/memory.sqlite"
    top_k_reflections: int = Field(default=3, ge=0, le=10)
    top_k_tasks: int = Field(default=3, ge=0, le=10)


class DirectGenerationConfig(BaseModel):
    """Settings for the direct (template-free) generation loop.

    Tuned for stable serial code generation on a single low-VRAM GPU
    (RTX 3060 6 GB) with cumulative chat-history merging across attempts.
    """

    profile_name: str = Field(
        default="thinking",
        description="Name of the ModelProfile to use for direct generation calls.",
    )
    max_attempts: int = Field(
        default=10,
        ge=1,
        le=200,
        description="Maximum direct-generation attempts before giving up on a task.",
    )
    max_history_chars: int = Field(
        default=24000,
        ge=2000,
        le=200000,
        description=(
            "Maximum total characters retained across all assistant/user turns; older "
            "intermediate turns are dropped first while preserving system prompt and "
            "the latest exchange."
        ),
    )
    request_timeout_sec: float = Field(
        default=600.0,
        ge=10.0,
        le=3600.0,
        description="Per-call HTTP timeout for direct chat generation.",
    )
    save_attempt_snapshots: bool = Field(
        default=True,
        description="When true, each attempt's generated code is saved under .mecha-agent/attempts/.",
    )
    error_summary_max_lines: int = Field(
        default=80,
        ge=5,
        le=400,
        description="How many lines of validation feedback to feed back to the model per turn.",
    )
    auto_judge_after_validation: bool = Field(
        default=True,
        description=(
            "When true, a short-form LLM judge runs after validator pass and can request "
            "additional repair attempts if semantic confidence is low or suspicious."
        ),
    )
    judge_profile_name: str = Field(
        default="critic",
        description="ModelProfile name used for post-validation adjudication.",
    )
    judge_max_repairs: int = Field(
        default=2,
        ge=0,
        le=20,
        description="Maximum extra judge-driven repair turns after validator pass.",
    )
    judge_min_confidence: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum judge confidence required to accept PASS verdict.",
    )
    judge_prompt_max_chars: int = Field(
        default=8000,
        ge=1000,
        le=50000,
        description="Maximum evidence payload size sent to the post-validation judge.",
    )


class LearningConfig(BaseModel):
    """Reinforcement-learning sidecar settings (Phase 1: contextual Thompson bandit).

    The bandit selects a ModelProfile override per episode (one task/run call).
    All learning state lives in SQLite (``MemoryConfig.path``), CPU-only,
    sub-millisecond per select/update, runtime-neutral.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Master switch for the contextual bandit. When false the loop "
            "uses ``direct.profile_name`` directly, identical to pre-RL behaviour."
        ),
    )
    mode: str = Field(
        default="thompson",
        description=(
            "Selection mode: 'thompson' (Beta-Bernoulli posterior sampling), "
            "'ucb1' (upper-confidence-bound on posterior mean), "
            "'greedy' (highest mean reward, ε-greedy fallback), "
            "or 'off' (always pick the registry baseline arm)."
        ),
    )
    ucb_c: float = Field(
        default=1.4,
        ge=0.0,
        le=10.0,
        description=(
            "Exploration coefficient for 'ucb1' mode. Higher values push the "
            "selector to explore arms with high uncertainty (default 1.4 ≈ √2)."
        ),
    )
    decay_factor: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "Multiplicative decay applied to (alpha-1, beta-1) before each "
            "posterior update. 1.0 disables decay; 0.99 keeps ~99% of prior "
            "evidence per update so old observations fade gracefully when the "
            "model or task distribution changes."
        ),
    )
    latency_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Reward penalty per ``latency_horizon_sec`` of wall-clock time used "
            "by the episode. 0.0 disables latency-aware shaping. Generalises "
            "across tasks: long, slow successes get less posterior credit."
        ),
    )
    latency_horizon_sec: float = Field(
        default=120.0,
        gt=0.0,
        description=(
            "Time normaliser for ``latency_penalty``. An episode that takes "
            "exactly this many seconds incurs the full ``latency_penalty``."
        ),
    )
    epsilon: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Exploration probability in 'greedy' mode; ignored otherwise.",
    )
    min_pulls_before_exploit: int = Field(
        default=2,
        ge=0,
        le=100,
        description=(
            "Each arm in a context is forced to be sampled at least this many "
            "times before Thompson posterior reasoning kicks in."
        ),
    )
    success_reward: float = Field(default=1.0, description="Reward for a passing validation report.")
    attempt_penalty: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Per-attempt penalty subtracted from success reward to favour fast solutions.",
    )
    progress_bonus: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Partial credit per validation check passed when the episode "
            "fails (syntax→import→runtime). Encourages partial progress."
        ),
    )
    extract_failure_penalty: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Extra negative reward when no fenced ```python``` block was returned.",
    )
    behavior_failure_penalty: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description=(
            "Extra negative reward when behavioral criteria fail. Helps RL avoid "
            "compile-clean but semantically-wrong outputs."
        ),
    )
    persist: bool = Field(
        default=True,
        description="When false the bandit runs in-memory only (useful for tests/benchmarks).",
    )


class AppConfig(BaseModel):
    """Top-level config object."""

    repo: RepoConfig = Field(default_factory=RepoConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    direct: DirectGenerationConfig = Field(default_factory=DirectGenerationConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)

    @property
    def target_file(self) -> str:
        """Return the configured target file."""
        return self.repo.target_file or self.validation.target_file

    def memory_path(self, repo_root: Path) -> Path:
        """Resolve the SQLite database path relative to ``repo_root``."""
        return (repo_root / self.memory.path).resolve()
