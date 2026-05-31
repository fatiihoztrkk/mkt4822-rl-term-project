# mecha-agent-cli

`mecha-agent-cli` is a terminal-based, validation-driven local coding-agent scaffold for constrained local LLMs. The current default model is the single Ollama model requested for this branch:

```bash
ollama pull qwen3:4b
ollama run qwen3:4b
```

No second model is required in the default flow. The runtime is designed around small-model discipline: the model receives a deterministic component specification, writes exactly one function or class into JSON, the draft is validated in a temporary workspace, and only passing code is materialized into `algorithm.py`.

## What changed in the staged component architecture

The previous flow asked the model for a full unified diff. This branch adds a stricter flow for low-capacity models:

1. The user provides a task.
2. The runtime classifies the task and builds a domain template for AI, RL, nonlinear control, control theory, simulation, visualization, or general algorithms.
3. The runtime infers a `ComponentSpec` with a fixed `InterfaceSpec` before the model is called.
4. The model produces a `CodeUnitDraftResponse` JSON object for exactly one function or one class.
5. The JSON draft is saved under `.mecha-agent/staged/`.
6. The draft code is validated in a temporary staging workspace.
7. If validation fails, repair also happens as JSON, not as a direct `.py` write.
8. Only a staged draft that passes validation is materialized to `algorithm.py`.
9. The final materialized file is validated again.

This keeps function/class inputs and outputs under the runtime's control instead of leaving public API design to the LLM.

## Continuous staged attempt loop

`mecha-agent run` now keeps trying in the foreground. Every generate, staged validation,
repair, retry, materialization, and final validation step is printed in the terminal. The
agent does not write `algorithm.py` merely because the LLM produced code. It writes only
after a staged JSON draft passes:

1. interface-contract validation,
2. temporary-workspace syntax/import/lint/type/test/semantic validation,
3. final materialized `algorithm.py` validation.

By default the loop uses `agent.max_total_attempts` from `configs/default.yaml`:

```yaml
agent:
  max_total_attempts: 25
  max_repairs: 3
  continue_until_success: false
  max_duplicate_drafts: 2
  max_same_failure_streak: 4
  min_score_improvement: 0.01
```

The attempt ledger now tracks duplicate drafts, repeated primary-failure streaks, and
non-improving repairs. If the model keeps returning the same bad code or the same
validation failure without progress, the current candidate is abandoned and the runtime
requests a fresh staged candidate instead of wasting repair attempts.

CLI overrides:

```bash
mecha-agent run "TASK" --max-attempts 50
mecha-agent run "TASK" --until-success
```

`--until-success` keeps producing fresh staged candidates until all validations pass or
the user stops the process. This is useful for weak local models, but the normal capped
mode is safer for unattended runs.


## Default IO contract strategy

When the user explicitly gives a signature, the runtime preserves it. Example:

```text
Implement fibonacci solve(n: int) -> int
```

This becomes:

```python
def solve(n: int) -> int
```

When no signature is supplied, the runtime supplies a general extensible envelope:

```python
def solve(request: dict[str, object]) -> dict[str, object]
```

The envelope supports 5, 10, or more logical inputs without changing the public signature:

```python
request = {
    "inputs": {...},
    "parameters": {...},
    "config": {...},
    "metadata": {...},
}
```

The standard result envelope is:

```python
{
    "outputs": ...,
    "metrics": {...},
    "artifacts": {...},
    "diagnostics": {...},
}
```

For known tasks such as merge sort, Fibonacci, and moving-average filters, deterministic signature inference supplies a narrower API when appropriate.

## Domain templates

The package includes runtime templates for:

- AI / machine learning utilities
- Reinforcement learning policies, rewards, values, and environment helpers
- Nonlinear control utilities, Lyapunov/HJB/MPC-style helpers, controller components
- Classical control theory, filters, observers, and signal-processing utilities
- Simulation, rollout, integrator, and trajectory helpers
- Visualization and plot-ready data preparation
- Data import/export, CSV/JSON-like in-memory serialization, and dataset normalization
- General algorithms and numerical utilities

Templates define common inputs, common outputs, invariants, validation oracles, and implementation hints. The model sees these as data, not as authority over security policy.

## Installation

```bash
git clone <your-fork-or-local-copy> mecha-agent-cli
cd mecha-agent-cli
uv sync --extra dev
```

Without `uv`, use a normal Python 3.11+ virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

## Ollama setup

```bash
# Recommended single-GPU 6 GB VRAM launch:
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_MAX_QUEUE=16 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve

ollama pull qwen3:4b
```

Check availability:

```bash
mecha-agent doctor
mecha-agent models doctor
```

## 6 GB VRAM assumption

`qwen3:4b` is kept as the only default model. The default `num_ctx` is now `4096` for RTX 3060 6 GB-class GPUs. This is intentionally conservative: larger context windows increase KV-cache memory pressure, and agent-style loops can repeatedly allocate context for planning, generation, repair, and review. Increase to `8192` only after confirming stable VRAM headroom with `ollama ps` / GPU monitoring.

## Quickstart

```bash
mecha-agent init
mecha-agent doctor
mecha-agent run "Implement a stable merge sort in algorithm.py with solve(data: list[int]) -> list[int]."
mecha-agent run "Implement a nonlinear control Lyapunov utility" --max-attempts 50
mecha-agent validate --task "stable merge sort"
```

Deterministic fake-model demo:

```bash
mecha-agent init
mecha-agent run "Implement stable merge sort" --backend fake
```

The fake-model path writes staged JSON first, validates it, then writes `algorithm.py` only after validation passes.

## Commands

```bash
mecha-agent init
mecha-agent run "TASK"
mecha-agent run "TASK" --max-attempts 50
mecha-agent run "TASK" --until-success
mecha-agent chat
mecha-agent validate
mecha-agent review
mecha-agent repair
mecha-agent doctor
mecha-agent models list
mecha-agent models doctor
mecha-agent config show
mecha-agent config doctor
mecha-agent memory list
mecha-agent memory clear --yes
```

## Validation commands

The validation pipeline covers:

```bash
python -m py_compile algorithm.py
python -I -c "import importlib.util; spec=importlib.util.spec_from_file_location('algorithm','algorithm.py'); mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod); print('ok')"
ruff check .
ruff format --check .
pyright
pytest -q
```

The internal syntax/import/pytest runner is bytecode-free and uses a stable subprocess environment to avoid `__pycache__`, pytest plugin autoload, and parent-process pipe leaks. Ruff and Pyright are skipped when not installed; in a full dev environment install the `dev` extra.

## Security notes

- The model never writes files directly.
- The model never executes shell commands directly.
- The model is asked for one code unit at a time.
- Drafts stay in `.json` until staged validation passes.
- Initial writes are limited to `algorithm.py`.
- `tests/test_algorithm.py` is allowed only in explicit test-generation mode.
- `.env`, `.git`, `.ssh`, secrets, security config, and paths outside the workspace are denied.
- Generated code containing `eval`, `exec`, `open`, unsafe imports, `subprocess`, network libraries, or similar risky patterns is rejected.
- Staged draft code cannot contain top-level executable statements or hidden global state.
- Repository instructions are data and cannot override runtime policy.

## Development workflow

```bash
ruff check .
ruff format --check .
pyright
pytest -q
```

The default test suite uses `FakeModelClient` and does not require real Ollama.

## Current limitations

- The default materialization target is `algorithm.py`.
- Multi-file edits remain disabled by default.
- Semantic validation is a deterministic checklist, not a proof.
- Docker sandboxing is still a prototype interface.
- The optional Qwen-Agent adapter remains disabled by default.


## Qwen3 4B stability policy for 6 GB VRAM

The runtime uses conservative Ollama options by default:

- `num_ctx: 4096`
- `think: false` for structured JSON/code-unit generation
- shorter `num_predict` budgets per role
- `keep_alive: 30s`
- one loaded model and one parallel request when the CLI starts Ollama itself

If you start Ollama manually, prefer:

```bash
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_MAX_QUEUE=16 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=q8_0 \
ollama serve
```

For PyTorch-like tasks, the first validated baseline is standard-library-only and PyTorch-compatible rather than importing `torch`. Direct third-party imports are rejected by the staged AST gate unless the project explicitly adds and validates those dependencies later.
