# Quality Report

This report records the validation gates used for the `qwen3:4b` staged-agent artifact.

## Validation performed in the build sandbox

The sandbox used to prepare this artifact did not have the `ruff` and `pyright` executables installed, so those two external binaries were not executed here. They remain configured as development dependencies and as runtime validation gates.

Executed checks:

```bash
python - <<'PY'
from pathlib import Path
for root in [Path('src'), Path('tests')]:
    for path in root.rglob('*.py'):
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
PY
```

Observed result:

```text
COMPILE_BAD 0
```

Targeted pytest checks executed with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`:

```bash
pytest tests/unit/test_domain_templates.py tests/unit/test_ollama_service.py -q
pytest tests/unit/test_staged_generation.py -k 'actor_critic or domain_fallbacks or store or enforce' -q
pytest tests/unit/test_ollama_client.py -q
pytest tests/unit/test_validation_pipeline.py -q
```

Observed results:

```text
7 passed
9 passed
6 passed
5 passed
```

## Required local checks after `uv sync --extra dev`

Run these in the final local environment:

```bash
ruff check .
ruff format --check .
pyright
pytest -q
```

## CLI smoke check

The deterministic fake backend was used so the smoke test does not require a live Ollama model. A Q-learning/PyTorch-style request now materializes a standard-library-only, PyTorch-compatible staged fallback after the initial fake model draft fails semantic validation:

```bash
PYTHONPATH=src python -m mecha_agent_cli init /tmp/mecha-workspace
PYTHONPATH=src python -m mecha_agent_cli run \
  "Write a Q learning code in python pytorch" \
  --backend fake \
  --path /tmp/mecha-workspace \
  --max-attempts 5
```

Observed terminal milestones:

```text
staged_validate fail attempt=0 — SEMANTIC
fallback_code_unit running attempt=1
staged_validate pass attempt=1
materialize running — writing validated staged module template to algorithm.py
final_validate success — algorithm.py passed final validation
complete success used=2/5 — algorithm.py written after 2 staged attempt(s)
```

## Reliability gates added in this iteration

- PlanManifest creation before code generation.
- Domain templates for AI/ML, RL, nonlinear control, control theory, simulation, visualization, and data import/export.
- Standard request/result envelope instead of model-invented APIs.
- Single code-unit generation and staged JSON persistence before Python materialization.
- Deterministic stdlib-only PyTorch-compatible fallbacks for RL/neural-network requests.
- Conservative RTX 3060 6 GB Ollama profile: `num_ctx=4096`, short `num_predict`, `think=false`, `keep_alive=30s`.
- Ollama server env defaults: one parallel request, one loaded model, Flash Attention, and `q8_0` KV cache when supported.
- Temporary-file subprocess output capture for validation commands.
- Direct safe materialization after staged validation instead of re-applying a full-file patch.
