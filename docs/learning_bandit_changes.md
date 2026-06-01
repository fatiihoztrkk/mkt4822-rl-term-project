# Contextual Bandit Learning Update

This update replaces the placeholder learning sidecar with a working
contextual bandit implementation for model-profile selection.

## Added behavior

- Classify requests into compact task families such as sorting, numerical
  work, signal filtering, reinforcement learning, graph, string, and IO tasks.
- Select among multiple direct-generation model profiles.
- Support Thompson Sampling, UCB1, epsilon-greedy, and disabled selection
  modes.
- Force initial exploration with `min_pulls_before_exploit`.
- Shape rewards from success, validator progress, attempts, latency, extraction
  failures, and behavior-check failures.
- Persist per-context arm posteriors in the existing SQLite `bandit_arms`
  schema.
- Add a second assignment fixture for benchmark coverage.

## Updated files

- `src/mecha_agent_cli/learning/arm_registry.py`
- `src/mecha_agent_cli/learning/bandit.py`
- `src/mecha_agent_cli/learning/context.py`
- `src/mecha_agent_cli/learning/reward.py`
- `src/mecha_agent_cli/learning/__init__.py`
- `project-assignment/mkt4822_q_learning.txt`

## Verification

The update was checked with:

```bash
python3 -m pytest -q
ruff check .
ruff format --check .
python3 -m compileall -q src tests scripts
python3 scripts/bandit_benchmark.py --episodes 24 --shared-bandit --max-attempts 2 --base-penalty 0.20
```

The fake-backend CLI flow was also exercised end to end with generation,
validation, and judge stages.

## Local Ollama runtime follow-up

The real Actor-Critic assignment exposed two runtime issues that were not
visible with the fake backend:

- The original `direct` profile enabled thinking with an unbounded output
  budget. On `qwen3:4b`, this could spend minutes in repetitive planning
  without returning a usable Python block.
- The environment doctor checked for a `python` executable only, even though
  this WSL environment correctly provides `python3`.

The follow-up adds a bounded `runtime_direct` profile (`think: false`,
`num_ctx: 4096`, `num_predict: 4096`) and points the default direct loop at
it. The doctor now recognizes either `python` or `python3`.

## Actor-Critic deliverable

The Actor-Critic assignment now declares required public function signatures
and a deterministic numeric acceptance criterion. This prevents a
syntactically valid but unrelated script from passing the semantic pipeline.

The checked-in `algorithm.py` is a runnable standard-library-only Actor-Critic
application. Running it trains a tabular policy, evaluates it separately, and
writes:

- `generation_report.csv`
- `training_progress.txt`

The local deterministic run reached a `97.78%` training success rate and a
`100.00%` evaluation success rate.
