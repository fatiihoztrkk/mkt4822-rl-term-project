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
