# Memory and Reward

The memory database is stored at:

```text
.mecha-agent/memory.sqlite
```

## Tables

- `tasks`: task request, type, strategy, model, status.
- `artifacts`: changed file, before/after hashes, diff.
- `validation_runs`: validation scores and compact failure summary.
- `reflections`: procedural lessons from failures.
- `repo_summaries`: cached symbol summaries.
- `strategy_stats`: attempts, successes, mean reward.

## Memory types

- Short-term/session memory: current run state.
- Repo-level memory: repo summaries.
- Episodic memory: historical tasks and artifacts.
- Reflection memory: failure-derived procedural rules.
- Procedural memory: future prompt/repair rules.
- Strategy memory: contextual-bandit reward statistics.

## Reward

The reward function adds positive signal for passing tests, Pyright, Ruff, semantic alignment, and no-repair success. It penalizes repair attempts, patch parse failure, import crash, security violation, semantic mismatch, and ignored public API.

The reward updates only external policies: prompt template choice, context strategy, repair strategy, validation policy, memory retrieval count, and schema strictness. Model weights are never updated.
