# Agent Loop

The runtime loop is deterministic and staged:

1. Preflight: resolve workspace, load config, initialize memory.
2. Classify task.
3. Build compact context: user request, target file, repo map, validation summary, and memory/reflection brief.
4. Select strategy with epsilon-greedy reward statistics.
5. Ask planner for JSON `PlanResponse`; no code is written.
6. Infer `DomainTemplateSpec`, `ComponentSpec`, and `InterfaceSpec` deterministically.
7. Ask the model for one `CodeUnitDraftResponse` JSON object.
8. Store the draft JSON under `.mecha-agent/staged/`.
9. Validate draft code in a temporary workspace.
10. If validation fails, record the attempt in `AttemptLedger` and repair the JSON draft only; do not write Python.
11. Repeat focused repair up to `agent.max_repairs` times for that candidate.
12. If the candidate is duplicated, repeatedly fails the same gate, or stops improving, abandon it and generate a fresh staged candidate.
13. Materialize passing code to `algorithm.py` through a path-checked full-file patch.
14. Run final validation on the real workspace.
15. If final validation fails, restore the previous `algorithm.py` and continue with a fresh staged candidate.
16. Stop only after success, attempt-budget exhaustion, security failure, or user interruption.
17. Store artifacts, validation, reward, reflection, and strategy stats.

## Repair policy

The repair loop is capped at three attempts. Each repair sees the same component specification, the previous staged code unit, compact validation feedback, and relevant reflections. It must preserve the public interface and fix only the primary failure category.

## JSON-before-Python policy

All model-generated code remains in formatted JSON until staging validation passes. This lets the runtime compare candidates, inspect inputs/outputs, and defer real file writes.


## Continuous attempt policy

The loop is foreground-continuous: every generate, staged validation, repair, retry,
materialization, and final validation event is emitted through an `AgentObserver`. The
Typer CLI uses `RichAgentObserver`, so the user sees what the agent is trying at each
moment.

Default attempt budget:

```yaml
agent:
  max_total_attempts: 25
  max_repairs: 3
  continue_until_success: false
  max_duplicate_drafts: 2
  max_same_failure_streak: 4
  min_score_improvement: 0.01
```

CLI overrides:

```bash
mecha-agent run "TASK" --max-attempts 50
mecha-agent run "TASK" --until-success
```

`--until-success` uses no fixed attempt cap and should be treated as an interactive mode.

## Attempt ledger

`AttemptLedger` stores a digest, primary failure type, validation score, and source
(`generate`, `repair`, or `final`) for each staged attempt. This gives the runtime a
non-LLM reliability controller that can stop duplicate or non-improving repairs and
request a fresh candidate while keeping the terminal user informed.
