# Architecture

`mecha-agent-cli` separates model, control, repository, validation, memory, sandbox, and UI concerns. This branch uses a staged component architecture for weak/small local models.

```text
User CLI
  -> Typer command layer
  -> AgentLoop
      -> TaskClassifier
      -> RepoMap + MemoryRetriever
      -> Planner(JSON only)
      -> DomainTemplate selection
      -> ComponentSpec + InterfaceSpec inference
      -> CodeUnitGenerator(JSON code unit only)
      -> Staged JSON storage under .mecha-agent/staged/
      -> Temporary workspace validation
      -> AttemptLedger duplicate/stagnation policy
      -> CodeUnitRepairer(JSON repair, max 3 per candidate)
      -> Fresh candidate retry loop until success or attempt budget exhaustion
      -> RichAgentObserver terminal progress events
      -> Materialization patch to algorithm.py only after staged validation passes
      -> Final ValidationPipeline
      -> SQLiteStore + Reward + Bandit
```

## Component-level generation

The model is not asked to design the public API. The runtime supplies:

- `DomainTemplateSpec`: domain-specific inputs, outputs, invariants, and validation oracles.
- `ComponentSpec`: one function or one class to implement.
- `InterfaceSpec`: exact signature and extensible input/output envelope.

The LLM returns `CodeUnitDraftResponse` JSON. The `code` field is raw Python, but it is not written to `algorithm.py` immediately.

## Model/context/tool separation

The model never receives direct file-writing or shell-execution tools. It produces structured JSON only. The CLI control layer validates JSON, validates code in staging, applies patches, runs validation, and decides whether repair is allowed.

## Validation before materialization

A draft code unit is copied to a temporary workspace, combined with existing tests/config where available, and checked with syntax/import/Ruff/Pyright/pytest/semantic validation. Passing drafts are then materialized to `algorithm.py` and validated again.

## Initial scope

The first stable target is `algorithm.py`. `tests/test_algorithm.py` can be written only by explicit test-generation mode. Multi-file editing is a future configuration option, not the default.


## Terminal observability

`AgentLoop` emits `AgentProgressEvent` records for preflight, planning, template
selection, code-unit generation, staged validation, repair, retry, materialization,
final validation, and completion. The CLI renders these as concise terminal lines so the
user can see exactly what the agent is trying while it runs.

## Reliability controller

The continuous loop has a deterministic `AttemptLedger` between validation and repair.
It tracks duplicate code-unit digests, same-failure streaks, and score improvement.
The ledger is intentionally outside the model so a weak local model cannot trap the
agent in repeated identical repairs.
