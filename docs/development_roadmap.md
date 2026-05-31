# Development Roadmap

## Phase 0: Research-grounded skeleton

Package structure, configs, README, docs, AGENTS.md.

## Phase 1: Minimal CLI + fake model

Typer CLI, workspace init, fake model, staged JSON generation, basic validation, deterministic tests.

## Phase 2: Ollama integration

Native `/api/chat`, structured outputs, model doctor, `qwen3:4b` model profiles, context handling.

## Phase 3: Component-level generation loop

Domain templates, deterministic `ComponentSpec`, deterministic `InterfaceSpec`, one function/class per call, JSON draft storage.

## Phase 4: Validation pipeline

Syntax, import, Ruff, Pyright, pytest, semantic checklist, and temporary staging workspace.

## Phase 5: Repair loop

Failure summary, staged JSON repair, max three attempts, replay logs.

## Phase 6: Memory + reward

SQLite tables, task/artifact/validation/reflection memory, reward computation.

## Phase 7: Contextual bandit

Epsilon-greedy strategy stats; future UCB.

## Phase 8: Security hardening

Secret scanner, command allowlist, prompt injection guard, Docker prototype.

## Phase 9: Documentation and examples

Examples, full docs, coverage.

## Phase 10: Optional Qwen-Agent adapter

Experimental optional adapter, disabled by default and never granted direct write/shell access.
