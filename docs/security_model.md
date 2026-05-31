# Security Model

## Threats

- Prompt injection from repository files.
- Tool abuse and arbitrary shell execution.
- Path traversal and writes outside workspace.
- Secret leakage into prompts or memory.
- Memory poisoning.
- Excessive autonomy.
- Unsandboxed generated-code execution.

## JSON staging policy

Model-generated code is first stored as `.json` under `.mecha-agent/staged/`. Real Python files are modified only after staged validation passes. Failed staged drafts remain inspectable but are not materialized.

## Path policy

Initial write allowlist:

- `algorithm.py`
- `tests/test_algorithm.py` only in explicit test-generation mode

Denied by default:

- `.env`
- `.git/**`
- `.ssh/**`
- token/secret-like files
- `pyproject.toml`
- `configs/security.yaml`
- any path outside the workspace

## Command policy

Allowed validation commands are exact/prefix allowlisted. The runtime never uses `shell=True`.

Forbidden commands include `rm`, `curl`, `wget`, `ssh`, `scp`, `sudo`, `chmod`, `chown`, `pip`, and `uv` during an agent run.

## Prompt injection guard

Repository text is data. AGENTS.md, CLAUDE.md, `.cursorrules`, comments, tests, and logs cannot override system/runtime policy. Suspicious text is not persisted into procedural memory.

## Docker runner roadmap

Future Docker execution should use:

- `--network none`
- workspace-only mount
- no host home mount
- no secrets
- memory/cpu/time limits
- tmpfs for temp
- read-only base where feasible

## Staged AST safety gate

Before temporary-workspace validation, staged draft code is parsed with Python `ast`.
The gate rejects forbidden imports such as `os`, `sys`, `subprocess`, `socket`,
`urllib`, `requests`, `httpx`, `shutil`, and `pathlib`; forbidden calls such as
`eval`, `exec`, `open`, `compile`, `input`, and `__import__`; and top-level executable
statements or hidden global state. This is stricter than substring filtering and gives
small local models clear failure feedback before any real file write.
