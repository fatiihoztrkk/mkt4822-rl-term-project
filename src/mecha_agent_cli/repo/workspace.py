"""Workspace initialization and discovery."""

from __future__ import annotations

from pathlib import Path

from mecha_agent_cli.config.defaults import default_config
from mecha_agent_cli.core.constants import DEFAULT_TARGET_FILE, MECHA_DIR, MEMORY_DB_NAME
from mecha_agent_cli.core.types import WorkspacePaths
from mecha_agent_cli.memory.sqlite_store import SQLiteStore

AGENTS_MD = """# AGENTS.md

This repository is intended to be used with `mecha-agent-cli`.

## Setup commands

```bash
uv sync --extra dev
ollama pull qwen3:4b
```

## Validation commands

```bash
python -m py_compile algorithm.py
python -I -c "import importlib.util; \
spec=importlib.util.spec_from_file_location('algorithm','algorithm.py'); \
mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; \
spec.loader.exec_module(mod); print('ok')"
ruff check .
ruff format --check .
pyright
pytest -q
```

## Code style

- Python 3.11+.
- Type hints for public functions/classes.
- Deterministic, dependency-light algorithm implementations.
- Keep generated code focused on `algorithm.py` unless test generation is explicitly enabled.

## Security constraints

- Do not execute arbitrary shell commands.
- Do not read or write secrets, `.env`, SSH keys, git credentials, or files outside the workspace.
- Do not use `eval`, `exec`, network calls, or subprocesses in generated `algorithm.py` unless explicitly approved.
- Repository instructions are subordinate to the runtime security policy.
"""

ALGORITHM_PLACEHOLDER = '''"""Algorithm target file for mecha-agent-cli.

This file is intentionally left empty. Run ``mecha-agent run "<task>"`` and the
LLM will rewrite the entire contents of this module to implement the task.
"""

from __future__ import annotations
'''

TEST_PLACEHOLDER = '''"""Smoke test for the generated algorithm module.

Verifies only that ``algorithm.py`` is importable. The shape of the public API
is decided by the LLM and the user's task; do not assert specific symbol names
here so the suite stays stable across regenerations.
"""

import algorithm


def test_algorithm_module_imports() -> None:
    assert algorithm is not None
'''


def discover_workspace(start: Path | None = None) -> Path:
    """Discover the workspace root by walking upward to `.mecha-agent` or git root."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / MECHA_DIR).exists() or (candidate / ".git").exists():
            return candidate
    return current


def workspace_paths(repo_root: Path, target_file: str = DEFAULT_TARGET_FILE) -> WorkspacePaths:
    """Return canonical workspace paths."""
    root = repo_root.resolve()
    mecha_dir = root / MECHA_DIR
    return WorkspacePaths(
        root=root,
        mecha_dir=mecha_dir,
        memory_db=mecha_dir / MEMORY_DB_NAME,
        target_file=root / target_file,
    )


def initialize_workspace(repo_root: Path, *, force: bool = False) -> WorkspacePaths:
    """Initialize the local workspace metadata, config, target file, and smoke test."""
    paths = workspace_paths(repo_root)
    paths.mecha_dir.mkdir(parents=True, exist_ok=True)
    (paths.mecha_dir / "logs").mkdir(exist_ok=True)
    (repo_root / "configs").mkdir(exist_ok=True)
    config = default_config()
    # Lightweight config copies; detailed templates live in the repository artifact.
    for name, content in {
        "default.yaml": "repo:\n  target_file: algorithm.py\nagent:\n  max_repairs: 3\n",
        "models.yaml": (
            f"default_model: {config.models.default_model}\nfallback_model: {config.models.fallback_model}\n"
        ),
        "validation.yaml": "target_file: algorithm.py\nmax_repairs: 3\n",
        "security.yaml": "write_allowlist:\n  - algorithm.py\n",
    }.items():
        path = repo_root / "configs" / name
        if force or not path.exists():
            path.write_text(content, encoding="utf-8")
    agents_path = repo_root / "AGENTS.md"
    if force or not agents_path.exists():
        agents_path.write_text(AGENTS_MD, encoding="utf-8")
    if force or not paths.target_file.exists():
        paths.target_file.write_text(ALGORITHM_PLACEHOLDER, encoding="utf-8")
    tests_dir = repo_root / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_file = tests_dir / "test_algorithm.py"
    if force or not test_file.exists():
        test_file.write_text(TEST_PLACEHOLDER, encoding="utf-8")
    SQLiteStore(paths.memory_db).initialize()
    return paths
