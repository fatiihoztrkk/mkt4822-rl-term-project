from pathlib import Path

from mecha_agent_cli.core.types import FailureType
from mecha_agent_cli.validation.undefined_name_check import UndefinedNameCheck


def test_undefined_name_check_flags_missing_import(tmp_path: Path) -> None:
    (tmp_path / "algorithm.py").write_text(
        "def f() -> int:\n    return random.randint(1, 3)\n",
        encoding="utf-8",
    )
    result = UndefinedNameCheck().run(tmp_path)
    assert result.passed is False
    assert result.failure_type == FailureType.TYPE
    assert "random" in result.stderr_excerpt


def test_undefined_name_check_passes_known_names(tmp_path: Path) -> None:
    (tmp_path / "algorithm.py").write_text(
        "import random\n\ndef f() -> int:\n    return random.randint(1, 3)\n",
        encoding="utf-8",
    )
    result = UndefinedNameCheck().run(tmp_path)
    assert result.passed is True
