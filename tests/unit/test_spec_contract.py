"""Unit tests for the spec-contract validator."""

from __future__ import annotations

from pathlib import Path

from mecha_agent_cli.validation.spec_contract import (
    SpecContractCheck,
    evaluate_contract,
    extract_required_functions,
)


def test_extract_picks_up_top_level_signatures() -> None:
    prompt = """
    Required public API:
        def train(episodes: int, alpha: float) -> dict[str, object]
        def evaluate(q_table: list[list[float]], episodes: int) -> dict[str, float]
        def main() -> None
    """
    specs = {s.name: s for s in extract_required_functions(prompt)}
    assert set(specs) == {"train", "evaluate", "main"}
    assert specs["train"].min_positional == 2
    assert specs["evaluate"].min_positional == 2
    assert specs["main"].min_positional == 0


def test_extract_ignores_underscore_and_dunder_names() -> None:
    prompt = "def __init__(self, x): ...\ndef _helper(a, b): ...\ndef solve(n: int) -> int: ..."
    specs = [s.name for s in extract_required_functions(prompt)]
    assert specs == ["solve"]


def test_extract_returns_empty_when_no_signatures() -> None:
    assert extract_required_functions("Just write code that sorts a list.") == []


def test_evaluate_passes_when_all_functions_present() -> None:
    prompt = "def solve(n: int, m: int) -> int: ..."
    source = "def solve(n, m):\n    return n + m\n"
    passed, diags = evaluate_contract(source, prompt)
    assert passed and diags == []


def test_evaluate_fails_when_function_missing() -> None:
    prompt = "def train(x, y): ...\ndef visualize(z): ..."
    source = "def train(x, y):\n    return x + y\n"  # visualize missing
    passed, diags = evaluate_contract(source, prompt)
    assert not passed
    names = {d.name for d in diags}
    assert names == {"visualize"}


def test_evaluate_fails_when_arity_too_low() -> None:
    prompt = "def train(episodes, alpha, gamma, epsilon, seed): ..."
    source = "def train(episodes):\n    return 0\n"
    passed, diags = evaluate_contract(source, prompt)
    assert not passed
    assert "signature mismatch" in diags[0].reason


def test_evaluate_accepts_var_args_implementation() -> None:
    """An implementation using *args satisfies any positional arity demand."""
    prompt = "def solve(a, b, c, d, e): ..."
    source = "def solve(*args):\n    return sum(args)\n"
    passed, _ = evaluate_contract(source, prompt)
    assert passed


def test_evaluate_accepts_class_method() -> None:
    """Top-level class methods count as defined names (covers OO solutions)."""
    prompt = "def step(self, action): ..."
    source = "class Env:\n    def step(self, action):\n        return 0\n"
    passed, _ = evaluate_contract(source, prompt)
    assert passed


def test_evaluate_no_op_when_prompt_has_no_signatures() -> None:
    passed, diags = evaluate_contract("garbage = 1\n", "Sort a list please.")
    assert passed and diags == []


def test_check_writes_to_disk_and_runs(tmp_path: Path) -> None:
    target = tmp_path / "algorithm.py"
    target.write_text("def main():\n    pass\n", encoding="utf-8")
    prompt = "def main() -> None: ...\ndef train(x, y, z): ..."
    result = SpecContractCheck(user_request=prompt).run(tmp_path)
    assert not result.passed
    assert "train" in result.stderr_excerpt
    assert "main" not in result.stderr_excerpt.replace("missing", "")  # main IS defined


def test_check_passes_with_full_implementation(tmp_path: Path) -> None:
    target = tmp_path / "algorithm.py"
    target.write_text(
        "def train(a, b):\n    return 0\n\ndef main():\n    train(1, 2)\n",
        encoding="utf-8",
    )
    prompt = "def train(a, b): ...\ndef main(): ..."
    result = SpecContractCheck(user_request=prompt).run(tmp_path)
    assert result.passed


def test_check_handles_syntax_error_gracefully(tmp_path: Path) -> None:
    target = tmp_path / "algorithm.py"
    target.write_text("def broken(:\n", encoding="utf-8")
    result = SpecContractCheck(user_request="def broken(): ...").run(tmp_path)
    assert not result.passed
