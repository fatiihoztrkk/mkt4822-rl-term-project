"""Unit tests for prompt-derived behavioral validation checks."""

from __future__ import annotations

from pathlib import Path

from mecha_agent_cli.validation.behavior_check import BehaviorCheck, extract_behavior_criteria


def test_extract_form_after_call() -> None:
    prompt = "Q[3][1] must be >= 0.5 after solve(2000, 0.5, 0.95, 0.1, 0)."
    criteria = extract_behavior_criteria(prompt)
    assert len(criteria) == 1
    c = criteria[0]
    assert c.call_expr == "solve(2000, 0.5, 0.95, 0.1, 0)"
    assert c.setup_call is None
    assert c.lhs_expr == "result[3][1]"
    assert c.op == ">="
    assert c.rhs == 0.5


def test_extract_form_after_setup() -> None:
    prompt = 'After train(400, 0.5, 0.95, 0.1, 0), evaluate(q_table, 20, 1)["success_rate"] should be >= 0.5.'
    criteria = extract_behavior_criteria(prompt)
    assert len(criteria) == 1
    c = criteria[0]
    assert c.setup_call == "train(400, 0.5, 0.95, 0.1, 0)"
    assert c.call_expr is None
    assert c.lhs_expr == 'evaluate(q_table, 20, 1)["success_rate"]'
    assert c.op == ">="
    assert c.rhs == 0.5


def test_behavior_check_skips_when_no_criteria(tmp_path: Path) -> None:
    (tmp_path / "algorithm.py").write_text("def solve(x):\n    return x\n", encoding="utf-8")
    result = BehaviorCheck(user_request="Just write a solver.").run(tmp_path)
    assert result.passed
    assert result.skipped
    assert "skipped" in result.stdout_excerpt


def test_behavior_check_passes_valid_numeric_criterion(tmp_path: Path) -> None:
    source = (
        "def solve(n, alpha, gamma, epsilon, seed):\n"
        "    q = [[0.0, 0.0] for _ in range(5)]\n"
        "    q[3][1] = 0.75\n"
        "    return q\n"
    )
    (tmp_path / "algorithm.py").write_text(source, encoding="utf-8")
    prompt = "Q[3][1] must be >= 0.5 after solve(2000, 0.5, 0.95, 0.1, 0)."
    result = BehaviorCheck(user_request=prompt).run(tmp_path)
    assert result.passed
    assert not result.skipped


def test_behavior_check_fails_invalid_numeric_criterion(tmp_path: Path) -> None:
    source = (
        "def solve(n, alpha, gamma, epsilon, seed):\n"
        "    q = [[0.0, 0.0] for _ in range(5)]\n"
        "    q[3][1] = 0.1\n"
        "    return q\n"
    )
    (tmp_path / "algorithm.py").write_text(source, encoding="utf-8")
    prompt = "Q[3][1] must be >= 0.5 after solve(2000, 0.5, 0.95, 0.1, 0)."
    result = BehaviorCheck(user_request=prompt).run(tmp_path)
    assert not result.passed
    assert result.failure_type.value == "SEMANTIC"


def test_behavior_check_after_setup_can_use_dict_keys_as_locals(tmp_path: Path) -> None:
    source = (
        "def train(episodes, alpha, gamma, epsilon, seed):\n"
        "    return {'q_table': [[0.0]], 'episode_returns': [0.1]}\n"
        "\n"
        "def evaluate(q_table, episodes, seed):\n"
        "    return {'success_rate': 0.9}\n"
    )
    (tmp_path / "algorithm.py").write_text(source, encoding="utf-8")
    prompt = 'After train(400, 0.5, 0.95, 0.1, 0), evaluate(q_table, 20, 1)["success_rate"] should be >= 0.5.'
    result = BehaviorCheck(user_request=prompt).run(tmp_path)
    assert result.passed
