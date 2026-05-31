"""Behavioral validation from prompt-level acceptance criteria.

This check adds a lightweight semantic gate on top of syntax/import/runtime:

- Parse simple numeric acceptance criteria from the user prompt.
- Execute the referenced function calls in an isolated subprocess.
- Verify inequalities such as ``>=``, ``<=``, ``==``, ``>``, ``<``.

Supported criterion forms (case-insensitive):

1. ``<lhs_expr> must/should be <op> <number> after <call_expr>``
2. ``After <setup_call>, <lhs_expr> must/should be <op> <number>``

Examples:
  - ``Q[3][1] must be >= 0.5 after solve(2000, 0.5, 0.95, 0.1, 0)``
  - ``After train(400, 0.5, 0.95, 0.1, 0), evaluate(q_table, 20, 1)[\"success_rate\"] should be >= 0.5``

Safety model:
- Expressions are parsed with :mod:`ast` and validated against a strict allowlist.
- Evaluation runs with ``__builtins__`` stripped.
- Only names from the imported module and explicit local bindings are visible.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from mecha_agent_cli.core.types import FailureType
from mecha_agent_cli.sandbox.local_runner import LocalRunner
from mecha_agent_cli.validation.base import ValidationCheck
from mecha_agent_cli.validation.report import ValidationResult

_FORM1_RE = re.compile(
    r"^(?P<lhs>.+?)\s+(?:must|should)\s+be\s*"
    r"(?P<op>>=|<=|==|>|<)\s*(?P<rhs>-?\d+(?:\.\d+)?)\s+"
    r"after\s+(?P<call>[A-Za-z_][A-Za-z0-9_]*\([^\n)]*\))\.?$",
    re.IGNORECASE,
)
_FORM2_RE = re.compile(
    r"^after\s+(?P<setup>[A-Za-z_][A-Za-z0-9_]*\([^\n)]*\))\s*,\s*"
    r"(?P<lhs>.+?)\s+(?:must|should)\s+be\s*"
    r"(?P<op>>=|<=|==|>|<)\s*(?P<rhs>-?\d+(?:\.\d+)?)\.?$",
    re.IGNORECASE,
)
_CANDIDATE_EXPR_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\([^\n)]*\))?(?:\[[^\n\]]+\])*")


@dataclass(frozen=True)
class BehaviorCriterion:
    """One parsed acceptance criterion."""

    lhs_expr: str
    op: str
    rhs: float
    call_expr: str | None = None
    setup_call: str | None = None


def _normalise_lhs(lhs: str) -> str:
    text = lhs.strip()
    # Common convention in RL prompts: "Q[...]" refers to the return value.
    if text.startswith("Q["):
        return "result" + text[1:]
    if text in {"Q", "q", "result"}:
        return "result"
    return text


def _extract_lhs_candidate(text: str) -> str | None:
    """Extract a likely expression candidate from prose left-hand side text."""
    text = text.strip()
    if not text:
        return None
    # Prefer the last bracketed candidate (often Q[...], arr[...], foo()[...]).
    candidates = _CANDIDATE_EXPR_RE.findall(text)
    if not candidates:
        return None
    bracketed = [c for c in candidates if "[" in c]
    return _normalise_lhs(bracketed[-1] if bracketed else candidates[-1])


def extract_behavior_criteria(prompt: str) -> list[BehaviorCriterion]:
    """Parse prompt text into executable behavioral criteria.

    Returns an empty list when no supported criterion lines are found.
    """
    criteria: list[BehaviorCriterion] = []
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m2 = _FORM2_RE.match(line)
        if m2 is not None:
            lhs = _normalise_lhs(m2.group("lhs").strip())
            criteria.append(
                BehaviorCriterion(
                    lhs_expr=lhs,
                    op=m2.group("op"),
                    rhs=float(m2.group("rhs")),
                    setup_call=m2.group("setup").strip(),
                )
            )
            continue
        m1 = _FORM1_RE.match(line)
        if m1 is not None:
            lhs_raw = m1.group("lhs").strip()
            lhs = _extract_lhs_candidate(lhs_raw)
            if lhs is None:
                continue
            criteria.append(
                BehaviorCriterion(
                    lhs_expr=lhs,
                    op=m1.group("op"),
                    rhs=float(m1.group("rhs")),
                    call_expr=m1.group("call").strip(),
                )
            )
            continue
    return criteria


def _build_runner_snippet(criteria: list[BehaviorCriterion]) -> str:
    payload = base64.b64encode(json.dumps([asdict(c) for c in criteria]).encode("utf-8")).decode("ascii")
    return (
        "import ast, base64, importlib.util, json, math, re, sys\n"
        "\n"
        f"_payload = {payload!r}\n"
        "criteria = json.loads(base64.b64decode(_payload).decode('utf-8'))\n"
        "\n"
        "spec = importlib.util.spec_from_file_location('algorithm', 'algorithm.py')\n"
        "if spec is None or spec.loader is None:\n"
        "    print('could not load algorithm.py')\n"
        "    raise SystemExit(1)\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "env = {k: v for k, v in vars(mod).items() if not k.startswith('__')}\n"
        "\n"
        "_ALLOWED = (\n"
        "    ast.Expression, ast.Name, ast.Load, ast.Constant, ast.Call, ast.Subscript,\n"
        "    ast.Tuple, ast.List, ast.Dict, ast.keyword, ast.Slice, ast.UnaryOp, ast.USub\n"
        ")\n"
        "\n"
        "def _validate(node: ast.AST) -> None:\n"
        "    for n in ast.walk(node):\n"
        "        if not isinstance(n, _ALLOWED):\n"
        "            raise ValueError(f'unsupported AST node: {type(n).__name__}')\n"
        "        if isinstance(n, ast.Call):\n"
        "            if not isinstance(n.func, ast.Name):\n"
        "                raise ValueError('only direct function-name calls are allowed')\n"
        "        if isinstance(n, ast.Subscript):\n"
        "            if isinstance(n.value, ast.Name) and n.value.id in {'__import__', 'eval', 'exec'}:\n"
        "                raise ValueError('unsafe subscript root')\n"
        "\n"
        "def _safe_eval(expr: str, scope: dict[str, object]) -> object:\n"
        "    tree = ast.parse(expr, mode='eval')\n"
        "    _validate(tree)\n"
        "    return eval(compile(tree, '<behavior>', 'eval'), {'__builtins__': {}}, scope)\n"
        "\n"
        "def _compare(lhs: float, op: str, rhs: float) -> bool:\n"
        "    if op == '>=': return lhs >= rhs\n"
        "    if op == '<=': return lhs <= rhs\n"
        "    if op == '==': return lhs == rhs\n"
        "    if op == '>': return lhs > rhs\n"
        "    if op == '<': return lhs < rhs\n"
        "    raise ValueError(f'unsupported operator: {op}')\n"
        "\n"
        "failures = []\n"
        "for idx, c in enumerate(criteria, start=1):\n"
        "    local_scope = dict(env)\n"
        "    if c.get('setup_call'):\n"
        "        setup = _safe_eval(c['setup_call'], local_scope)\n"
        "        local_scope['result'] = setup\n"
        "        local_scope['Q'] = setup\n"
        "        if isinstance(setup, dict):\n"
        "            for k, v in setup.items():\n"
        "                if isinstance(k, str) and re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k):\n"
        "                    local_scope[k] = v\n"
        "    if c.get('call_expr'):\n"
        "        out = _safe_eval(c['call_expr'], local_scope)\n"
        "        local_scope['result'] = out\n"
        "        local_scope['Q'] = out\n"
        "    lhs = float(_safe_eval(c['lhs_expr'], local_scope))\n"
        "    rhs = float(c['rhs'])\n"
        "    ok = _compare(lhs, c['op'], rhs)\n"
        "    if not ok:\n"
        "        failures.append(f\"[{idx}] {c['lhs_expr']} {c['op']} {rhs} failed (got {lhs})\")\n"
        "\n"
        "if failures:\n"
        "    print('Behavioral checks failed:')\n"
        "    for item in failures:\n"
        "        print(item)\n"
        "    raise SystemExit(1)\n"
        "print(f'behavior ok ({len(criteria)} check(s))')\n"
    )


class BehaviorCheck(ValidationCheck):
    """Validate prompt-defined numeric behavior checks against algorithm output."""

    name = "behavior"

    def __init__(
        self,
        *,
        user_request: str,
        runner: LocalRunner | None = None,
        timeout_sec: float = 20.0,
    ) -> None:
        self.user_request = user_request
        self.runner = runner or LocalRunner()
        self.timeout_sec = timeout_sec

    def run(self, repo_root: Path) -> ValidationResult:
        criteria = extract_behavior_criteria(self.user_request)
        if not criteria:
            return ValidationResult(
                name=self.name,
                command=["python", "-I", "-c", "# behavior skipped"],
                exit_code=0,
                stdout_excerpt="behavior skipped (no criteria found)",
                stderr_excerpt="",
                passed=True,
                skipped=True,
                duration_sec=0.0,
            )
        snippet = _build_runner_snippet(criteria)
        command = ["python", "-I", "-c", snippet]
        result = self.runner.run(command, repo_root, timeout_sec=self.timeout_sec)
        return ValidationResult(
            name=self.name,
            command=command,
            exit_code=result.exit_code,
            stdout_excerpt=result.stdout,
            stderr_excerpt=result.stderr,
            passed=result.passed,
            skipped=False,
            duration_sec=result.duration_sec,
            failure_type=FailureType.SEMANTIC if not result.passed else FailureType.UNKNOWN,
        )


__all__ = ["BehaviorCheck", "BehaviorCriterion", "extract_behavior_criteria"]
