"""Spec-contract validation: enforce that required public functions exist.

The pipeline already covers SYNTAX → IMPORT → RUNTIME, but each of those
checks is satisfied by *any* code that compiles, imports and prints something.
A model can ship a 30-line stub that ignores the user's spec and still pass.

This check parses the user request for ``def NAME(arg1, arg2, ...) [-> Ret]``
patterns and verifies, via :mod:`ast`, that:

  1. each named function is defined at the top level of the target file
     (or as a top-level method on any class), and
  2. its parameter count meets the minimum required by the prompt
     (``*args`` / ``**kwargs`` count as flexible matches).

Generalises across families: anything the user writes as a python signature
becomes a hard contract. No NLP, no LLM call, sub-millisecond.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from mecha_agent_cli.core.types import FailureType
from mecha_agent_cli.validation.base import ValidationCheck
from mecha_agent_cli.validation.report import ValidationResult

# Match ``def NAME(args)`` anywhere in the prompt. Captures name + raw args.
# Matches both bare lines and indented examples (e.g. inside numbered lists).
_DEF_RE = re.compile(
    r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)",
    re.MULTILINE,
)
# Names that prompts often mention only as type hints / cross-references but
# that the model is not expected to define. Filtered out of the contract.
_INFRASTRUCTURE_NAMES = frozenset({"__init__", "lambda"})


@dataclass(frozen=True)
class FunctionSpec:
    """One function the user explicitly asked for in the prompt."""

    name: str
    min_positional: int
    has_var_positional: bool
    has_var_keyword: bool

    def matches(self, found: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Return True when ``found``'s signature can fulfil this spec."""
        args = found.args
        # Total positional slots the implementation accepts.
        positional_total = len(args.posonlyargs) + len(args.args)
        accepts_varargs = args.vararg is not None
        # An impl with *args trivially fits any positional arity.
        if accepts_varargs:
            return True
        return positional_total >= self.min_positional


def _count_params(raw_args: str) -> tuple[int, bool, bool]:
    """Return (min_positional, has_var_positional, has_var_keyword) for one signature."""
    raw = raw_args.strip()
    if not raw:
        return 0, False, False
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    has_var_positional = False
    has_var_keyword = False
    positional = 0
    for part in parts:
        head = part.split(":", 1)[0].split("=", 1)[0].strip()
        if head.startswith("**"):
            has_var_keyword = True
            continue
        if head.startswith("*"):
            has_var_positional = True
            continue
        if head in {"/", "self", "cls"}:
            continue
        # Only count parameters without defaults toward "minimum positional".
        # Heuristic: prompt-style signatures rarely include defaults.
        if "=" in part:
            continue
        positional += 1
    return positional, has_var_positional, has_var_keyword


def extract_required_functions(prompt: str) -> list[FunctionSpec]:
    """Extract :class:`FunctionSpec` entries from a user-request prompt.

    Deduplicates by name, keeping the *strictest* (largest ``min_positional``)
    signature per name. Returns an empty list when no ``def`` patterns are
    present in the prompt — in which case the contract check is a no-op.
    """
    seen: dict[str, FunctionSpec] = {}
    for match in _DEF_RE.finditer(prompt):
        name = match.group(1)
        if name in _INFRASTRUCTURE_NAMES or name.startswith("_"):
            continue
        positional, va, vk = _count_params(match.group(2))
        spec = FunctionSpec(
            name=name,
            min_positional=positional,
            has_var_positional=va,
            has_var_keyword=vk,
        )
        prior = seen.get(name)
        if prior is None or spec.min_positional > prior.min_positional:
            seen[name] = spec
    return list(seen.values())


def _collect_defined_functions(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return ``{name: FunctionDef}`` for every top-level def AND class method."""
    out: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    if not isinstance(tree, ast.Module):
        return out
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Methods are exposed under their bare name; if a top-level
                    # def with the same name also exists, the top-level wins.
                    out.setdefault(item.name, item)
    return out


@dataclass(frozen=True)
class ContractDiagnostic:
    """One line of complaint emitted to the model when the contract fails."""

    name: str
    reason: str

    def render(self) -> str:
        """Return a single human/LLM-readable line."""
        return f"  - {self.name}: {self.reason}"


def evaluate_contract(source: str, prompt: str) -> tuple[bool, list[ContractDiagnostic]]:
    """Return ``(passed, diagnostics)`` for ``source`` against ``prompt``.

    ``passed`` is ``True`` when every required function is present with
    matching minimum arity. Pure function — no I/O.
    """
    specs = extract_required_functions(prompt)
    if not specs:
        return True, []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, [ContractDiagnostic(name="<file>", reason=f"could not parse: {exc}")]
    defined = _collect_defined_functions(tree)
    diagnostics: list[ContractDiagnostic] = []
    for spec in specs:
        node = defined.get(spec.name)
        if node is None:
            diagnostics.append(
                ContractDiagnostic(
                    name=spec.name,
                    reason=(
                        f"required by the task prompt but not defined. Add `def {spec.name}(...)` at module scope."
                    ),
                )
            )
            continue
        if not spec.matches(node):
            actual = len(node.args.posonlyargs) + len(node.args.args)
            diagnostics.append(
                ContractDiagnostic(
                    name=spec.name,
                    reason=(
                        f"signature mismatch: prompt requires at least {spec.min_positional} "
                        f"positional parameter(s); the current definition accepts {actual}."
                    ),
                )
            )
    return not diagnostics, diagnostics


class SpecContractCheck(ValidationCheck):
    """AST-level check that all prompt-declared functions exist with matching arity."""

    name = "spec_contract"

    def __init__(self, target_file: str = "algorithm.py", user_request: str = "") -> None:
        self.target_file = target_file
        self.user_request = user_request

    def run(self, repo_root: Path) -> ValidationResult:
        """Run the contract check; passes silently when the prompt has no signatures."""
        path = repo_root / self.target_file
        command = ["python", "-c", "ast.parse(...)"]
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            return ValidationResult(
                name=self.name,
                command=command,
                exit_code=1,
                stdout_excerpt="",
                stderr_excerpt=str(exc),
                passed=False,
                duration_sec=0.0,
                failure_type=FailureType.SEMANTIC,
            )
        passed, diagnostics = evaluate_contract(source, self.user_request)
        if passed:
            return ValidationResult(
                name=self.name,
                command=command,
                exit_code=0,
                stdout_excerpt=f"contract ok ({len(extract_required_functions(self.user_request))} sig(s))",
                stderr_excerpt="",
                passed=True,
                duration_sec=0.0,
            )
        message = "Spec contract violations:\n" + "\n".join(d.render() for d in diagnostics)
        return ValidationResult(
            name=self.name,
            command=command,
            exit_code=1,
            stdout_excerpt="",
            stderr_excerpt=message,
            passed=False,
            duration_sec=0.0,
            failure_type=FailureType.SEMANTIC,
        )


__all__ = [
    "ContractDiagnostic",
    "FunctionSpec",
    "SpecContractCheck",
    "evaluate_contract",
    "extract_required_functions",
]
