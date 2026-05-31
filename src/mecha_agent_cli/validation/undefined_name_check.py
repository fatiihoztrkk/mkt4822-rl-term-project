"""Static undefined-name validation.

Catches obvious ``NameError`` risks early (e.g. referencing ``random`` without
an import) before runtime paths happen to execute those lines.
"""

from __future__ import annotations

import builtins
import symtable
from pathlib import Path

from mecha_agent_cli.core.types import FailureType
from mecha_agent_cli.validation.base import ValidationCheck
from mecha_agent_cli.validation.report import ValidationResult


class UndefinedNameCheck(ValidationCheck):
    """Detect unresolved global name references in ``algorithm.py``."""

    name = "undefined_name"

    def __init__(self, target_file: str = "algorithm.py") -> None:
        self.target_file = target_file

    def run(self, repo_root: Path) -> ValidationResult:
        path = repo_root / self.target_file
        command = ["python", "-m", "symtable", self.target_file]
        try:
            source = path.read_text(encoding="utf-8")
            table = symtable.symtable(source, str(path), "exec")
        except OSError as exc:
            return ValidationResult(
                name=self.name,
                command=command,
                exit_code=1,
                stdout_excerpt="",
                stderr_excerpt=str(exc),
                passed=False,
                duration_sec=0.0,
                failure_type=FailureType.TYPE,
            )
        except SyntaxError:
            return ValidationResult(
                name=self.name,
                command=command,
                exit_code=0,
                stdout_excerpt="skipped: syntax not yet valid",
                stderr_excerpt="",
                passed=True,
                duration_sec=0.0,
                skipped=True,
            )

        module_defs = _module_defined_names(table)
        unresolved = _collect_unresolved(table, module_defs)
        if unresolved:
            names = ", ".join(sorted(unresolved))
            return ValidationResult(
                name=self.name,
                command=command,
                exit_code=1,
                stdout_excerpt="",
                stderr_excerpt=f"Undefined names detected: {names}",
                passed=False,
                duration_sec=0.0,
                failure_type=FailureType.TYPE,
            )
        return ValidationResult(
            name=self.name,
            command=command,
            exit_code=0,
            stdout_excerpt="undefined-name check ok",
            stderr_excerpt="",
            passed=True,
            duration_sec=0.0,
        )


def _module_defined_names(table: symtable.SymbolTable) -> set[str]:
    names: set[str] = set()
    for sym in table.get_symbols():
        if sym.is_assigned() or sym.is_imported() or sym.is_parameter() or sym.is_namespace():
            names.add(sym.get_name())
    return names


def _collect_unresolved(table: symtable.SymbolTable, module_defs: set[str]) -> set[str]:
    builtins_set = set(dir(builtins))
    unresolved: set[str] = set()

    def walk(current: symtable.SymbolTable) -> None:
        for sym in current.get_symbols():
            name = sym.get_name()
            if name in builtins_set:
                continue
            if not sym.is_referenced():
                continue

            if current.get_type() == "module":
                if name not in module_defs:
                    unresolved.add(name)
                continue

            # For nested scopes, only unresolved globals are relevant here.
            if sym.is_local() or sym.is_parameter() or sym.is_imported() or sym.is_free() or sym.is_nonlocal():
                continue
            if name not in module_defs:
                unresolved.add(name)

        for child in current.get_children():
            walk(child)

    walk(table)
    return unresolved
