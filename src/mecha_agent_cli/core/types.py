"""Shared enums and value objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class TaskType(StrEnum):
    """Supported task classes."""

    NEW_CODE = "NEW_CODE"
    EDIT_EXISTING_CODE = "EDIT_EXISTING_CODE"
    FIX_VALIDATION = "FIX_VALIDATION"
    REFACTOR = "REFACTOR"
    REVIEW_ONLY = "REVIEW_ONLY"
    EXPLAIN_ONLY = "EXPLAIN_ONLY"
    TEST_GENERATION = "TEST_GENERATION"
    UNKNOWN = "UNKNOWN"


class FailureType(StrEnum):
    """Primary validation failure taxonomy."""

    SYNTAX = "SYNTAX"
    IMPORT = "IMPORT"
    RUNTIME = "RUNTIME"
    TYPE = "TYPE"
    LINT = "LINT"
    TEST = "TEST"
    SEMANTIC = "SEMANTIC"
    PATCH = "PATCH"
    SECURITY = "SECURITY"
    UNKNOWN = "UNKNOWN"


class ValidationStatus(StrEnum):
    """Compact validation status values."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class WorkspacePaths:
    """Canonical workspace paths used by the runtime."""

    root: Path
    mecha_dir: Path
    memory_db: Path
    target_file: Path


JsonDict = dict[str, Any]
