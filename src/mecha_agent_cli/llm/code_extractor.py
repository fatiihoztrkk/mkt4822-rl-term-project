"""Extract Python source from raw LLM responses.

Inspired by pythalab-cortex's fenced-block normalization. Supports closed
``\u0060\u0060\u0060python ... \u0060\u0060\u0060`` blocks, generic fenced blocks, and
truncated/unclosed fences (common when generation stops early on small models).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
_UNCLOSED_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n([\s\S]*?)\Z", re.DOTALL)
_PYTHON_LANG_TAGS: frozenset[str] = frozenset({"python", "py", "python3", ""})


@dataclass(frozen=True)
class ExtractedCode:
    """Python source extracted from a model response."""

    code: str
    language: str
    closed_fence: bool
    raw_response: str


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from a model response."""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    return cleaned.strip()


def extract_python_code(response_text: str) -> ExtractedCode | None:
    """Return the largest Python code block in ``response_text``.

    The function prefers blocks tagged ``python``/``py``/``python3`` and falls back
    to untagged fenced blocks whose contents parse as Python. If a closing fence is
    missing the trailing block is taken as-is. Returns ``None`` when no code block
    is recognized.
    """
    cleaned = strip_thinking(response_text)
    candidates: list[tuple[str, str, bool]] = []
    for match in _FENCE_RE.finditer(cleaned):
        language = (match.group(1) or "").strip().lower()
        body = match.group(2).strip("\n")
        candidates.append((language, body, True))
    if not candidates:
        unclosed = _UNCLOSED_FENCE_RE.search(cleaned)
        if unclosed is not None:
            language = (unclosed.group(1) or "").strip().lower()
            body = unclosed.group(2).strip("\n")
            candidates.append((language, body, False))
    if not candidates:
        if _looks_like_python(cleaned):
            return ExtractedCode(
                code=_normalize(cleaned),
                language="python",
                closed_fence=False,
                raw_response=response_text,
            )
        return None
    typed = [item for item in candidates if item[0] in _PYTHON_LANG_TAGS]
    pool = typed if typed else candidates
    pool.sort(key=lambda item: (item[2], len(item[1])), reverse=True)
    language, body, closed = pool[0]
    return ExtractedCode(
        code=_normalize(body),
        language=language or "python",
        closed_fence=closed,
        raw_response=response_text,
    )


def _normalize(code: str) -> str:
    """Trim and ensure a single trailing newline."""
    cleaned = code.strip()
    return cleaned + "\n" if cleaned else ""


def _looks_like_python(text: str) -> bool:
    """Cheap heuristic: parses as Python AND defines at least one function or class."""
    if not text.strip():
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) for node in tree.body)
