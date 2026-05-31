"""Tests for the fenced code-block extractor."""

from __future__ import annotations

from mecha_agent_cli.llm.code_extractor import extract_python_code, strip_thinking


def test_extract_simple_python_block() -> None:
    response = "Here you go:\n```python\ndef solve(n: int) -> int:\n    return n + 1\n```\n"
    extracted = extract_python_code(response)
    assert extracted is not None
    assert extracted.closed_fence is True
    assert extracted.language == "python"
    assert "def solve" in extracted.code
    assert extracted.code.endswith("\n")


def test_extract_handles_unclosed_fence() -> None:
    response = "Here is the truncated answer:\n```python\ndef solve(x: int) -> int:\n    return x\n"
    extracted = extract_python_code(response)
    assert extracted is not None
    assert extracted.closed_fence is False
    assert "def solve" in extracted.code


def test_extract_prefers_python_tagged_block_over_others() -> None:
    response = "Some shell:\n```bash\necho hi\n```\nThen Python:\n```python\nclass Foo:\n    pass\n```\n"
    extracted = extract_python_code(response)
    assert extracted is not None
    assert "class Foo" in extracted.code
    assert extracted.language == "python"


def test_extract_returns_none_when_no_code() -> None:
    assert extract_python_code("Just a sentence with no code blocks at all.") is None


def test_extract_handles_thinking_prefix() -> None:
    response = "<think>let me plan this out carefully</think>\n```python\ndef solve(): return 1\n```\n"
    extracted = extract_python_code(response)
    assert extracted is not None
    assert "def solve" in extracted.code


def test_strip_thinking_handles_unclosed_block() -> None:
    cleaned = strip_thinking("<think>partial reasoning without close")
    assert cleaned == ""


def test_extract_falls_back_to_raw_python_text() -> None:
    response = "def solve(n: int) -> int:\n    return n * 2\n"
    extracted = extract_python_code(response)
    assert extracted is not None
    assert "def solve" in extracted.code
    assert extracted.closed_fence is False
