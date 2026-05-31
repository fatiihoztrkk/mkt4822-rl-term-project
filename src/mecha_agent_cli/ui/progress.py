"""Live Rich observer: spinner + streamed thinking output.

Behavior:
  * ``running``  — start a Rich spinner on its own line with an elapsed
    counter that updates twice a second.
  * ``stream``   — pause the spinner, print streamed thinking text live in a
    boxed ``│ `` block; supports newline-aware wrapping so multi-paragraph
    reasoning is shown line by line, not crammed into one line.
  * any other status (``done``, ``success``, ``fail``, ``error``, ...) — close
    any open thinking block, stop the spinner, print a final glyph + elapsed
    line for that stage.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager

from rich.console import Console
from rich.status import Status

from mecha_agent_cli.agent.observer import AgentProgressEvent

_RUNNING = "running"
_STREAM = "stream"
_TERMINAL_GLYPHS: dict[str, tuple[str, str]] = {
    "success": ("✓", "green"),
    "done": ("✓", "green"),
    "pass": ("✓", "green"),
    "fail": ("✗", "red"),
    "failed": ("✗", "red"),
    "error": ("✗", "red"),
    "retry": ("↻", "yellow"),
    "repair": ("↻", "yellow"),
    "warning": ("!", "yellow"),
}


@contextmanager
def null_progress() -> Generator[None]:
    """No-op progress context for deterministic tests."""
    yield


class RichAgentObserver:
    """Render agent progress events with live spinner and streamed thinking."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._status: Status | None = None
        self._current: AgentProgressEvent | None = None
        self._start: float = 0.0
        self._stop_tick = threading.Event()
        self._ticker: threading.Thread | None = None
        self._lock = threading.Lock()
        # thinking-stream state
        self._thinking_open = False
        self._thinking_at_line_start = True
        self._thinking_chars = 0
        self._thinking_started_at = 0.0

    def __call__(self, event: AgentProgressEvent) -> None:
        """Handle one progress event."""
        if event.status == _STREAM:
            self._on_stream(event)
            return
        if self._thinking_open:
            self._close_thinking()
        if event.status == _RUNNING:
            self._begin(event)
        else:
            self._end(event)

    # -- spinner lifecycle ------------------------------------------------

    def _begin(self, event: AgentProgressEvent) -> None:
        self._close_silently()
        self._current = event
        self._start = time.monotonic()
        text = self._compose(event, elapsed=0.0)
        status = self.console.status(text, spinner="dots")
        status.start()
        self._status = status
        self._stop_tick = threading.Event()
        ticker = threading.Thread(target=self._tick_loop, name="mecha-agent-progress", daemon=True)
        self._ticker = ticker
        ticker.start()

    def _end(self, event: AgentProgressEvent) -> None:
        elapsed = self._stop_spinner()
        self._print_terminal(event, elapsed)

    def _close_silently(self) -> None:
        if self._status is None or self._current is None:
            return
        elapsed = self._stop_spinner()
        previous = self._current
        head = _format_attempt(previous)
        detail = f" — {previous.detail}" if previous.detail else ""
        self.console.print(f"[dim]·[/dim] {previous.stage}{head} [dim]{elapsed:5.1f}s[/dim]{detail}")

    def _stop_spinner(self) -> float:
        with self._lock:
            status = self._status
            ticker = self._ticker
            self._status = None
        if status is None:
            return 0.0
        self._stop_tick.set()
        if ticker is not None:
            ticker.join(timeout=1.5)
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            status.stop()
        return time.monotonic() - self._start

    def _pause_spinner_for_thinking(self) -> None:
        """Stop the spinner without losing ``_current`` so we can resume after thinking."""
        with self._lock:
            status = self._status
            ticker = self._ticker
            self._status = None
        if status is None:
            return
        self._stop_tick.set()
        if ticker is not None:
            ticker.join(timeout=1.5)
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            status.stop()

    def _resume_spinner_after_thinking(self) -> None:
        if self._current is None:
            return
        text = self._compose(self._current, elapsed=time.monotonic() - self._start)
        status = self.console.status(text, spinner="dots")
        status.start()
        self._status = status
        self._stop_tick = threading.Event()
        ticker = threading.Thread(target=self._tick_loop, name="mecha-agent-progress", daemon=True)
        self._ticker = ticker
        ticker.start()

    @contextmanager
    def pause(self) -> Generator[None]:
        """Temporarily stop the spinner so callers can prompt the user.

        The currently-running stage is preserved; on exit the spinner restarts
        with the same stage label and elapsed counter continuing from where it
        left off.
        """
        self._pause_spinner_for_thinking()
        try:
            yield
        finally:
            self._resume_spinner_after_thinking()

    def _tick_loop(self) -> None:
        while not self._stop_tick.wait(0.5):
            with self._lock:
                status = self._status
                current = self._current
            if status is None or current is None:
                return
            elapsed = time.monotonic() - self._start
            try:
                status.update(self._compose(current, elapsed=elapsed))
            except Exception:  # pragma: no cover - defensive
                return

    # -- thinking stream --------------------------------------------------

    def _on_stream(self, event: AgentProgressEvent) -> None:
        chunk = event.detail
        if not chunk:
            return
        if not self._thinking_open:
            self._open_thinking()
        self._write_thinking_chunk(chunk)

    def _open_thinking(self) -> None:
        self._pause_spinner_for_thinking()
        self._thinking_open = True
        self._thinking_at_line_start = True
        self._thinking_chars = 0
        self._thinking_started_at = time.monotonic()
        self.console.print("[bold magenta]╭─ thinking[/bold magenta]")

    def _close_thinking(self) -> None:
        if not self._thinking_at_line_start:
            sys.stdout.write("\n")
            sys.stdout.flush()
        elapsed = time.monotonic() - self._thinking_started_at
        self.console.print(
            f"[bold magenta]╰─ end thinking[/bold magenta] [dim]({self._thinking_chars} chars · {elapsed:.1f}s)[/dim]"
        )
        self._thinking_open = False
        self._thinking_at_line_start = True
        self._thinking_chars = 0
        # resume spinner for the still-running stage so the user sees activity
        # continue while final tokens stream off-channel
        self._resume_spinner_after_thinking()

    def _write_thinking_chunk(self, chunk: str) -> None:
        # Write directly to stdout (no Rich markup) so model text is preserved
        # exactly. We prefix every line with ``│ `` for visual grouping.
        out = sys.stdout
        prefix = "\033[2m│\033[0m "  # dim │ followed by space; ANSI dim
        for ch in chunk:
            if self._thinking_at_line_start:
                out.write(prefix)
                self._thinking_at_line_start = False
            if ch == "\n":
                out.write("\n")
                self._thinking_at_line_start = True
            else:
                out.write(ch)
            self._thinking_chars += 1
        out.flush()

    # -- rendering --------------------------------------------------------

    def _compose(self, event: AgentProgressEvent, *, elapsed: float) -> str:
        head = _format_attempt(event)
        elapsed_str = f" [cyan]{elapsed:5.1f}s[/cyan]"
        detail = f" — {event.detail}" if event.detail else ""
        return f"[bold cyan]{event.stage}[/bold cyan]{head}{elapsed_str}{detail}"

    def _print_terminal(self, event: AgentProgressEvent, elapsed: float) -> None:
        glyph, style = _TERMINAL_GLYPHS.get(event.status, ("•", "cyan"))
        head = _format_attempt(event)
        detail = f" — {event.detail}" if event.detail else ""
        elapsed_str = f" [{style}]{elapsed:.1f}s[/{style}]" if elapsed >= 0.05 else ""
        self.console.print(f"[{style}]{glyph}[/] {event.stage}{head}{elapsed_str}{detail}")


def _format_attempt(event: AgentProgressEvent) -> str:
    parts: list[str] = []
    if event.attempt_index is not None:
        parts.append(f"#{event.attempt_index + 1}")
    if event.total_attempts is not None and event.max_attempts is not None:
        limit = "∞" if event.max_attempts <= 0 else str(event.max_attempts)
        parts.append(f"{event.total_attempts}/{limit}")
    return f" [dim]({' '.join(parts)})[/dim]" if parts else ""
