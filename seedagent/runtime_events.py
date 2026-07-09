from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


EventHandler = Callable[[RuntimeEvent], None]


def _one_line(text: str, *, max_length: int = 180) -> str:
    line = " ".join(text.splitlines()[0].split()) if text.splitlines() else ""
    if len(line) <= max_length:
        return line
    return line[: max_length - 3].rstrip() + "..."


def _terminal_error_summary(message: Any) -> tuple[str, str]:
    text = str(message or "").strip()
    if not text:
        return "error", ""

    if "Authorized imports are:" in text:
        return "forbidden", _one_line(text.split("Authorized imports are:", 1)[0].rstrip())

    forbidden_markers = (
        "InterpreterError: Import of ",
        "Forbidden access to ",
        "Forbidden function ",
        "Shell command blocked",
        "not allowed by shell policy",
        "not allowed by sandbox policy",
    )
    if any(marker in text for marker in forbidden_markers):
        return "forbidden", _one_line(text)

    return "error", text


class RuntimeEventRouter:
    """Small synchronous event bus for terminal-visible agent runtime events."""

    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def emit(self, event_type: str, **payload: Any) -> RuntimeEvent:
        event = RuntimeEvent(type=event_type, payload=payload)
        for handler in list(self._handlers):
            handler(event)
        return event


class TerminalRenderer:
    """Human-oriented renderer for RuntimeEventRouter events."""

    def __init__(self, *, show_code: bool = False, show_model_trace: bool = False) -> None:
        self.show_code = show_code
        self.show_model_trace = show_model_trace
        self._commentary_open = False
        self._final_open = False
        self._streamed_final_text = ""
        self._last_error_display: tuple[str, str] | None = None

    def __call__(self, event: RuntimeEvent) -> None:
        handler = getattr(self, f"_on_{event.type}", None)
        if event.type != "error":
            self._last_error_display = None
        if handler is None:
            if self.show_model_trace:
                print(f"[{event.type}] {event.payload}")
            return
        handler(event)

    def _ensure_commentary_prefix(self) -> None:
        if not self._commentary_open:
            print("agent> ", end="", flush=True)
            self._commentary_open = True

    def _close_commentary_line(self) -> None:
        if self._commentary_open:
            print()
            self._commentary_open = False

    def _ensure_final_prefix(self) -> None:
        if not self._final_open:
            self._close_commentary_line()
            print("agent> ", end="", flush=True)
            self._final_open = True

    def _close_final_line(self) -> None:
        if self._final_open:
            print()
            self._final_open = False

    def _close_open_lines(self) -> None:
        self._close_commentary_line()
        self._close_final_line()

    def _on_run_started(self, event: RuntimeEvent) -> None:
        self._close_open_lines()
        self._streamed_final_text = ""
        self._last_error_display = None

    def _on_commentary_delta(self, event: RuntimeEvent) -> None:
        text = event.payload.get("text") or ""
        if not text:
            return
        self._ensure_commentary_prefix()
        print(text, end="", flush=True)

    def _on_commentary_finished(self, event: RuntimeEvent) -> None:
        self._close_commentary_line()

    def _on_final_delta(self, event: RuntimeEvent) -> None:
        text = event.payload.get("text") or ""
        if not text:
            return
        self._streamed_final_text += text
        self._ensure_final_prefix()
        print(text, end="", flush=True)

    def _on_final_finished(self, event: RuntimeEvent) -> None:
        self._close_final_line()

    def _on_model_output_finished(self, event: RuntimeEvent) -> None:
        self._close_open_lines()

    def _on_code_started(self, event: RuntimeEvent) -> None:
        self._close_open_lines()
        code = event.payload.get("code") or ""
        if self.show_code and code:
            print("code>")
            print(code.rstrip())
        else:
            first_line = code.strip().splitlines()[0] if code.strip() else "python"
            print(f"code> {first_line} ...")

    def _on_tool_started(self, event: RuntimeEvent) -> None:
        self._close_open_lines()
        print(f"tool> {event.payload.get('name', 'tool')} started")

    def _on_tool_finished(self, event: RuntimeEvent) -> None:
        self._close_open_lines()
        name = event.payload.get("name", "tool")
        print(f"tool> {name} finished")

    def _on_execution_started(self, event: RuntimeEvent) -> None:
        self._close_open_lines()
        print("exec> running")

    def _on_execution_finished(self, event: RuntimeEvent) -> None:
        self._close_open_lines()
        if event.payload.get("is_final_answer"):
            return
        print("exec> finished")

    def _on_shell_started(self, event: RuntimeEvent) -> None:
        self._close_open_lines()
        command = str(event.payload.get("command", "")).strip()
        first_line = command.splitlines()[0] if command else "shell"
        print(f"shell> {first_line} ...")

    def _on_shell_finished(self, event: RuntimeEvent) -> None:
        self._close_open_lines()
        return_code = event.payload.get("return_code")
        print(f"shell> finished exit={return_code}")

    def _on_final_response(self, event: RuntimeEvent) -> None:
        self._close_open_lines()
        text = str(event.payload.get("text", ""))
        if self._streamed_final_text and text.strip() == self._streamed_final_text.strip():
            return
        print(f"agent> {text}")

    def _on_error(self, event: RuntimeEvent) -> None:
        self._close_open_lines()
        label, message = _terminal_error_summary(event.payload.get("message", ""))
        display = (label, message)
        if display == self._last_error_display:
            return
        self._last_error_display = display
        print(f"{label}> {message}")


__all__ = ["RuntimeEvent", "RuntimeEventRouter", "TerminalRenderer"]
