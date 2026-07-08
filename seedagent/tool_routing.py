from __future__ import annotations

from typing import Any

from smolagents import Tool
from smolagents.tools import BaseTool

try:
    from .runtime_events import RuntimeEventRouter
except ImportError:
    from runtime_events import RuntimeEventRouter


class RoutedTool(BaseTool):
    """Thin Tool wrapper that emits start/end/error events around an existing Tool."""

    def __init__(self, wrapped: Tool, router: RuntimeEventRouter) -> None:
        self.wrapped = wrapped
        self.router = router
        self.name = wrapped.name
        self.description = wrapped.description
        self.inputs = wrapped.inputs
        self.output_type = wrapped.output_type
        self.output_schema = getattr(wrapped, "output_schema", None)
        self.is_initialized = False

    def setup(self) -> None:
        if not self.wrapped.is_initialized:
            self.wrapped.setup()
        self.is_initialized = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.is_initialized:
            self.setup()
        self.router.emit("tool_started", name=self.name, args=args, kwargs=kwargs)
        try:
            output = self.wrapped(*args, **kwargs)
        except Exception as exc:
            self.router.emit("error", message=f"{self.name} failed: {exc}")
            raise
        self.router.emit("tool_finished", name=self.name, output=output)
        return output

    def to_code_prompt(self) -> str:
        return self.wrapped.to_code_prompt()

    def to_tool_calling_prompt(self) -> str:
        return self.wrapped.to_tool_calling_prompt()


def route_tools(tools: list[Tool], router: RuntimeEventRouter) -> list[Tool]:
    return [RoutedTool(tool, router) for tool in tools]


__all__ = ["RoutedTool", "route_tools"]
