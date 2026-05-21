"""In-process tool calling — no MCP.

A `Tool` is a plain Python callable wrapped with a JSON-Schema description.
`ToolRegistry` collects them, exposes provider-agnostic schemas (which
`sdk/normalize.py` translates to Anthropic/OpenAI/Google wire formats), and
executes a call by name.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


HandlerResult = Any
Handler = Callable[..., HandlerResult] | Callable[..., Awaitable[HandlerResult]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Handler

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolUnknownError(KeyError):
    """Raised when the model asks for a tool we don't have."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    async def call(self, name: str, args: dict[str, Any]) -> HandlerResult:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolUnknownError(name)
        handler = tool.handler
        if inspect.iscoroutinefunction(handler):
            return await handler(**args)
        return await asyncio.to_thread(handler, **args)
