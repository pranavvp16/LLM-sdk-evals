"""Shared pytest fixtures.

These fixtures keep tests offline: no real LLM calls, no real Postgres, no real
Redis, no real ClickHouse. Each downstream test that needs a live FastAPI app
should compose them with `httpx.AsyncClient` + `ASGITransport`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ── reusable fake LLM provider ────────────────────────────────────────────


class FakeStream:
    """Minimal async iterator emitting the SDK's StreamEvent shape.

    Tasks 2/3 can swap real providers for this fake by monkeypatching
    `LLMWrapper._providers` or constructing a wrapper with no api_keys.
    """

    def __init__(self, text: str = "hello world"):
        self.text = text

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        from sdk.types import (
            AssistantMessage,
            StreamEventDone,
            StreamEventStart,
            StreamEventTextDelta,
            Usage,
        )

        yield StreamEventStart()
        for token in self.text.split():
            yield StreamEventTextDelta(delta=token + " ")
        yield StreamEventDone(
            message=AssistantMessage(
                content=self.text,
                usage=Usage(input_tokens=5, output_tokens=len(self.text.split())),
            )
        )
