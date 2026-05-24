"""Shared pytest fixtures.

These fixtures keep tests offline: no real LLM calls, no real Postgres, no real
Redis, no real ClickHouse. Each downstream test that needs a live FastAPI app
should compose them with `httpx.AsyncClient` + `ASGITransport`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ── reusable fake LLM provider ────────────────────────────────────────────


class FakeJudgeWrapper:
    """Mimics ``LLMWrapper.complete()`` with per-model scripted JSON responses.

    Used by ``tests/test_eval_dag.py``. Each judge in the DAG panel pops the
    next response off its own FIFO queue, so a multi-judge run is fully
    deterministic.

    Responses are dicts (auto JSON-serialized into the assistant text) OR
    raw strings (sent as-is — use for invalid-JSON / parse-failure cases).

    Keys may be either ``"<provider>/<id>"`` or just ``"<id>"``.
    """

    def __init__(self, responses_by_model: dict[str, list[dict | list | str]]):
        self._queues = {k: list(v) for k, v in responses_by_model.items()}
        self.calls: list[str] = []   # for debugging assertions

    def _resolve(self, model) -> list:
        for key in (f"{model.provider}/{model.id}", model.id):
            if key in self._queues:
                return self._queues[key]
        raise RuntimeError(
            f"FakeJudgeWrapper: no scripted responses for {model.provider}/{model.id}; "
            f"have: {list(self._queues.keys())}"
        )

    async def complete(self, model, ctx, session_id: str = "", conversation_id: str = ""):
        from sdk.types import AssistantMessage, TextContent, Usage

        queue = self._resolve(model)
        if not queue:
            raise RuntimeError(
                f"FakeJudgeWrapper: queue exhausted for {model.provider}/{model.id}"
            )
        payload = queue.pop(0)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls.append(f"{model.provider}/{model.id}:{text[:80]}")
        return AssistantMessage(
            content=[TextContent(text=text)],
            usage=Usage(input_tokens=10, output_tokens=20),
        )


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
