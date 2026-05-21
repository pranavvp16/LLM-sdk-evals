"""Chat router tests — Task 2.

These are placeholders that import the app and assert the routes are wired up.
Fill in real assertions (against a real Postgres + a fake LLM provider) as the
chat router is implemented.
"""

from __future__ import annotations

import pytest


def test_app_imports() -> None:
    """The FastAPI app must build without raising."""
    from services.api.main import create_app

    app = create_app()
    routes = {r.path for r in app.routes}
    assert "/health" in routes
    assert "/chat/stream" in routes
    assert "/conversations" in routes


@pytest.mark.skip(reason="Implement with httpx.AsyncClient + ASGITransport + real Postgres in Task 2")
async def test_chat_stream_creates_conversation() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="Implement with cancel/resume round-trip in Task 2")
async def test_cancel_and_resume_conversation() -> None:
    raise NotImplementedError
