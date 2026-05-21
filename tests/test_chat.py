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


def test_rebuild_history_restores_tool_names() -> None:
    from sdk import AssistantMessage, ToolResultMessage, UserMessage

    from services.api.routers.chat import _rebuild_history

    history = [
        {"role": "user", "content": "book a call"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "name": "schedule_call",
                    "arguments": {"participant": "Ankur", "when": "2026-05-22T15:00:00Z"},
                }
            ],
        },
        {
            "role": "tool_result",
            "content": '{"event_id": "evt_abc"}',
            "tool_call_id": "tc_1",
            "is_error": False,
        },
        {"role": "assistant", "content": "Done, the call is booked."},
    ]

    messages = _rebuild_history(history)

    assert isinstance(messages[0], UserMessage)
    assert isinstance(messages[1], AssistantMessage)
    assert isinstance(messages[2], ToolResultMessage)
    assert messages[2].results[0].name == "schedule_call"
    assert isinstance(messages[3], AssistantMessage)


def test_rebuild_history_drops_orphaned_tool_results() -> None:
    from sdk import AssistantMessage, ToolResultMessage, UserMessage

    from services.api.routers.chat import _rebuild_history

    history = [
        {"role": "user", "content": "first turn"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "tc_1", "name": "get_weather", "arguments": {"location": "NYC"}},
            ],
        },
        {
            "role": "tool_result",
            "content": '{"temp_c": 22}',
            "tool_call_id": "tc_1",
            "is_error": False,
        },
    ]

    messages = _rebuild_history(history)

    assert len(messages) == 2
    assert isinstance(messages[0], UserMessage)
    assert isinstance(messages[1], AssistantMessage)
    assert messages[1].content == []
    assert messages[1].tool_calls == []
    assert not any(isinstance(m, ToolResultMessage) for m in messages)


def test_rebuild_history_keeps_assistant_text_when_orphaned() -> None:
    from sdk import AssistantMessage, TextContent, UserMessage

    from services.api.routers.chat import _rebuild_history

    history = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "Checking weather now…",
            "tool_calls": [
                {"id": "tc_1", "name": "get_weather", "arguments": {"location": "NYC"}},
            ],
        },
        {
            "role": "tool_result",
            "content": '{"temp_c": 22}',
            "tool_call_id": "tc_1",
            "is_error": False,
        },
    ]

    messages = _rebuild_history(history)

    assert len(messages) == 2
    assert messages[0] == UserMessage(content="hi")
    assert messages[1].content == [TextContent(text="Checking weather now…")]
    assert messages[1].tool_calls == []


@pytest.mark.skip(reason="Implement with httpx.AsyncClient + ASGITransport + real Postgres in Task 2")
async def test_chat_stream_creates_conversation() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="Implement with cancel/resume round-trip in Task 2")
async def test_cancel_and_resume_conversation() -> None:
    raise NotImplementedError
