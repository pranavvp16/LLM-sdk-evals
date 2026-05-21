"""Unit tests for the in-process tool registry and the mock builtins."""

from __future__ import annotations

import pytest

from services.api.tools.builtins import register_builtins
from services.api.tools.registry import Tool, ToolRegistry, ToolUnknownError


# ── basics ────────────────────────────────────────────────────────────────


def test_tool_schema_matches_normalize_expectations() -> None:
    tool = Tool(
        name="echo",
        description="echo back the input",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=lambda text: text,
    )
    schema = tool.schema()
    assert schema == {
        "name": "echo",
        "description": "echo back the input",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
    }


def test_registry_schemas_returns_in_registration_order() -> None:
    r = ToolRegistry()
    r.register(Tool(name="a", description="", parameters={"type": "object"}, handler=lambda: 1))
    r.register(Tool(name="b", description="", parameters={"type": "object"}, handler=lambda: 2))
    assert [s["name"] for s in r.schemas()] == ["a", "b"]


# ── sync vs async handlers ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_registry_call_invokes_sync_handler_off_main_thread() -> None:
    captured: dict = {}

    def handler(x: int, y: int) -> dict:
        captured["called"] = True
        return {"sum": x + y}

    r = ToolRegistry()
    r.register(Tool(name="add", description="", parameters={"type": "object"}, handler=handler))
    result = await r.call("add", {"x": 1, "y": 2})
    assert result == {"sum": 3}
    assert captured["called"] is True


@pytest.mark.asyncio
async def test_registry_call_invokes_async_handler_directly() -> None:
    async def handler(name: str) -> dict:
        return {"hello": name}

    r = ToolRegistry()
    r.register(Tool(name="greet", description="", parameters={"type": "object"}, handler=handler))
    assert await r.call("greet", {"name": "ollive"}) == {"hello": "ollive"}


@pytest.mark.asyncio
async def test_registry_call_unknown_tool_raises_tool_unknown_error() -> None:
    r = ToolRegistry()
    with pytest.raises(ToolUnknownError):
        await r.call("nope", {})


# ── builtins ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_builtins_loads_all_five_tools() -> None:
    r = ToolRegistry()
    register_builtins(r)
    assert set(r.names()) == {
        "schedule_call",
        "update_calendar",
        "list_calendar",
        "set_reminder",
        "get_weather",
    }


@pytest.mark.asyncio
async def test_schedule_call_returns_event_id_and_link() -> None:
    r = ToolRegistry()
    register_builtins(r)
    result = await r.call(
        "schedule_call",
        {"participant": "Ankur", "when": "2026-06-01T15:00:00Z", "duration_min": 45},
    )
    assert result["participant"] == "Ankur"
    assert result["duration_min"] == 45
    assert result["event_id"].startswith("evt_")
    assert result["link"].startswith("https://")


@pytest.mark.asyncio
async def test_get_weather_respects_units() -> None:
    r = ToolRegistry()
    register_builtins(r)
    c = await r.call("get_weather", {"location": "SF"})
    f = await r.call("get_weather", {"location": "SF", "units": "fahrenheit"})
    assert c["units"] == "celsius" and c["temp"] == 22
    assert f["units"] == "fahrenheit" and f["temp"] == 72


@pytest.mark.asyncio
async def test_list_calendar_returns_three_events() -> None:
    r = ToolRegistry()
    register_builtins(r)
    result = await r.call(
        "list_calendar",
        {"start": "2026-05-21T00:00:00Z", "end": "2026-05-23T00:00:00Z"},
    )
    assert isinstance(result["events"], list)
    assert len(result["events"]) == 3
    assert {e["event_id"] for e in result["events"]} == {
        "evt_aaaa1111",
        "evt_bbbb2222",
        "evt_cccc3333",
    }


# ── schema sanity for SDK normalize ───────────────────────────────────────


def test_builtin_schemas_round_trip_through_anthropic_normalize() -> None:
    from sdk.normalize import to_anthropic_tools

    r = ToolRegistry()
    register_builtins(r)
    out = to_anthropic_tools(r.schemas())
    assert len(out) == 5
    assert all("input_schema" in t for t in out)
    assert all(t["input_schema"]["type"] == "object" for t in out)


def test_builtin_schemas_round_trip_through_openai_normalize() -> None:
    from sdk.normalize import to_openai_tools

    r = ToolRegistry()
    register_builtins(r)
    out = to_openai_tools(r.schemas())
    assert len(out) == 5
    assert all(t["type"] == "function" for t in out)
    assert all("parameters" in t["function"] for t in out)
