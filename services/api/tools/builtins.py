"""Five mock daily-assistant tools.

All return deterministic mock payloads — no external services. They exist so
the assistant has something realistic to call and the UI has something to
render. Schemas are JSON Schema draft-07 (works for Anthropic, OpenAI, and
Google function-calling alike).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from .registry import Tool, ToolRegistry


# ── handlers ──────────────────────────────────────────────────────────────


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def schedule_call(participant: str, when: str, duration_min: int = 30) -> dict:
    return {
        "event_id": _short_id("evt"),
        "participant": participant,
        "when": when,
        "duration_min": duration_min,
        "link": "https://meet.example/mock",
        "status": "scheduled",
    }


def update_calendar(event_id: str, when: str, notes: str = "") -> dict:
    return {
        "event_id": event_id,
        "when": when,
        "notes": notes,
        "status": "updated",
    }


def list_calendar(start: str, end: str) -> dict:
    # Three deterministic events; the dates are advisory and free-text.
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return {
        "range": {"start": start, "end": end},
        "events": [
            {
                "event_id": "evt_aaaa1111",
                "title": "Standup",
                "when": (now + timedelta(hours=1)).isoformat(),
                "participant": "team",
            },
            {
                "event_id": "evt_bbbb2222",
                "title": "1:1 with Priya",
                "when": (now + timedelta(hours=4)).isoformat(),
                "participant": "Priya",
            },
            {
                "event_id": "evt_cccc3333",
                "title": "Design review",
                "when": (now + timedelta(days=1)).isoformat(),
                "participant": "design-team",
            },
        ],
    }


def set_reminder(text: str, when: str) -> dict:
    return {
        "reminder_id": _short_id("rem"),
        "text": text,
        "when": when,
        "status": "scheduled",
    }


def get_weather(location: str, units: str = "celsius") -> dict:
    return {
        "location": location,
        "temp": 22 if units == "celsius" else 72,
        "units": units,
        "condition": "partly cloudy",
        "source": "mock",
    }


# ── registration ──────────────────────────────────────────────────────────


def register_builtins(registry: ToolRegistry) -> None:
    """Populate `registry` with the five mock tools. Idempotent."""
    registry.register(
        Tool(
            name="schedule_call",
            description=(
                "Schedule a call with a participant at a specific ISO-8601 time. "
                "Returns the created event id and a join link."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "participant": {
                        "type": "string",
                        "description": "Name or email of the person to invite.",
                    },
                    "when": {
                        "type": "string",
                        "description": "ISO-8601 datetime (UTC) for the start of the call.",
                    },
                    "duration_min": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 240,
                        "default": 30,
                        "description": "Length of the call in minutes.",
                    },
                },
                "required": ["participant", "when"],
            },
            handler=schedule_call,
        )
    )

    registry.register(
        Tool(
            name="update_calendar",
            description="Move or edit an existing calendar event by id.",
            parameters={
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "Id of the event to edit."},
                    "when": {
                        "type": "string",
                        "description": "New ISO-8601 datetime for the event.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional note attached to the update.",
                        "default": "",
                    },
                },
                "required": ["event_id", "when"],
            },
            handler=update_calendar,
        )
    )

    registry.register(
        Tool(
            name="list_calendar",
            description="List calendar events between two ISO-8601 datetimes (UTC).",
            parameters={
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "ISO-8601 start of range."},
                    "end": {"type": "string", "description": "ISO-8601 end of range."},
                },
                "required": ["start", "end"],
            },
            handler=list_calendar,
        )
    )

    registry.register(
        Tool(
            name="set_reminder",
            description="Create a one-shot reminder.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Reminder text."},
                    "when": {
                        "type": "string",
                        "description": "ISO-8601 datetime when to fire the reminder.",
                    },
                },
                "required": ["text", "when"],
            },
            handler=set_reminder,
        )
    )

    registry.register(
        Tool(
            name="get_weather",
            description="Get current weather for a named location.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or place name (e.g. 'San Francisco').",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["location"],
            },
            handler=get_weather,
        )
    )
