"""Input / output / tool-argument filters.

Each function is pure and deterministic — no network, no model calls.
The chat router calls them inline and emits a `guardrail_block` SSE event
when a check fails.
"""

from __future__ import annotations

from dataclasses import dataclass

from services.api.guardrails.patterns import (
    BANNED_OUTPUT_PATTERNS,
    CONTROL_CHAR_PATTERN,
    ISO_8601_PATTERN,
    MAX_INPUT_CHARS,
    PROMPT_INJECTION_PATTERNS,
)


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str | None = None

    @classmethod
    def ok(cls) -> "GuardrailResult":
        return cls(allowed=True)

    @classmethod
    def block(cls, reason: str) -> "GuardrailResult":
        return cls(allowed=False, reason=reason)


def check_input(text: str) -> GuardrailResult:
    """Validate a user-supplied message before it reaches the model."""
    if not text:
        return GuardrailResult.block("empty input")
    if len(text) > MAX_INPUT_CHARS:
        return GuardrailResult.block(f"input exceeds {MAX_INPUT_CHARS} chars")
    if CONTROL_CHAR_PATTERN.search(text):
        return GuardrailResult.block("input contains control characters")
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return GuardrailResult.block("prompt-injection pattern detected")
    return GuardrailResult.ok()


def check_output(text: str) -> GuardrailResult:
    """Validate an assistant message before it's persisted / sent."""
    for pattern in BANNED_OUTPUT_PATTERNS:
        if pattern.search(text):
            return GuardrailResult.block("output contains banned pattern")
    return GuardrailResult.ok()


def validate_tool_args(name: str, args: dict) -> GuardrailResult:
    """Tighten beyond the registered JSON schema.

    The mock tools accept ISO-8601 `when` timestamps and a `duration_min`
    bound. The chat router calls this after the model produces a tool_call
    but before dispatching to the registry.
    """
    if not isinstance(args, dict):
        return GuardrailResult.block(f"{name}: arguments must be an object")

    when = args.get("when")
    if isinstance(when, str) and when and not ISO_8601_PATTERN.match(when):
        return GuardrailResult.block(f"{name}: 'when' is not ISO-8601")

    duration = args.get("duration_min")
    if duration is not None:
        if not isinstance(duration, int) or duration < 5 or duration > 240:
            return GuardrailResult.block(
                f"{name}: 'duration_min' must be an int in [5, 240]"
            )

    for key in ("participant", "text", "location"):
        value = args.get(key)
        if isinstance(value, str) and CONTROL_CHAR_PATTERN.search(value):
            return GuardrailResult.block(f"{name}: '{key}' contains control chars")

    return GuardrailResult.ok()
