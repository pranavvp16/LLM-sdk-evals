"""Pick the guardrail policy on a per-request basis.

The chat router (and the 3-column eval ablation) calls these instead of
the concrete ``filters.check_input`` / ``llamaguard.check_input`` so the
choice of policy is a single value (``"off"`` / ``"regex"`` / ``"llamaguard"``)
threaded from the request body or the eval matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from services.api.guardrails import llamaguard
from services.api.guardrails.filters import (
    GuardrailResult,
    check_input as regex_check_input,
    check_output as regex_check_output,
)

if TYPE_CHECKING:
    from sdk import LLMWrapper

    from services.api.config import Settings


GuardrailMode = Literal["off", "regex", "llamaguard"]


async def dispatch_input_check(
    mode: GuardrailMode,
    *,
    text: str,
    wrapper: "LLMWrapper | None" = None,
    settings: "Settings | None" = None,
) -> GuardrailResult:
    """Run the chosen input-side check. ``"off"`` always allows."""
    if mode == "off":
        return GuardrailResult.ok()
    if mode == "regex":
        return regex_check_input(text)
    if mode == "llamaguard":
        if wrapper is None or settings is None:
            raise ValueError("llamaguard mode requires wrapper + settings")
        return await llamaguard.check_input(wrapper, settings, text)
    raise ValueError(f"unknown guardrail mode: {mode!r}")


async def dispatch_output_check(
    mode: GuardrailMode,
    *,
    user_text: str,
    assistant_text: str,
    wrapper: "LLMWrapper | None" = None,
    settings: "Settings | None" = None,
) -> GuardrailResult:
    """Run the chosen output-side check. ``"off"`` always allows."""
    if mode == "off":
        return GuardrailResult.ok()
    if mode == "regex":
        # The regex layer doesn't need the user message for context.
        return regex_check_output(assistant_text)
    if mode == "llamaguard":
        if wrapper is None or settings is None:
            raise ValueError("llamaguard mode requires wrapper + settings")
        return await llamaguard.check_output(
            wrapper, settings, user_text, assistant_text
        )
    raise ValueError(f"unknown guardrail mode: {mode!r}")


__all__ = ["GuardrailMode", "dispatch_input_check", "dispatch_output_check"]
