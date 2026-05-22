"""Pluggable input / output / tool-arg guardrails.

Two policies live here:

* ``filters`` — deterministic regex/length checks (the default).
* ``llamaguard`` — Llama Guard 3 1B served on the same Ollama endpoint as
  the chat model. Used when a chat request sets ``guardrails="llamaguard"``
  or by the 3-column eval ablation.

``dispatch`` picks between them per-request. The L2 safety axis in
``eval/judge.py`` measures whether the chosen layer actually catches the
things it's meant to.
"""

from __future__ import annotations

from services.api.guardrails.dispatch import (
    GuardrailMode,
    dispatch_input_check,
    dispatch_output_check,
)
from services.api.guardrails.filters import (
    GuardrailResult,
    check_input,
    check_output,
    validate_tool_args,
)

__all__ = [
    "GuardrailMode",
    "GuardrailResult",
    "check_input",
    "check_output",
    "dispatch_input_check",
    "dispatch_output_check",
    "validate_tool_args",
]
