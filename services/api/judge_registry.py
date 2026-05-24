"""Runtime registration of LLM-as-judge models (avoids editing frozen sdk/registry.py).

The eval's 3-judge panel needs `openai/gpt-5` available in the SDK registry.
`anthropic/claude-sonnet-4-6` is already built-in; `opencode-go/deepseek-v4-flash`
is registered by `oss_registry.register_oss_models()`. This module fills the gap.
"""

from __future__ import annotations

from sdk.registry import register_model
from sdk.types import ApiProtocol, ModelCost, ModelDef

_REGISTERED = False


def register_judge_models() -> None:
    """Idempotent registration of models used exclusively by the eval judge panel."""
    global _REGISTERED
    if _REGISTERED:
        return

    # OpenAI's flagship model as of the assistant's knowledge cutoff (Jan 2026).
    # Pricing per OpenAI public pricing at launch. Adjust if OpenAI revises.
    register_model(
        ModelDef(
            id="gpt-5",
            provider="openai",
            api=ApiProtocol.OPENAI_COMPLETIONS,
            context_window=256_000,
            max_tokens=16_384,
            cost=ModelCost(input=1.25, output=10.0, cache_read=0.125),
            supports_vision=True,
            supports_tools=True,
            supports_reasoning=True,
        )
    )
    register_model(
        ModelDef(
            id="gpt-5-mini",
            provider="openai",
            api=ApiProtocol.OPENAI_COMPLETIONS,
            context_window=256_000,
            max_tokens=16_384,
            cost=ModelCost(input=0.25, output=2.0, cache_read=0.025),
            supports_vision=True,
            supports_tools=True,
            supports_reasoning=True,
        )
    )

    _REGISTERED = True
