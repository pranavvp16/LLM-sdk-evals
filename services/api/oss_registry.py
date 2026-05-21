"""Runtime OSS model registration (avoids editing frozen sdk/registry.py)."""

from __future__ import annotations

from sdk.registry import register_model
from sdk.types import ApiProtocol, ModelCost, ModelDef

_REGISTERED = False


def register_oss_models() -> None:
    """Idempotent catalog entries for vLLM and OpenCode providers."""
    global _REGISTERED
    if _REGISTERED:
        return

    register_model(
        ModelDef(
            id="qwen2.5-0.5b-instruct",
            provider="vllm",
            api=ApiProtocol.OPENAI_COMPLETIONS,
            context_window=32_768,
            max_tokens=2_048,
            cost=ModelCost(),
            supports_tools=False,
        )
    )
    register_model(
        ModelDef(
            id="kimi-k2.5",
            provider="opencode",
            api=ApiProtocol.OPENAI_COMPLETIONS,
            context_window=262_144,
            max_tokens=65_536,
            cost=ModelCost(input=0.6, output=3.0, cache_read=0.1),
            supports_vision=True,
            supports_tools=True,
            supports_reasoning=True,
        )
    )
    register_model(
        ModelDef(
            id="glm-5",
            provider="opencode-go",
            api=ApiProtocol.OPENAI_COMPLETIONS,
            context_window=202_752,
            max_tokens=32_768,
            cost=ModelCost(input=1.0, output=3.2, cache_read=0.2),
            supports_tools=True,
            supports_reasoning=True,
        )
    )
    register_model(
        ModelDef(
            id="deepseek-v4-flash",
            provider="opencode-go",
            api=ApiProtocol.OPENAI_COMPLETIONS,
            context_window=128_000,
            max_tokens=8_192,
            cost=ModelCost(input=0.27, output=1.1, cache_read=0.07),
            supports_tools=True,
            supports_reasoning=True,
        )
    )
    _REGISTERED = True
