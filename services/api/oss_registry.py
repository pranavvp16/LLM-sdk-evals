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
    # Default vLLM target for this deployment. supports_tools=True relies on
    # vLLM being launched with `--enable-auto-tool-choice --tool-call-parser hermes`.
    register_model(
        ModelDef(
            id="qwen2.5-1.5b-instruct",
            provider="vllm",
            api=ApiProtocol.OPENAI_COMPLETIONS,
            context_window=32_768,
            max_tokens=2_048,
            cost=ModelCost(),
            supports_tools=True,
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
    # Self-hosted OSS endpoint on the prod VM. The colon-form id must match
    # OLLAMA_MODEL exactly — Ollama uses that string both as the pull tag
    # and the OpenAI-compat /v1/chat/completions model name. Qwen2.5
    # instruct ships native tool calling via Ollama's OpenAI shim.
    register_model(
        ModelDef(
            id="qwen2.5:1.5b",
            provider="ollama",
            api=ApiProtocol.OPENAI_COMPLETIONS,
            context_window=32_768,
            max_tokens=2_048,
            cost=ModelCost(),
            supports_tools=True,
        )
    )
    # Llama Guard 3 1B — safety classifier (input + output) hosted on the
    # same Ollama endpoint. Not a chat model: `supports_tools=False`. Used
    # by services/api/guardrails/llamaguard.py and by the 3-column eval.
    register_model(
        ModelDef(
            id="llama-guard3:1b",
            provider="ollama",
            api=ApiProtocol.OPENAI_COMPLETIONS,
            context_window=8_192,
            max_tokens=64,
            cost=ModelCost(),
            supports_tools=False,
        )
    )
    # gemini-2.5-flash — the previously-registered gemini-2.0-flash has
    # 0 free-tier quota on the deployment key; 2.5-flash still serves.
    register_model(
        ModelDef(
            id="gemini-2.5-flash",
            provider="google",
            api=ApiProtocol.GOOGLE_GENERATIVE,
            context_window=1_048_576,
            max_tokens=8_192,
            cost=ModelCost(input=0.30, output=2.50),
            supports_vision=True,
            supports_tools=True,
        )
    )
    _REGISTERED = True
