"""
Model registry.

Inspired by pi-ai's ModelRegistry + models.generated.ts:
- Every model has a provider, api protocol, context window, cost, and capability flags.
- get_model("anthropic", "claude-sonnet-4-6") returns the full ModelDef.
- Providers can be extended at runtime (for proxies, custom endpoints, etc.)
"""

from __future__ import annotations

from typing import Optional
from .types import ApiProtocol, ModelCost, ModelDef

# ── Built-in model catalog ────────────────────────────────────────────────────

_MODELS: dict[tuple[str, str], ModelDef] = {}


def _register(*models: ModelDef) -> None:
    for m in models:
        _MODELS[(m.provider, m.id)] = m


# ── Anthropic ─────────────────────────────────────────────────────────────────
_register(
    ModelDef(
        id="claude-sonnet-4-6",
        provider="anthropic",
        api=ApiProtocol.ANTHROPIC_MESSAGES,
        context_window=200_000,
        max_tokens=16_000,
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.30, cache_write=3.75),
        supports_vision=True,
        supports_tools=True,
        supports_reasoning=False,
    ),
    ModelDef(
        id="claude-opus-4-7",
        provider="anthropic",
        api=ApiProtocol.ANTHROPIC_MESSAGES,
        context_window=200_000,
        max_tokens=16_000,
        cost=ModelCost(input=15.0, output=75.0, cache_read=1.50, cache_write=18.75),
        supports_vision=True,
        supports_tools=True,
        supports_reasoning=True,
    ),
    ModelDef(
        id="claude-haiku-4-5-20251001",
        provider="anthropic",
        api=ApiProtocol.ANTHROPIC_MESSAGES,
        context_window=200_000,
        max_tokens=8_000,
        cost=ModelCost(input=0.80, output=4.0, cache_read=0.08, cache_write=1.0),
        supports_vision=True,
        supports_tools=True,
        supports_reasoning=False,
    ),
)


# Legacy aliases — `tests/test_sdk.py` is frozen and still references these IDs.
# Map them to the current canonical Sonnet/Opus models so lookups keep working.
_MODELS[("anthropic", "claude-sonnet-4-20250514")] = _MODELS[("anthropic", "claude-sonnet-4-6")]
_MODELS[("anthropic", "claude-opus-4-20250514")] = _MODELS[("anthropic", "claude-opus-4-7")]


# ── OpenAI ────────────────────────────────────────────────────────────────────
_register(
    ModelDef(
        id="gpt-4o",
        provider="openai",
        api=ApiProtocol.OPENAI_COMPLETIONS,
        context_window=128_000,
        max_tokens=16_384,
        cost=ModelCost(input=2.50, output=10.0),
        supports_vision=True,
        supports_tools=True,
    ),
    ModelDef(
        id="gpt-4o-mini",
        provider="openai",
        api=ApiProtocol.OPENAI_COMPLETIONS,
        context_window=128_000,
        max_tokens=16_384,
        cost=ModelCost(input=0.15, output=0.60),
        supports_vision=True,
        supports_tools=True,
    ),
    ModelDef(
        id="gpt-4.1",
        provider="openai",
        api=ApiProtocol.OPENAI_COMPLETIONS,
        context_window=1_000_000,
        max_tokens=32_768,
        cost=ModelCost(input=2.0, output=8.0),
        supports_vision=True,
        supports_tools=True,
    ),
    ModelDef(
        id="o3",
        provider="openai",
        api=ApiProtocol.OPENAI_RESPONSES,
        context_window=200_000,
        max_tokens=100_000,
        cost=ModelCost(input=10.0, output=40.0),
        supports_vision=True,
        supports_tools=True,
        supports_reasoning=True,
    ),
)

# ── Google ────────────────────────────────────────────────────────────────────
_register(
    ModelDef(
        id="gemini-2.0-flash",
        provider="google",
        api=ApiProtocol.GOOGLE_GENERATIVE,
        context_window=1_048_576,
        max_tokens=8_192,
        cost=ModelCost(input=0.10, output=0.40),
        supports_vision=True,
        supports_tools=True,
    ),
    ModelDef(
        id="gemini-2.5-pro",
        provider="google",
        api=ApiProtocol.GOOGLE_GENERATIVE,
        context_window=2_000_000,
        max_tokens=65_536,
        cost=ModelCost(input=1.25, output=10.0),
        supports_vision=True,
        supports_tools=True,
        supports_reasoning=True,
    ),
)

# ── OpenAI-compatible providers (Groq, Together, Ollama, vLLM, DeepSeek, etc.)
# These all speak openai-completions but have different base URLs + keys.
_register(
    ModelDef(
        id="llama-3.3-70b-versatile",
        provider="groq",
        api=ApiProtocol.OPENAI_COMPLETIONS,
        context_window=128_000,
        max_tokens=32_768,
        cost=ModelCost(input=0.59, output=0.79),
        supports_tools=True,
    ),
    ModelDef(
        id="deepseek-chat",
        provider="deepseek",
        api=ApiProtocol.OPENAI_COMPLETIONS,
        context_window=64_000,
        max_tokens=8_192,
        cost=ModelCost(input=0.27, output=1.10),
        supports_tools=True,
    ),
    ModelDef(
        id="deepseek-reasoner",
        provider="deepseek",
        api=ApiProtocol.OPENAI_COMPLETIONS,
        context_window=64_000,
        max_tokens=8_192,
        cost=ModelCost(input=0.55, output=2.19),
        supports_tools=True,
        supports_reasoning=True,
    ),
    # Ollama local — zero cost, configurable base URL
    ModelDef(
        id="llama3.1:8b",
        provider="ollama",
        api=ApiProtocol.OPENAI_COMPLETIONS,
        context_window=128_000,
        max_tokens=4_096,
        cost=ModelCost(),  # free / self-hosted
        supports_tools=True,
    ),
    ModelDef(
        id="qwen2.5-0.5b-instruct",
        provider="huggingface",
        api=ApiProtocol.OPENAI_COMPLETIONS,
        context_window=32_768,
        max_tokens=2_048,
        cost=ModelCost(),  # free tier
        supports_tools=False,
    ),
    # vLLM — self-hosted OpenAI-compatible (base URL from env VLLM_BASE_URL)
    ModelDef(
        id="qwen2.5-0.5b-instruct",
        provider="vllm",
        api=ApiProtocol.OPENAI_COMPLETIONS,
        context_window=32_768,
        max_tokens=2_048,
        cost=ModelCost(),
        supports_tools=False,
    ),
    # OpenCode Zen (pay-as-you-go) — https://opencode.ai/zen/v1 (OPENCODE_API_KEY)
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
    ),
    # OpenCode Go (subscription) — https://opencode.ai/zen/go/v1 (same OPENCODE_API_KEY)
    ModelDef(
        id="glm-5",
        provider="opencode-go",
        api=ApiProtocol.OPENAI_COMPLETIONS,
        context_window=202_752,
        max_tokens=32_768,
        cost=ModelCost(input=1.0, output=3.2, cache_read=0.2),
        supports_tools=True,
        supports_reasoning=True,
    ),
)


# ── Public API ────────────────────────────────────────────────────────────────

def get_model(provider: str, model_id: str) -> ModelDef:
    key = (provider, model_id)
    if key not in _MODELS:
        raise ValueError(
            f"Unknown model '{model_id}' for provider '{provider}'. "
            f"Call register_model() to add it, or check list_models()."
        )
    return _MODELS[key]


def list_models(provider: Optional[str] = None) -> list[ModelDef]:
    if provider:
        return [m for (p, _), m in _MODELS.items() if p == provider]
    return list(_MODELS.values())


def register_model(model: ModelDef) -> None:
    """Runtime registration — for proxies, custom endpoints, tests."""
    _MODELS[(model.provider, model.id)] = model
