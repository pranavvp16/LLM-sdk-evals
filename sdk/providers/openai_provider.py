"""
OpenAI Chat Completions provider.

One provider implementation covers the entire OpenAI-compatible ecosystem:
  OpenAI, Groq, Together AI, DeepSeek, Ollama, vLLM, LM Studio,
  HuggingFace Inference Endpoints, OpenRouter, etc.

Each just needs a different base_url + api_key.

Handles:
  - Streaming SSE via openai SDK
  - Tool calls (partial JSON accumulation)
  - Reasoning content (reasoning_content / reasoning fields for DeepSeek-R1, etc.)
  - Usage including prompt_tokens_details for cached tokens

Mirrors pi-ai's packages/ai/src/providers/openai-completions.ts
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator, Optional

from ..types import (
    AssistantMessage,
    ModelDef,
    StreamEvent,
    StreamEventDone,
    StreamEventError,
    StreamEventStart,
    StreamEventTextDelta,
    StreamEventThinkingDelta,
    StreamEventToolCallDelta,
    StreamEventToolCallEnd,
    StreamEventToolCallStart,
    TextContent,
    ToolCall,
    Usage,
    Context,
)
from ..normalize import to_openai_messages, to_openai_tools


# Provider-level compat flags (mirrors pi-ai's OpenAICompletionsCompat)
# Providers can set these to work around quirks.
PROVIDER_COMPAT: dict[str, dict] = {
    "ollama":       {"max_tokens_field": "max_tokens", "supports_usage_in_streaming": False},
    "groq":         {"supports_usage_in_streaming": True},
    "deepseek":     {"reasoning_field": "reasoning_content"},
    "huggingface":  {"max_tokens_field": "max_tokens", "supports_usage_in_streaming": False},
    "together":     {"reasoning_field": "reasoning"},
}

PROVIDER_BASE_URLS: dict[str, str] = {
    "openai":       "https://api.openai.com/v1",
    "groq":         "https://api.groq.com/openai/v1",
    "deepseek":     "https://api.deepseek.com/v1",
    "together":     "https://api.together.xyz/v1",
    "ollama":       "http://localhost:11434/v1",
    "huggingface":  "https://api-inference.huggingface.co/v1",
}


async def stream_openai_completions(
    model: ModelDef,
    ctx: Context,
    api_key: str,
    base_url: Optional[str] = None,
) -> AsyncIterator[StreamEvent]:
    """
    Stream from any OpenAI-compatible Chat Completions endpoint.
    Yields canonical StreamEvent objects.
    """
    try:
        import openai as _openai
    except ImportError:
        raise RuntimeError("pip install openai")

    compat = PROVIDER_COMPAT.get(model.provider, {})
    resolved_base = base_url or PROVIDER_BASE_URLS.get(model.provider)

    client = _openai.AsyncOpenAI(
        api_key=api_key or "ollama",  # Ollama ignores the key
        **({"base_url": resolved_base} if resolved_base else {}),
    )

    messages = to_openai_messages(ctx)
    max_tokens_key = compat.get("max_tokens_field", "max_completion_tokens")

    params: dict = {
        "model": model.id,
        max_tokens_key: ctx.max_tokens or model.max_tokens,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ctx.temperature is not None:
        params["temperature"] = ctx.temperature
    if ctx.tools:
        params["tools"] = to_openai_tools(ctx.tools)
        params["tool_choice"] = "auto"

    # Remove stream_options for providers that don't support it
    if not compat.get("supports_usage_in_streaming", True):
        params.pop("stream_options", None)

    output = AssistantMessage(
        provider=model.provider,
        model=model.id,
        trace_id=str(uuid.uuid4()),
        timestamp=int(time.time() * 1000),
    )

    yield StreamEventStart(partial=output)

    started_at = time.monotonic()

    # tool call accumulation: index → {id, name, args_buffer}
    _tool_buffers: dict[int, dict] = {}
    _text_buffer = ""
    _reasoning_buffer = ""
    reasoning_field = compat.get("reasoning_field", "reasoning_content")

    try:
        stream = await client.chat.completions.create(**params)

        async for chunk in stream:
            # ── Usage (final chunk) ──────────────────────────────────────────
            if chunk.usage:
                u = chunk.usage
                details = getattr(u, "prompt_tokens_details", None)
                output.usage = Usage(
                    input_tokens=u.prompt_tokens or 0,
                    output_tokens=u.completion_tokens or 0,
                    cache_read_tokens=getattr(details, "cached_tokens", 0) if details else 0,
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # ── Stop reason ──────────────────────────────────────────────────
            if choice.finish_reason:
                output.stop_reason = choice.finish_reason

            # ── Reasoning / thinking (DeepSeek-R1, o1, etc.) ────────────────
            reasoning_text = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
                or getattr(delta, "reasoning_text", None)
            )
            if reasoning_text:
                _reasoning_buffer += reasoning_text
                yield StreamEventThinkingDelta(delta=reasoning_text)

            # ── Text delta ───────────────────────────────────────────────────
            if delta.content:
                _text_buffer += delta.content
                yield StreamEventTextDelta(delta=delta.content)

            # ── Tool calls ───────────────────────────────────────────────────
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index

                    if idx not in _tool_buffers:
                        # First chunk for this tool call
                        _tool_buffers[idx] = {
                            "id": tc_delta.id or f"call_{uuid.uuid4().hex[:8]}",
                            "name": (tc_delta.function.name or "") if tc_delta.function else "",
                            "args": "",
                        }
                        yield StreamEventToolCallStart(
                            tool_call_id=_tool_buffers[idx]["id"],
                            name=_tool_buffers[idx]["name"],
                        )
                    else:
                        # Update name if it arrives late (some providers stream it)
                        if tc_delta.function and tc_delta.function.name:
                            _tool_buffers[idx]["name"] += tc_delta.function.name

                    if tc_delta.function and tc_delta.function.arguments:
                        _tool_buffers[idx]["args"] += tc_delta.function.arguments
                        yield StreamEventToolCallDelta(
                            tool_call_id=_tool_buffers[idx]["id"],
                            arguments_delta=tc_delta.function.arguments,
                        )

        # ── Finalize tool calls ───────────────────────────────────────────────
        for buf in _tool_buffers.values():
            try:
                args = json.loads(buf["args"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": buf["args"]}

            tc = ToolCall(id=buf["id"], name=buf["name"], arguments=args)
            output.tool_calls.append(tc)
            yield StreamEventToolCallEnd(tool_call=tc)

        # ── Assemble final message ────────────────────────────────────────────
        if _text_buffer:
            output.content.append(TextContent(text=_text_buffer))
        if _reasoning_buffer:
            output.thinking = _reasoning_buffer

        output.latency_ms = (time.monotonic() - started_at) * 1000
        yield StreamEventDone(message=output)

    except Exception as exc:
        retryable = _is_retryable(exc)
        yield StreamEventError(error=str(exc), retryable=retryable)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("rate limit", "429", "timeout", "502", "503", "504", "overload"))
