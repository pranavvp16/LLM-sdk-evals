"""
Anthropic Messages API provider.

Translates Anthropic's native SSE events into our canonical StreamEvent sequence.
Handles: text, thinking blocks, tool_use, usage extraction.

Mirrors pi-ai's packages/ai/src/providers/anthropic.ts
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
from ..normalize import to_anthropic_messages, to_anthropic_tools


async def stream_anthropic(
    model: ModelDef,
    ctx: Context,
    api_key: str,
    base_url: Optional[str] = None,
) -> AsyncIterator[StreamEvent]:
    """
    Stream from Anthropic Messages API.
    Yields canonical StreamEvent objects.
    """
    try:
        import anthropic as _anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic")

    client = _anthropic.AsyncAnthropic(
        api_key=api_key,
        **({"base_url": base_url} if base_url else {}),
    )

    system, messages = to_anthropic_messages(ctx)

    params: dict = {
        "model": model.id,
        "max_tokens": ctx.max_tokens or model.max_tokens,
        "messages": messages,
    }
    if system:
        params["system"] = system
    if ctx.temperature is not None:
        params["temperature"] = ctx.temperature
    if ctx.tools:
        params["tools"] = to_anthropic_tools(ctx.tools)
    # Extended thinking (opt-in). The installed anthropic SDK predates the
    # top-level `thinking` kwarg, so we pass it via extra_body — the body
    # field lands on the wire identically.
    if ctx.thinking:
        budget = ctx.thinking if isinstance(ctx.thinking, int) and ctx.thinking > 1 else 2048
        # Anthropic requires max_tokens > budget_tokens for thinking mode.
        if params["max_tokens"] <= budget:
            params["max_tokens"] = budget + 1024
        # temperature must be 1.0 (or omitted) when thinking is enabled.
        params.pop("temperature", None)
        params["extra_body"] = {
            "thinking": {"type": "enabled", "budget_tokens": budget}
        }

    # Initialize the assembled output message
    output = AssistantMessage(
        provider=model.provider,
        model=model.id,
        trace_id=str(uuid.uuid4()),
        timestamp=int(time.time() * 1000),
    )

    yield StreamEventStart(partial=output)

    started_at = time.monotonic()

    # Tracks in-progress tool calls: index → {id, name, raw_args}
    _tool_buffers: dict[int, dict] = {}
    _text_buffer = ""
    _thinking_buffer = ""

    try:
        async with client.messages.stream(**params) as stream:
            async for event in stream:
                etype = event.type

                # ── content_block_start ──────────────────────────────────────
                if etype == "content_block_start":
                    block = event.content_block
                    idx = event.index

                    if block.type == "thinking":
                        pass  # handled in content_block_delta

                    elif block.type == "text":
                        pass  # handled in content_block_delta

                    elif block.type == "tool_use":
                        _tool_buffers[idx] = {
                            "id": block.id,
                            "name": block.name,
                            "raw_args": "",
                        }
                        yield StreamEventToolCallStart(
                            tool_call_id=block.id,
                            name=block.name,
                        )

                # ── content_block_delta ──────────────────────────────────────
                elif etype == "content_block_delta":
                    delta = event.delta

                    if delta.type == "thinking_delta":
                        _thinking_buffer += delta.thinking
                        yield StreamEventThinkingDelta(delta=delta.thinking)

                    elif delta.type == "text_delta":
                        _text_buffer += delta.text
                        yield StreamEventTextDelta(delta=delta.text)

                    elif delta.type == "input_json_delta":
                        idx = event.index
                        if idx in _tool_buffers:
                            _tool_buffers[idx]["raw_args"] += delta.partial_json
                            yield StreamEventToolCallDelta(
                                tool_call_id=_tool_buffers[idx]["id"],
                                arguments_delta=delta.partial_json,
                            )

                # ── content_block_stop ───────────────────────────────────────
                elif etype == "content_block_stop":
                    idx = event.index
                    if idx in _tool_buffers:
                        buf = _tool_buffers.pop(idx)
                        try:
                            args = json.loads(buf["raw_args"] or "{}")
                        except json.JSONDecodeError:
                            args = {"_raw": buf["raw_args"]}

                        tc = ToolCall(id=buf["id"], name=buf["name"], arguments=args)
                        output.tool_calls.append(tc)
                        yield StreamEventToolCallEnd(tool_call=tc)

                # ── message_delta (stop_reason + usage) ─────────────────────
                elif etype == "message_delta":
                    if hasattr(event, "delta") and hasattr(event.delta, "stop_reason"):
                        output.stop_reason = event.delta.stop_reason or "stop"
                    if hasattr(event, "usage"):
                        u = event.usage
                        output.usage = Usage(
                            input_tokens=getattr(u, "input_tokens", 0),
                            output_tokens=getattr(u, "output_tokens", 0),
                            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0),
                            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0),
                        )

                # ── message_start (initial usage) ────────────────────────────
                elif etype == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        u = event.message.usage
                        output.usage = Usage(
                            input_tokens=getattr(u, "input_tokens", 0),
                            output_tokens=getattr(u, "output_tokens", 0),
                        )

        # Assemble final content
        if _text_buffer:
            output.content.append(TextContent(text=_text_buffer))
        if _thinking_buffer:
            output.thinking = _thinking_buffer

        output.latency_ms = (time.monotonic() - started_at) * 1000
        yield StreamEventDone(message=output)

    except Exception as exc:
        retryable = _is_retryable(exc)
        yield StreamEventError(error=str(exc), retryable=retryable)


def _is_retryable(exc: Exception) -> bool:
    """Rate limits and 5xx errors are worth retrying."""
    msg = str(exc).lower()
    return any(k in msg for k in ("rate limit", "529", "overloaded", "timeout", "502", "503", "504"))
