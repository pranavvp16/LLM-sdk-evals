"""
Google Generative AI provider (Gemini).

Translates Google's native streaming response into canonical StreamEvent sequence.
Handles: text, function_call, usage_metadata, thinking (Gemini 2.5+).

Uses the modern ``google-genai`` SDK (``from google import genai``). The
older ``google-generativeai`` Python package only exposed the OpenAPI 3.03
``parameters`` field for tool schemas, which rejects JSON-Schema vocabulary
like ``minimum`` / ``maximum`` / ``default``. The new SDK accepts
``parameters_json_schema`` so we can pass tool schemas through untouched.
"""

from __future__ import annotations

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
    StreamEventToolCallEnd,
    StreamEventToolCallStart,
    TextContent,
    ToolCall,
    Usage,
    Context,
)
from ..normalize import to_google_messages, to_google_tools


async def stream_google(
    model: ModelDef,
    ctx: Context,
    api_key: str,
    base_url: Optional[str] = None,
) -> AsyncIterator[StreamEvent]:
    """Stream from Gemini via the ``google-genai`` async client."""
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError:
        raise RuntimeError("pip install google-genai")

    client = genai.Client(api_key=api_key)

    system, contents = to_google_messages(ctx)

    config_kwargs: dict = {
        "max_output_tokens": ctx.max_tokens or model.max_tokens,
    }
    if system:
        config_kwargs["system_instruction"] = system
    if ctx.temperature is not None:
        config_kwargs["temperature"] = ctx.temperature
    if ctx.tools:
        config_kwargs["tools"] = [
            gtypes.Tool(**td) for td in to_google_tools(ctx.tools)
        ]
    # Gemini 2.5+ supports extended thinking. ctx.thinking controls it
    # (True = on with default budget, int = budget, False = disable).
    if model.supports_reasoning and ctx.thinking is not False:
        budget = ctx.thinking if type(ctx.thinking) is int else None
        thinking_cfg: dict = {"include_thoughts": True}
        if budget is not None:
            thinking_cfg["thinking_budget"] = budget
        config_kwargs["thinking_config"] = gtypes.ThinkingConfig(**thinking_cfg)

    config = gtypes.GenerateContentConfig(**config_kwargs)

    output = AssistantMessage(
        provider=model.provider,
        model=model.id,
        trace_id=str(uuid.uuid4()),
        timestamp=int(time.time() * 1000),
    )

    yield StreamEventStart(partial=output)

    started_at = time.monotonic()
    _text_buffer = ""
    _thinking_buffer = ""
    last_response = None

    try:
        stream = await client.aio.models.generate_content_stream(
            model=model.id,
            contents=contents,
            config=config,
        )
        async for chunk in stream:
            last_response = chunk
            candidates = chunk.candidates or []
            if not candidates:
                continue
            cand = candidates[0]
            parts = (cand.content.parts if cand.content else None) or []
            for part in parts:
                if getattr(part, "thought", False):
                    text = getattr(part, "text", "") or ""
                    if text:
                        _thinking_buffer += text
                        yield StreamEventThinkingDelta(delta=text)
                    continue

                text = getattr(part, "text", None)
                if text:
                    _text_buffer += text
                    yield StreamEventTextDelta(delta=text)
                    continue

                fc = getattr(part, "function_call", None)
                if fc and fc.name:
                    tc_id = f"call_{uuid.uuid4().hex[:8]}"
                    args = dict(fc.args) if fc.args else {}
                    yield StreamEventToolCallStart(tool_call_id=tc_id, name=fc.name)
                    tc = ToolCall(id=tc_id, name=fc.name, arguments=args)
                    output.tool_calls.append(tc)
                    yield StreamEventToolCallEnd(tool_call=tc)

        meta = getattr(last_response, "usage_metadata", None) if last_response else None
        if meta:
            output.usage = Usage(
                input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            )

        if _text_buffer:
            output.content.append(TextContent(text=_text_buffer))
        if _thinking_buffer:
            output.thinking = _thinking_buffer

        output.latency_ms = (time.monotonic() - started_at) * 1000

        try:
            finish = last_response.candidates[0].finish_reason if last_response else None
            output.stop_reason = str(finish).split(".")[-1].lower() if finish else "stop"
        except Exception:
            output.stop_reason = "stop"

        yield StreamEventDone(message=output)

    except Exception as exc:
        yield StreamEventError(error=str(exc), retryable=_is_retryable(exc))


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("rate limit", "quota", "429", "503", "timeout"))
