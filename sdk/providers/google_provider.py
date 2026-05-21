"""
Google Generative AI provider (Gemini).

Translates Google's native streaming response into canonical StreamEvent sequence.
Handles: text, function_call, usage_metadata, thinking (Gemini 2.5+).

Mirrors pi-ai's packages/ai/src/providers/google.ts
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
    """
    Stream from Google Generative AI (Gemini) API.
    Yields canonical StreamEvent objects.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("pip install google-generativeai")

    genai.configure(api_key=api_key)

    system, contents = to_google_messages(ctx)

    generation_config: dict = {
        "max_output_tokens": ctx.max_tokens or model.max_tokens,
    }
    if ctx.temperature is not None:
        generation_config["temperature"] = ctx.temperature

    client = genai.GenerativeModel(
        model_name=model.id,
        system_instruction=system or None,
        generation_config=generation_config,
        **({"tools": to_google_tools(ctx.tools)} if ctx.tools else {}),
    )

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

    try:
        response = client.generate_content(contents, stream=True)

        for chunk in response:
            # Google's streaming is synchronous in the SDK (wraps a generator)
            # For async, we iterate and yield — good enough for FastAPI with run_in_executor
            for part in (chunk.parts or []):
                # Thinking parts (Gemini 2.5 extended thinking)
                if getattr(part, "thought", False):
                    text = getattr(part, "text", "")
                    if text:
                        _thinking_buffer += text
                        yield StreamEventThinkingDelta(delta=text)

                # Text parts
                elif hasattr(part, "text") and part.text:
                    _text_buffer += part.text
                    yield StreamEventTextDelta(delta=part.text)

                # Function call parts
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tc_id = f"call_{uuid.uuid4().hex[:8]}"
                    args = dict(fc.args) if fc.args else {}

                    yield StreamEventToolCallStart(tool_call_id=tc_id, name=fc.name)
                    tc = ToolCall(id=tc_id, name=fc.name, arguments=args)
                    output.tool_calls.append(tc)
                    yield StreamEventToolCallEnd(tool_call=tc)

        # Usage metadata (available after full response)
        try:
            meta = response.usage_metadata
            if meta:
                output.usage = Usage(
                    input_tokens=getattr(meta, "prompt_token_count", 0),
                    output_tokens=getattr(meta, "candidates_token_count", 0),
                )
        except Exception:
            pass

        if _text_buffer:
            output.content.append(TextContent(text=_text_buffer))
        if _thinking_buffer:
            output.thinking = _thinking_buffer

        output.latency_ms = (time.monotonic() - started_at) * 1000

        # Stop reason
        try:
            output.stop_reason = str(response.candidates[0].finish_reason).lower()
        except Exception:
            output.stop_reason = "stop"

        yield StreamEventDone(message=output)

    except Exception as exc:
        yield StreamEventError(error=str(exc), retryable=_is_retryable(exc))


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("rate limit", "quota", "429", "503", "timeout"))
