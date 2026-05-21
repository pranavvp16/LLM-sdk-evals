"""
LLMWrapper — the single public entry point for all LLM calls.

Usage:
    wrapper = LLMWrapper(api_keys={"anthropic": "sk-...", "openai": "sk-..."})

    # Streaming
    async for event in wrapper.stream(model, context, session_id="s1", conversation_id="c1"):
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
        elif event.type == "done":
            print()  # final message in event.message

    # Non-streaming (convenience)
    message = await wrapper.complete(model, context, session_id="s1", conversation_id="c1")

The wrapper:
1. Routes to the correct provider based on model.api
2. Captures an InferenceLog on every call (success or failure)
3. Fires the log to the ingestion endpoint asynchronously (fire-and-forget)
4. Applies PII redaction to previews before logging
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import AsyncIterator, Callable, Optional

from .types import (
    ApiProtocol,
    AssistantMessage,
    Context,
    InferenceLog,
    ModelDef,
    StreamEvent,
    StreamEventDone,
    StreamEventError,
    StreamEventTextDelta,
    TextContent,
    Usage,
)
from .registry import get_model
from .pii import redact


IngestionCallback = Callable[[InferenceLog], None]  # sync or async


class LLMWrapper:
    """
    Multi-provider LLM wrapper with built-in observability.

    Parameters
    ----------
    api_keys:
        Map of provider name → API key.
        e.g. {"anthropic": "sk-ant-...", "openai": "sk-...", "google": "AIza..."}
    base_urls:
        Optional overrides for provider base URLs (proxies, local endpoints).
        e.g. {"ollama": "http://localhost:11434/v1"}
    ingestion_url:
        If set, InferenceLogs are POSTed here after each call.
    on_log:
        Optional sync/async callback called with every InferenceLog.
        Use for testing or custom sinks.
    """

    def __init__(
        self,
        api_keys: dict[str, str] | None = None,
        base_urls: dict[str, str] | None = None,
        ingestion_url: str | None = None,
        on_log: IngestionCallback | None = None,
    ):
        self._api_keys = api_keys or {}
        self._base_urls = base_urls or {}
        self._ingestion_url = ingestion_url
        self._on_log = on_log

    # ── Public API ─────────────────────────────────────────────────────────────

    async def stream(
        self,
        model: ModelDef,
        ctx: Context,
        session_id: str = "",
        conversation_id: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream from any provider. Yields canonical StreamEvent objects.
        An InferenceLog is fired after the stream completes (done or error).
        """
        provider_stream = self._route(model, ctx)

        started_at_ms = int(time.time() * 1000)
        started_mono = time.monotonic()

        final_message: Optional[AssistantMessage] = None
        error_message: Optional[str] = None
        status = "success"

        async for event in provider_stream:
            yield event

            if isinstance(event, StreamEventDone):
                final_message = event.message
            elif isinstance(event, StreamEventError):
                error_message = event.error
                status = "error"

        # ── Build and dispatch InferenceLog ───────────────────────────────────
        ended_at_ms = int(time.time() * 1000)
        latency_ms = (time.monotonic() - started_mono) * 1000

        usage = final_message.usage if final_message else Usage()
        cost = usage.cost_usd(model)

        # Input preview = last user message, truncated + redacted
        input_preview = _last_user_content(ctx)
        output_preview = ""
        if final_message:
            texts = [b.text for b in final_message.content if isinstance(b, TextContent)]
            output_preview = "".join(texts)

        log = InferenceLog(
            trace_id=final_message.trace_id if final_message else str(uuid.uuid4()),
            session_id=session_id,
            conversation_id=conversation_id,
            provider=model.provider,
            model=model.id,
            api_protocol=model.api.value,
            started_at=started_at_ms,
            ended_at=ended_at_ms,
            latency_ms=round(latency_ms, 2),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=round(cost, 8),
            input_preview=redact(input_preview[:200]),
            output_preview=redact(output_preview[:200]),
            status=status,
            error_message=error_message,
            stop_reason=final_message.stop_reason if final_message else "error",
            stream=ctx.stream,
            temperature=ctx.temperature,
            max_tokens=ctx.max_tokens,
        )

        asyncio.create_task(self._dispatch_log(log))

    async def complete(
        self,
        model: ModelDef,
        ctx: Context,
        session_id: str = "",
        conversation_id: str = "",
    ) -> AssistantMessage:
        """
        Non-streaming convenience wrapper.
        Drains the stream and returns the final AssistantMessage.
        """
        ctx = Context(
            system_prompt=ctx.system_prompt,
            messages=ctx.messages,
            tools=ctx.tools,
            temperature=ctx.temperature,
            max_tokens=ctx.max_tokens,
            stream=False,
            thinking=ctx.thinking,
        )
        final: Optional[AssistantMessage] = None
        async for event in self.stream(model, ctx, session_id, conversation_id):
            if isinstance(event, StreamEventDone):
                final = event.message
            elif isinstance(event, StreamEventError):
                raise RuntimeError(event.error)

        if final is None:
            raise RuntimeError("Stream ended without a done event")
        return final

    # ── Routing ────────────────────────────────────────────────────────────────

    def _route(self, model: ModelDef, ctx: Context) -> AsyncIterator[StreamEvent]:
        api_key = self._api_keys.get(model.provider, "")
        base_url = self._base_urls.get(model.provider)

        if model.api == ApiProtocol.ANTHROPIC_MESSAGES:
            from .providers.anthropic_provider import stream_anthropic
            return stream_anthropic(model, ctx, api_key, base_url)

        elif model.api in (ApiProtocol.OPENAI_COMPLETIONS, ApiProtocol.OPENAI_RESPONSES):
            from .providers.openai_provider import stream_openai_completions
            return stream_openai_completions(model, ctx, api_key, base_url)

        elif model.api == ApiProtocol.GOOGLE_GENERATIVE:
            from .providers.google_provider import stream_google
            return stream_google(model, ctx, api_key, base_url)

        else:
            raise ValueError(f"Unsupported API protocol: {model.api}")

    # ── Log dispatch ──────────────────────────────────────────────────────────

    async def _dispatch_log(self, log: InferenceLog) -> None:
        """Fire-and-forget: send log to ingestion endpoint + callback."""
        tasks = []

        if self._on_log:
            if asyncio.iscoroutinefunction(self._on_log):
                tasks.append(self._on_log(log))
            else:
                self._on_log(log)

        if self._ingestion_url:
            tasks.append(_post_log(self._ingestion_url, log))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def _post_log(url: str, log: InferenceLog) -> None:
    """POST InferenceLog JSON to the ingestion endpoint. Never raises."""
    try:
        import aiohttp, dataclasses
        payload = dataclasses.asdict(log)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as r:
                pass  # best-effort
    except Exception:
        pass  # logging must never break the chat path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _last_user_content(ctx: Context) -> str:
    from .types import UserMessage
    for msg in reversed(ctx.messages):
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                return msg.content
            texts = [b.text for b in msg.content if hasattr(b, "text")]
            return " ".join(texts)
    return ""
