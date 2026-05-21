"""Chat and conversation routes.

Owns:
    POST /chat/stream                — start or resume a conversation, SSE response
    GET  /conversations              — list conversations (newest first)
    GET  /conversations/{id}         — full message history
    DELETE /conversations/{id}       — soft-cancel (sets status='cancelled')
    POST /conversations/{id}/resume  — sets status='active'

The route handler never instantiates `LLMWrapper` — it pulls the shared one from
`app.state` via `deps.llm_wrapper`. `InferenceLog` is fired off to the ingestion
endpoint by the wrapper itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sdk import (
    AssistantMessage,
    Context,
    LLMWrapper,
    TextContent,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    get_model,
)

from services.api.db import postgres as pg
from services.api.deps import llm_wrapper, pg_pool, tools_registry
from services.api.tools.registry import ToolRegistry, ToolUnknownError

MAX_TOOL_HOPS = 5

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ── request / response models ─────────────────────────────────────────────


class ChatStreamRequest(BaseModel):
    conversation_id: Optional[UUID] = None
    message: str = Field(min_length=1, max_length=32_000)
    provider: str
    model: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # True (or int budget) → enable reasoning; False → ask provider to disable
    # if supported; None → provider default (Anthropic off, OSS reasoning on).
    thinking: Optional[bool | int] = None


class ConversationOut(BaseModel):
    id: UUID
    session_id: UUID
    title: str
    provider: str
    model: str
    status: str
    created_at: str
    updated_at: str


# ── helpers ───────────────────────────────────────────────────────────────


def _resolve_session_id(request: Request, response: Response) -> UUID:
    header = request.headers.get("X-Session-ID")
    if header:
        try:
            return UUID(header)
        except ValueError:
            pass
    new_id = uuid4()
    response.headers["X-Session-ID"] = str(new_id)
    return new_id


def _sse(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")


def _rebuild_history(history: list[dict]) -> list:
    """Reconstruct UserMessage / AssistantMessage / ToolResultMessage objects
    from message rows in insertion order.

    Anthropic's wire format requires the tool_result blocks that satisfy a
    given assistant turn to live in one user-role message that follows it
    immediately. We mirror that here by buffering consecutive tool_result
    rows and flushing them on the next non-tool row.
    """
    out: list = []
    pending_results: list[ToolResult] = []
    # Populated from the most recent assistant row's tool_calls JSONB so
    # replayed ToolResult objects carry the same name as live execution.
    tool_names: dict[str, str] = {}

    def flush_results() -> None:
        nonlocal pending_results
        if pending_results:
            out.append(ToolResultMessage(results=pending_results))
            pending_results = []

    for m in history:
        role = m["role"]
        if role == "tool_result":
            tc_id = m.get("tool_call_id") or ""
            pending_results.append(
                ToolResult(
                    tool_call_id=tc_id,
                    name=tool_names.get(tc_id, ""),
                    content=m["content"],
                    is_error=bool(m.get("is_error")),
                )
            )
            continue

        flush_results()

        if role == "user":
            tool_names = {}
            out.append(UserMessage(content=m["content"]))
        elif role == "assistant":
            content_blocks: list = []
            if m["content"]:
                content_blocks.append(TextContent(text=m["content"]))
            raw_calls = m.get("tool_calls") or []
            tool_names = {tc["id"]: tc["name"] for tc in raw_calls}
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in raw_calls
            ]
            out.append(AssistantMessage(content=content_blocks, tool_calls=tool_calls))

    # Orphaned tool results (persisted before the model's follow-up hop
    # finished) must not be replayed: flushing them here would place a
    # ToolResultMessage immediately before the new UserMessage, producing
    # consecutive user-role messages that Anthropic rejects. Strip the
    # unanswered tool_calls from the trailing assistant turn instead.
    if pending_results:
        pending_results = []
        if out and isinstance(out[-1], AssistantMessage) and out[-1].tool_calls:
            last = out[-1]
            out[-1] = AssistantMessage(content=last.content, tool_calls=[])

    return out


# ── routes ────────────────────────────────────────────────────────────────


@router.post("/chat/stream")
async def chat_stream(
    body: ChatStreamRequest,
    request: Request,
    response: Response,
    pool=Depends(pg_pool),
    wrapper: LLMWrapper = Depends(llm_wrapper),
    tools: ToolRegistry = Depends(tools_registry),
):
    session_id = _resolve_session_id(request, response)
    await pg.upsert_session(
        pool,
        session_id,
        user_agent=request.headers.get("user-agent"),
        ip=request.client.host if request.client else None,
    )

    # Create the conversation lazily on the first turn.
    if body.conversation_id is None:
        conv = await pg.create_conversation(
            pool,
            session_id=session_id,
            provider=body.provider,
            model=body.model,
            title=body.message[:80],
        )
        conv_id: UUID = conv["id"]
        history: list[dict] = []
    else:
        conv = await pg.get_conversation(pool, body.conversation_id)
        if conv is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
        conv_id = conv["id"]
        history = await pg.list_messages(pool, conv_id)

    await pg.add_message(pool, conv_id, role="user", content=body.message)

    try:
        model = get_model(body.provider, body.model)
    except KeyError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e

    # Rebuild full history including tool round-trips so the model can refer
    # back to anything it called previously (event ids, weather it fetched,
    # reminders it set, etc.). Anthropic requires tool_result blocks to
    # immediately follow the assistant message that triggered them — we group
    # consecutive tool_result rows after each assistant turn to satisfy that.
    messages: list = _rebuild_history(history)
    messages.append(UserMessage(content=body.message))

    ctx = Context(
        system_prompt=body.system_prompt or "You are a helpful assistant.",
        messages=messages,
        tools=tools.schemas(),
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        thinking=body.thinking,
    )

    async def event_generator() -> AsyncIterator[bytes]:
        cumulative_usage = {"input": 0, "output": 0, "cost_usd": 0.0}
        try:
            for hop in range(MAX_TOOL_HOPS):
                hop_text = ""
                final_msg: Optional[AssistantMessage] = None

                async for event in wrapper.stream(
                    model,
                    ctx,
                    session_id=str(session_id),
                    conversation_id=str(conv_id),
                ):
                    if event.type == "text_delta":
                        hop_text += event.delta
                        yield _sse({"type": "text_delta", "delta": event.delta})
                    elif event.type == "thinking_delta":
                        yield _sse({"type": "thinking_delta", "delta": event.delta})
                    elif event.type == "tool_call_end":
                        tc = event.tool_call
                        yield _sse({
                            "type": "tool_call",
                            "id": tc.id,
                            "name": tc.name,
                            "args": tc.arguments,
                        })
                    elif event.type == "done":
                        final_msg = event.message
                    elif event.type == "error":
                        yield _sse({
                            "type": "error",
                            "error": event.error,
                            "retryable": event.retryable,
                        })
                        return

                if final_msg is None:
                    # Stream ended without a `done` event — treat as terminal.
                    break

                usage = final_msg.usage
                cumulative_usage["input"] += usage.input_tokens
                cumulative_usage["output"] += usage.output_tokens
                cumulative_usage["cost_usd"] += usage.cost_usd(model)

                # Persist the assistant turn — both text content (if any) and
                # the tool calls it issued. Even when hop_text is empty we
                # still need a row so history replay can pair tool_calls with
                # the tool_result rows that follow.
                tool_calls_payload: list[dict] | None = (
                    [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in final_msg.tool_calls
                    ]
                    if final_msg.tool_calls
                    else None
                )
                if hop_text or tool_calls_payload:
                    await pg.add_message(
                        pool,
                        conv_id,
                        role="assistant",
                        content=hop_text,
                        token_count=usage.output_tokens,
                        tool_calls=tool_calls_payload,
                    )

                # No tool calls → conversation turn is done.
                if not final_msg.tool_calls:
                    yield _sse({"type": "done", "usage": cumulative_usage})
                    return

                # Execute every requested tool call in parallel.
                results = await asyncio.gather(
                    *[_safe_call(tools, tc.name, tc.arguments) for tc in final_msg.tool_calls],
                )

                tool_results: list[ToolResult] = []
                for tc, result in zip(final_msg.tool_calls, results):
                    payload = json.dumps(result["value"])
                    yield _sse({
                        "type": "tool_result",
                        "id": tc.id,
                        "name": tc.name,
                        "result": result["value"],
                        "is_error": result["is_error"],
                    })
                    await pg.add_message(
                        pool,
                        conv_id,
                        role="tool_result",
                        content=payload,
                        tool_call_id=tc.id,
                        is_error=result["is_error"],
                    )
                    tool_results.append(ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=payload,
                        is_error=result["is_error"],
                    ))

                ctx.messages.append(final_msg)
                ctx.messages.append(ToolResultMessage(results=tool_results))
                # loop to next hop so the model can react to results
            else:
                # Exhausted MAX_TOOL_HOPS without the model wrapping up.
                yield _sse({
                    "type": "error",
                    "error": f"hit MAX_TOOL_HOPS={MAX_TOOL_HOPS} without a final answer",
                    "retryable": False,
                })

        except Exception as e:  # noqa: BLE001
            logger.exception("stream failed")
            yield _sse({"type": "error", "error": str(e), "retryable": False})
        finally:
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Conversation-ID": str(conv_id),
            "X-Session-ID": str(session_id),
        },
    )


async def _safe_call(tools: ToolRegistry, name: str, args: dict) -> dict:
    """Run a tool by name; never raise — wrap errors as a tool result."""
    try:
        value = await tools.call(name, args)
        return {"value": value, "is_error": False}
    except ToolUnknownError:
        return {"value": {"error": f"unknown tool: {name}"}, "is_error": True}
    except Exception as e:  # noqa: BLE001
        logger.exception("tool %s failed", name)
        return {"value": {"error": str(e)}, "is_error": True}


@router.get("/conversations")
async def list_conversations(
    request: Request,
    response: Response,
    limit: int = 50,
    offset: int = 0,
    pool=Depends(pg_pool),
) -> list[dict]:
    session_id = _resolve_session_id(request, response)
    return await pg.list_conversations(pool, session_id=session_id, limit=limit, offset=offset)


@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: UUID, pool=Depends(pg_pool)) -> dict:
    conv = await pg.get_conversation(pool, conv_id)
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    conv["messages"] = await pg.list_messages(pool, conv_id)
    return conv


@router.delete("/conversations/{conv_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_conversation(conv_id: UUID, pool=Depends(pg_pool)) -> Response:
    await pg.set_conversation_status(pool, conv_id, "cancelled")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/conversations/{conv_id}/resume", status_code=status.HTTP_204_NO_CONTENT)
async def resume_conversation(conv_id: UUID, pool=Depends(pg_pool)) -> Response:
    await pg.set_conversation_status(pool, conv_id, "active")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
