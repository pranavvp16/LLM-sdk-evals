"""
Message normalization.

Each provider speaks a different wire format.  This module converts our
canonical Context + Message types into the exact shapes each SDK expects.

Mirrors pi-ai's transformMessages() per provider:
  - openai-completions  → openai_sdk messages list
  - anthropic-messages  → anthropic_sdk messages + system string
  - google-generative-ai → google_sdk contents list
"""

from __future__ import annotations

from typing import Any
from .types import (
    AssistantMessage,
    ContentBlock,
    Context,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)


# ── OpenAI Completions normalization ─────────────────────────────────────────

def to_openai_messages(ctx: Context) -> list[dict]:
    """
    Produces the `messages` list for openai.chat.completions.create().
    Handles: text, images, tool_calls, tool_results.

    Key quirks handled (mirroring pi-ai's openai-completions.ts):
    - Tool call IDs must start with 'call_' for some proxies
    - Tool results become role='tool' messages
    - Images become content array items with type='image_url'
    """
    out: list[dict] = []

    if ctx.system_prompt:
        out.append({"role": "system", "content": ctx.system_prompt})

    for msg in ctx.messages:
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                out.append({"role": "user", "content": msg.content})
            else:
                out.append({
                    "role": "user",
                    "content": [_block_to_openai(b) for b in msg.content],
                })

        elif isinstance(msg, AssistantMessage):
            m: dict[str, Any] = {"role": "assistant", "content": ""}

            # Collect text content
            text_parts = [b.text for b in msg.content if isinstance(b, TextContent)]
            m["content"] = "".join(text_parts)

            # Reasoning content roundtrip — DeepSeek (and any future OpenAI-
            # compatible reasoning model) rejects multi-hop tool requests
            # when the prior assistant turn's reasoning_content is dropped.
            # Vanilla OpenAI / vLLM / Ollama ignore unknown fields, so this
            # is safe to include unconditionally when msg.thinking is set.
            if msg.thinking:
                m["reasoning_content"] = msg.thinking

            # Tool calls
            if msg.tool_calls:
                m["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": _safe_json(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            out.append(m)

        elif isinstance(msg, ToolResultMessage):
            # Each result becomes its own role='tool' message
            for r in msg.results:
                out.append({
                    "role": "tool",
                    "tool_call_id": r.tool_call_id,
                    "content": r.content,
                })

    return out


def _block_to_openai(block: ContentBlock) -> dict:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageContent):
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
        }
    raise ValueError(f"Unknown content block type: {type(block)}")


# ── Anthropic Messages normalization ─────────────────────────────────────────

def to_anthropic_messages(ctx: Context) -> tuple[str, list[dict]]:
    """
    Returns (system_prompt, messages) for anthropic.messages.create().

    Key quirks handled (mirroring pi-ai's anthropic.ts):
    - system is a top-level param, NOT a message role
    - tool_use content blocks (not role='tool')
    - tool_result content blocks inside a user message
    - Thinking blocks are passed through unchanged
    """
    messages: list[dict] = []

    for msg in ctx.messages:
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                messages.append({"role": "user", "content": msg.content})
            else:
                messages.append({
                    "role": "user",
                    "content": [_block_to_anthropic(b) for b in msg.content],
                })

        elif isinstance(msg, AssistantMessage):
            content_blocks: list[dict] = []

            # Thinking block (extended reasoning). Anthropic requires the
            # per-block `signature` from the original response to be replayed
            # on subsequent hops; without it the API rejects the request with
            # "Extended thinking blocks must include a signature when sent in
            # the messages list." When no signature was captured (e.g. the
            # message originated from a different provider), we still include
            # the thinking text so callers can see it, but Anthropic will
            # reject — that's the same failure mode as before this branch.
            if msg.thinking:
                thinking_block: dict = {"type": "thinking", "thinking": msg.thinking}
                if msg.thinking_signature:
                    thinking_block["signature"] = msg.thinking_signature
                content_blocks.append(thinking_block)

            # Text blocks
            for block in msg.content:
                if isinstance(block, TextContent) and block.text:
                    content_blocks.append({"type": "text", "text": block.text})

            # Tool use blocks
            for tc in msg.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })

            messages.append({
                "role": "assistant",
                "content": content_blocks or "",
            })

        elif isinstance(msg, ToolResultMessage):
            # All results go into a single user message as tool_result blocks
            content_blocks = []
            for r in msg.results:
                content_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": r.tool_call_id,
                    "content": r.content,
                    **({"is_error": True} if r.is_error else {}),
                })
            messages.append({"role": "user", "content": content_blocks})

    return ctx.system_prompt, messages


def _block_to_anthropic(block: ContentBlock) -> dict:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageContent):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.mime_type,
                "data": block.data,
            },
        }
    raise ValueError(f"Unknown content block type: {type(block)}")


# ── Google Generative AI normalization ────────────────────────────────────────

def to_google_messages(ctx: Context) -> tuple[str, list[dict]]:
    """
    Returns (system_instruction, contents) for google.generativeai.

    Key quirks:
    - system_instruction is a top-level param
    - roles are 'user' and 'model' (not 'assistant')
    - Tool calls are 'functionCall' parts
    - Tool results are 'functionResponse' parts inside a 'user' turn
    """
    contents: list[dict] = []

    for msg in ctx.messages:
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            else:
                contents.append({
                    "role": "user",
                    "parts": [_block_to_google(b) for b in msg.content],
                })

        elif isinstance(msg, AssistantMessage):
            parts: list[dict] = []
            for block in msg.content:
                if isinstance(block, TextContent) and block.text:
                    parts.append({"text": block.text})
            for tc in msg.tool_calls:
                parts.append({
                    "functionCall": {"name": tc.name, "args": tc.arguments}
                })
            contents.append({"role": "model", "parts": parts})

        elif isinstance(msg, ToolResultMessage):
            parts = []
            for r in msg.results:
                parts.append({
                    "functionResponse": {
                        "name": r.name,
                        "response": {"content": r.content},
                    }
                })
            contents.append({"role": "user", "parts": parts})

    return ctx.system_prompt, contents


def _block_to_google(block: ContentBlock) -> dict:
    if isinstance(block, TextContent):
        return {"text": block.text}
    if isinstance(block, ImageContent):
        return {
            "inlineData": {"mimeType": block.mime_type, "data": block.data}
        }
    raise ValueError(f"Unknown content block type: {type(block)}")


# ── Tool schema normalization ─────────────────────────────────────────────────

def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Wrap JSON-schema tool defs in OpenAI's function wrapper."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Anthropic uses a flat list with input_schema."""
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


# Keys allowed in Google's Schema (OpenAPI subset used by function_declarations).
# JSON-Schema constraints like minimum/maximum/minLength/pattern are rejected
# with "Unknown field for Schema: <key>", so we strip them recursively.
_GOOGLE_SCHEMA_KEYS = frozenset({
    "type", "format", "description", "nullable", "enum",
    "properties", "required", "items", "default", "example",
    "anyOf", "title",
})


def _sanitize_google_schema(node: Any) -> Any:
    if isinstance(node, dict):
        cleaned: dict = {}
        for k, v in node.items():
            if k not in _GOOGLE_SCHEMA_KEYS:
                continue
            if k in ("properties",) and isinstance(v, dict):
                cleaned[k] = {pk: _sanitize_google_schema(pv) for pk, pv in v.items()}
            elif k in ("items", "default", "example"):
                cleaned[k] = _sanitize_google_schema(v) if isinstance(v, (dict, list)) else v
            elif k == "anyOf" and isinstance(v, list):
                cleaned[k] = [_sanitize_google_schema(x) for x in v]
            else:
                cleaned[k] = v
        return cleaned
    if isinstance(node, list):
        return [_sanitize_google_schema(x) for x in node]
    return node


def to_google_tools(tools: list[dict]) -> list[dict]:
    """Google uses function declarations."""
    return [
        {
            "function_declarations": [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": _sanitize_google_schema(
                        t.get("parameters", {"type": "object", "properties": {}})
                    ),
                }
            ]
        }
        for t in tools
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_json(obj: Any) -> str:
    import json
    try:
        return json.dumps(obj)
    except Exception:
        return str(obj)
