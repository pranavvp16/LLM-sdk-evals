"""
LLM Wrapper SDK — public API surface.

    from sdk import LLMWrapper, Context, UserMessage, get_model

    wrapper = LLMWrapper(api_keys={"anthropic": "sk-ant-..."})
    model = get_model("anthropic", "claude-sonnet-4-6")
    ctx = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Hello!")],
    )
    async for event in wrapper.stream(model, ctx, session_id="s1", conversation_id="c1"):
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
"""

from .types import (
    ApiProtocol,
    AssistantMessage,
    ContentBlock,
    Context,
    ImageContent,
    InferenceLog,
    Message,
    ModelCost,
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
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .registry import get_model, list_models, register_model
from .wrapper import LLMWrapper
from .pii import redact

__all__ = [
    # types
    "ApiProtocol",
    "AssistantMessage",
    "ContentBlock",
    "Context",
    "ImageContent",
    "InferenceLog",
    "Message",
    "ModelCost",
    "ModelDef",
    "StreamEvent",
    "StreamEventDone",
    "StreamEventError",
    "StreamEventStart",
    "StreamEventTextDelta",
    "StreamEventThinkingDelta",
    "StreamEventToolCallDelta",
    "StreamEventToolCallEnd",
    "StreamEventToolCallStart",
    "TextContent",
    "ToolCall",
    "ToolResult",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    # registry
    "get_model",
    "list_models",
    "register_model",
    # wrapper
    "LLMWrapper",
    # utils
    "redact",
]
