"""
Canonical types for the LLM wrapper.

Inspired by pi-ai's design: every provider (openai-completions, anthropic-messages,
google-generative-ai, etc.) converts its native wire format INTO these types.
Consumers never touch provider-specific shapes.

Stream event sequence (mirrors pi's AssistantMessageEventStream):
    start → text_delta* → [tool_call_start → tool_call_delta* → tool_call_end]* → done | error
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Literal, Optional, Union


# ── Protocol discriminator ────────────────────────────────────────────────────

class ApiProtocol(str, Enum):
    """Which wire protocol the provider speaks.
    Maps 1-to-1 with pi-ai's api field values.
    """
    OPENAI_COMPLETIONS  = "openai-completions"   # OpenAI, Groq, Together, Ollama, vLLM, etc.
    ANTHROPIC_MESSAGES  = "anthropic-messages"   # Anthropic Claude (native SDK)
    GOOGLE_GENERATIVE   = "google-generative-ai" # Gemini via google-generativeai SDK
    OPENAI_RESPONSES    = "openai-responses"     # OpenAI Responses API (o1/o3)


# ── Message types (canonical, provider-agnostic) ──────────────────────────────

@dataclass
class TextContent:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageContent:
    type: Literal["image"] = "image"
    data: str = ""          # base64
    mime_type: str = "image/jpeg"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]  # fully parsed JSON


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


ContentBlock = Union[TextContent, ImageContent]


@dataclass
class UserMessage:
    role: Literal["user"] = "user"
    content: Union[str, list[ContentBlock]] = ""


@dataclass
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: Optional[str] = None          # reasoning trace, normalized across providers
    # ── filled in after stream completes ─────────────────────────────────────
    provider: str = ""
    model: str = ""
    usage: "Usage" = field(default_factory=lambda: Usage())
    stop_reason: str = "stop"
    latency_ms: float = 0.0
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ToolResultMessage:
    role: Literal["tool_result"] = "tool_result"
    results: list[ToolResult] = field(default_factory=list)


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


# ── Model definition ──────────────────────────────────────────────────────────

@dataclass
class ModelCost:
    """Cost in USD per million tokens."""
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass
class ModelDef:
    """
    A registered model.  Mirrors pi-ai's ModelDefinition interface:
        { id, provider, api, contextWindow, maxTokens, cost, capabilities }
    """
    id: str
    provider: str
    api: ApiProtocol
    context_window: int = 128_000
    max_tokens: int = 4_096
    cost: ModelCost = field(default_factory=ModelCost)
    supports_vision: bool = False
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_reasoning: bool = False


# ── Token usage + cost tracking ───────────────────────────────────────────────

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, model: ModelDef) -> float:
        c = model.cost
        return (
            self.input_tokens      * c.input       / 1_000_000
            + self.output_tokens   * c.output      / 1_000_000
            + self.cache_read_tokens  * c.cache_read  / 1_000_000
            + self.cache_write_tokens * c.cache_write / 1_000_000
        )


# ── Streaming event types ─────────────────────────────────────────────────────
# Every provider emits these; consumers are provider-agnostic.

@dataclass
class StreamEventStart:
    """Emitted once at the start of a stream, before any tokens."""
    type: Literal["start"] = "start"
    partial: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class StreamEventTextDelta:
    """A new chunk of text from the model."""
    type: Literal["text_delta"] = "text_delta"
    delta: str = ""


@dataclass
class StreamEventThinkingDelta:
    """Reasoning/thinking token chunk (Anthropic extended thinking, o1 reasoning_content, etc.)"""
    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str = ""


@dataclass
class StreamEventToolCallStart:
    """Model started emitting a tool call (id + name known, args still streaming)."""
    type: Literal["tool_call_start"] = "tool_call_start"
    tool_call_id: str = ""
    name: str = ""


@dataclass
class StreamEventToolCallDelta:
    """Partial JSON arguments for an in-progress tool call."""
    type: Literal["tool_call_delta"] = "tool_call_delta"
    tool_call_id: str = ""
    arguments_delta: str = ""


@dataclass
class StreamEventToolCallEnd:
    """Tool call fully received — arguments are now parseable."""
    type: Literal["tool_call_end"] = "tool_call_end"
    tool_call: ToolCall = field(default_factory=lambda: ToolCall("", "", {}))


@dataclass
class StreamEventDone:
    """Stream finished successfully.  Contains the complete assembled message."""
    type: Literal["done"] = "done"
    message: AssistantMessage = field(default_factory=AssistantMessage)


@dataclass
class StreamEventError:
    """Stream terminated with an error."""
    type: Literal["error"] = "error"
    error: str = ""
    retryable: bool = False


StreamEvent = Union[
    StreamEventStart,
    StreamEventTextDelta,
    StreamEventThinkingDelta,
    StreamEventToolCallStart,
    StreamEventToolCallDelta,
    StreamEventToolCallEnd,
    StreamEventDone,
    StreamEventError,
]

# A provider stream is just an async generator of StreamEvent
AssistantMessageEventStream = AsyncIterator[StreamEvent]


# ── Context passed to every provider ─────────────────────────────────────────

@dataclass
class Context:
    """
    Everything a provider needs to make a call.
    Mirrors pi-ai's Context interface:
        { systemPrompt, messages, tools?, temperature?, maxTokens? }
    """
    system_prompt: str
    messages: list[Message]
    tools: list[dict] = field(default_factory=list)  # JSON-schema tool defs
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = True


# ── Inference log (captured by the wrapper, sent to ingestion) ────────────────

@dataclass
class InferenceLog:
    trace_id: str
    session_id: str
    conversation_id: str
    provider: str
    model: str
    api_protocol: str
    # timing
    started_at: int        # unix ms
    ended_at: int
    latency_ms: float
    # tokens
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    total_tokens: int
    cost_usd: float
    # content previews (100-char truncations, PII redacted before this)
    input_preview: str
    output_preview: str
    # status
    status: Literal["success", "error", "cancelled"]
    error_message: Optional[str]
    stop_reason: str
    # request metadata
    stream: bool
    temperature: Optional[float]
    max_tokens: Optional[int]
