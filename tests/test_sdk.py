"""
Unit tests for the LLM wrapper SDK.
Run with: pytest tests/test_sdk.py -v
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Registry tests ─────────────────────────────────────────────────────────────

def test_get_model_anthropic():
    from sdk.registry import get_model
    m = get_model("anthropic", "claude-sonnet-4-20250514")
    assert m.provider == "anthropic"
    assert m.api.value == "anthropic-messages"
    assert m.context_window == 200_000
    assert m.cost.input == 3.0


def test_get_model_openai():
    from sdk.registry import get_model
    m = get_model("openai", "gpt-4o")
    assert m.api.value == "openai-completions"
    assert m.supports_vision is True


def test_get_model_vllm():
    from sdk.registry import get_model
    m = get_model("vllm", "qwen2.5-0.5b-instruct")
    assert m.provider == "vllm"
    assert m.api.value == "openai-completions"
    assert m.supports_tools is False


def test_get_model_opencode_zen():
    from sdk.registry import get_model
    m = get_model("opencode", "kimi-k2.5")
    assert m.provider == "opencode"
    assert m.api.value == "openai-completions"


def test_get_model_opencode_go():
    from sdk.registry import get_model
    m = get_model("opencode-go", "glm-5")
    assert m.provider == "opencode-go"
    assert m.api.value == "openai-completions"
    assert m.supports_tools is True


def test_get_model_not_found():
    from sdk.registry import get_model
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("anthropic", "claude-fake-9")


def test_list_models_by_provider():
    from sdk.registry import list_models
    anthropic_models = list_models("anthropic")
    assert all(m.provider == "anthropic" for m in anthropic_models)
    assert len(anthropic_models) >= 2


def test_register_custom_model():
    from sdk.registry import register_model, get_model
    from sdk.types import ModelDef, ApiProtocol, ModelCost
    custom = ModelDef(
        id="my-custom-model",
        provider="my-provider",
        api=ApiProtocol.OPENAI_COMPLETIONS,
        cost=ModelCost(input=1.0, output=2.0),
    )
    register_model(custom)
    assert get_model("my-provider", "my-custom-model").id == "my-custom-model"


# ── Normalization tests ────────────────────────────────────────────────────────

def test_to_openai_messages_simple():
    from sdk.normalize import to_openai_messages
    from sdk.types import Context, UserMessage, AssistantMessage, TextContent

    ctx = Context(
        system_prompt="You are helpful.",
        messages=[
            UserMessage(content="Hello"),
            AssistantMessage(content=[TextContent(text="Hi there!")]),
            UserMessage(content="What's 2+2?"),
        ],
    )
    msgs = to_openai_messages(ctx)
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assert msgs[1] == {"role": "user", "content": "Hello"}
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "Hi there!"
    assert msgs[3]["role"] == "user"


def test_to_openai_messages_tool_calls():
    from sdk.normalize import to_openai_messages
    from sdk.types import Context, AssistantMessage, ToolCall, ToolResultMessage, ToolResult, UserMessage

    ctx = Context(
        system_prompt="",
        messages=[
            UserMessage(content="Search for cats"),
            AssistantMessage(
                content=[],
                tool_calls=[ToolCall(id="call_abc", name="search", arguments={"q": "cats"})]
            ),
            ToolResultMessage(results=[
                ToolResult(tool_call_id="call_abc", name="search", content="Found 100 cat pics")
            ]),
        ],
    )
    msgs = to_openai_messages(ctx)
    # assistant with tool_calls
    asst = next(m for m in msgs if m["role"] == "assistant")
    assert "tool_calls" in asst
    assert asst["tool_calls"][0]["function"]["name"] == "search"
    # tool result
    tool = next(m for m in msgs if m["role"] == "tool")
    assert tool["tool_call_id"] == "call_abc"
    assert tool["content"] == "Found 100 cat pics"


def test_to_anthropic_messages_system_separate():
    from sdk.normalize import to_anthropic_messages
    from sdk.types import Context, UserMessage

    ctx = Context(
        system_prompt="Be concise.",
        messages=[UserMessage(content="Hi")],
    )
    system, messages = to_anthropic_messages(ctx)
    assert system == "Be concise."
    assert messages[0]["role"] == "user"
    # system NOT in messages list
    assert not any(m.get("role") == "system" for m in messages)


def test_to_anthropic_messages_tool_use_blocks():
    from sdk.normalize import to_anthropic_messages
    from sdk.types import Context, AssistantMessage, ToolCall, ToolResultMessage, ToolResult, UserMessage

    ctx = Context(
        system_prompt="",
        messages=[
            UserMessage(content="Do something"),
            AssistantMessage(
                tool_calls=[ToolCall(id="tu_1", name="bash", arguments={"cmd": "ls"})]
            ),
            ToolResultMessage(results=[
                ToolResult(tool_call_id="tu_1", name="bash", content="file1.py\nfile2.py")
            ]),
        ],
    )
    _, messages = to_anthropic_messages(ctx)

    asst = messages[1]
    assert asst["role"] == "assistant"
    tool_use = next(b for b in asst["content"] if b["type"] == "tool_use")
    assert tool_use["id"] == "tu_1"
    assert tool_use["name"] == "bash"

    # Tool result → user message with tool_result block
    user_result = messages[2]
    assert user_result["role"] == "user"
    tr = next(b for b in user_result["content"] if b["type"] == "tool_result")
    assert tr["tool_use_id"] == "tu_1"


def test_to_google_messages():
    from sdk.normalize import to_google_messages
    from sdk.types import Context, UserMessage, AssistantMessage, TextContent

    ctx = Context(
        system_prompt="You are Gemini.",
        messages=[
            UserMessage(content="Tell me a joke"),
            AssistantMessage(content=[TextContent(text="Why did the chicken...")]),
        ],
    )
    system, contents = to_google_messages(ctx)
    assert system == "You are Gemini."
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"  # not "assistant"
    assert contents[1]["parts"][0]["text"] == "Why did the chicken..."


# ── PII redaction tests ────────────────────────────────────────────────────────

def test_pii_email():
    from sdk.pii import redact
    assert redact("Email me at user@example.com please") == "Email me at [EMAIL] please"


def test_pii_phone():
    from sdk.pii import redact
    result = redact("Call me at 123-456-7890 ok?")
    assert "[PHONE]" in result


def test_pii_ssn():
    from sdk.pii import redact
    assert "[SSN]" in redact("My SSN is 123-45-6789")


def test_pii_credit_card():
    from sdk.pii import redact
    assert "[CC]" in redact("My card is 4111111111111111")


def test_pii_no_false_positives():
    from sdk.pii import redact
    clean = "The answer is 42 and pi is 3.14159"
    assert redact(clean) == clean


# ── Usage cost calculation ────────────────────────────────────────────────────

def test_usage_cost():
    from sdk.types import Usage
    from sdk.registry import get_model
    model = get_model("anthropic", "claude-sonnet-4-20250514")
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = usage.cost_usd(model)
    # $3 input + $15 output = $18
    assert abs(cost - 18.0) < 0.001


# ── Wrapper routing test (mocked providers) ───────────────────────────────────

@pytest.mark.asyncio
async def test_wrapper_routes_to_anthropic():
    from sdk.wrapper import LLMWrapper
    from sdk.registry import get_model
    from sdk.types import Context, UserMessage, StreamEventDone, AssistantMessage, TextContent, Usage

    # Build a fake stream that yields start + text_delta + done
    from sdk.types import StreamEventStart, StreamEventTextDelta

    async def fake_stream(*args, **kwargs):
        msg = AssistantMessage(provider="anthropic", model="claude-sonnet-4-20250514")
        msg.content = [TextContent(text="Hello from mock!")]
        msg.usage = Usage(input_tokens=10, output_tokens=5)
        yield StreamEventStart(partial=msg)
        yield StreamEventTextDelta(delta="Hello ")
        yield StreamEventTextDelta(delta="from mock!")
        msg.stop_reason = "stop"
        yield StreamEventDone(message=msg)

    logs = []
    wrapper = LLMWrapper(
        api_keys={"anthropic": "sk-test"},
        on_log=lambda log: logs.append(log),
    )

    model = get_model("anthropic", "claude-sonnet-4-20250514")
    ctx = Context(system_prompt="", messages=[UserMessage(content="Hi")])

    with patch("sdk.providers.anthropic_provider.stream_anthropic", side_effect=fake_stream):
        events = []
        async for event in wrapper.stream(model, ctx, session_id="s1", conversation_id="c1"):
            events.append(event)

    # Verify stream events
    assert any(isinstance(e, StreamEventTextDelta) for e in events)
    assert any(isinstance(e, StreamEventDone) for e in events)

    # Give asyncio a tick to process the fire-and-forget log task
    await asyncio.sleep(0.05)

    assert len(logs) == 1
    log = logs[0]
    assert log.provider == "anthropic"
    assert log.session_id == "s1"
    assert log.conversation_id == "c1"
    assert log.status == "success"
    assert log.input_tokens == 10
    assert log.output_tokens == 5


@pytest.mark.asyncio
async def test_wrapper_logs_error():
    from sdk.wrapper import LLMWrapper
    from sdk.registry import get_model
    from sdk.types import Context, UserMessage, StreamEventError

    async def failing_stream(*args, **kwargs):
        yield StreamEventError(error="Rate limit exceeded", retryable=True)

    logs = []
    wrapper = LLMWrapper(
        api_keys={"openai": "sk-test"},
        on_log=lambda log: logs.append(log),
    )
    model = get_model("openai", "gpt-4o")
    ctx = Context(system_prompt="", messages=[UserMessage(content="Hi")])

    with patch("sdk.providers.openai_provider.stream_openai_completions", side_effect=failing_stream):
        async for event in wrapper.stream(model, ctx):
            pass

    await asyncio.sleep(0.05)
    assert logs[0].status == "error"
    assert logs[0].error_message == "Rate limit exceeded"
