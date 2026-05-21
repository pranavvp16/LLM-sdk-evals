"""Drive the multi-hop tool loop for a single agent eval prompt.

Ports the loop in ``services/api/routers/chat.py`` minus the Postgres and SSE
plumbing — the eval runner has no DB and no streaming consumer, it just
captures the full ``Trajectory`` (every tool call, every tool result, the
final assistant text, cumulative tokens / cost / latency) for the judge.

Designed to be import-clean: only depends on ``services/api/tools`` (stdlib
imports) and ``sdk``. Does not pull FastAPI / asyncpg / Redis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from sdk import (
    AssistantMessage,
    Context,
    LLMWrapper,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    get_model,
)
from services.api.tools.builtins import register_builtins
from services.api.tools.registry import ToolRegistry, ToolUnknownError

logger = logging.getLogger(__name__)


MAX_TOOL_HOPS = 5
PER_HOP_TIMEOUT_S = 60.0


@dataclass
class ToolStep:
    hop: int
    call_id: str
    name: str
    arguments: dict[str, Any]
    result: Any
    is_error: bool
    error: str | None
    latency_ms: float


@dataclass
class Trajectory:
    prompt_id: str
    provider: str
    model: str
    user_prompt: str
    tool_calls: list[ToolStep] = field(default_factory=list)
    final_text: str = ""
    thinking: str = ""               # concatenated reasoning across hops
    hops_used: int = 0
    hit_max_hops: bool = False
    status: str = "success"          # success | model_error | max_hops
    error: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tool_calls"] = [asdict(s) for s in self.tool_calls]
        return d


def build_default_registry() -> ToolRegistry:
    """Build a ToolRegistry pre-loaded with the five mock tools."""
    registry = ToolRegistry()
    register_builtins(registry)
    return registry


async def _execute_one_call(
    tools: ToolRegistry, hop: int, call_id: str, name: str, args: dict
) -> ToolStep:
    """Run a single tool call, wrapping every failure mode as an error step."""
    t0 = time.perf_counter()
    # glm-5 / OpenAI-compatible providers stuff malformed JSON args under {"_raw": "..."}.
    if isinstance(args, dict) and set(args.keys()) == {"_raw"}:
        return ToolStep(
            hop=hop,
            call_id=call_id,
            name=name,
            arguments=args,
            result={"error": "malformed JSON arguments from model"},
            is_error=True,
            error="malformed_json",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    try:
        value = await tools.call(name, args)
        return ToolStep(
            hop=hop,
            call_id=call_id,
            name=name,
            arguments=args,
            result=value,
            is_error=False,
            error=None,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    except ToolUnknownError:
        return ToolStep(
            hop=hop,
            call_id=call_id,
            name=name,
            arguments=args,
            result={"error": f"unknown tool: {name}"},
            is_error=True,
            error="unknown_tool",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    except TypeError as e:
        return ToolStep(
            hop=hop,
            call_id=call_id,
            name=name,
            arguments=args,
            result={"error": str(e)},
            is_error=True,
            error="bad_args",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("tool %s failed", name)
        return ToolStep(
            hop=hop,
            call_id=call_id,
            name=name,
            arguments=args,
            result={"error": str(e)},
            is_error=True,
            error="handler_exception",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


async def _consume_stream(
    wrapper: LLMWrapper, model, ctx: Context, session_id: str, conversation_id: str
) -> tuple[str, AssistantMessage | None, str | None]:
    """Drain one provider stream. Returns (hop_text, final_msg, error_str)."""
    hop_text = ""
    final_msg: AssistantMessage | None = None
    error_str: str | None = None
    async for event in wrapper.stream(
        model, ctx, session_id=session_id, conversation_id=conversation_id
    ):
        if event.type == "text_delta":
            hop_text += event.delta
        elif event.type == "done":
            final_msg = event.message
        elif event.type == "error":
            error_str = event.error
    return hop_text, final_msg, error_str


async def run_trajectory(
    wrapper: LLMWrapper,
    provider: str,
    model_name: str,
    tools: ToolRegistry,
    user_prompt: str,
    prompt_id: str,
    system_prompt: str,
    max_hops: int = MAX_TOOL_HOPS,
    per_hop_timeout_s: float = PER_HOP_TIMEOUT_S,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    # Default off because Anthropic thinking blocks carry a per-block
    # `signature` that must be replayed on the next hop, and the SDK's
    # AssistantMessage shape does not currently capture/roundtrip it.
    # Multi-hop frontier calls with thinking=True fail at hop 2. Callers
    # can opt back in explicitly once signature roundtripping lands.
    thinking: bool | int | None = None,
) -> Trajectory:
    """Run one agent prompt end-to-end and return the captured trajectory.

    Identical loop semantics to ``services/api/routers/chat.py:220-324`` minus
    Postgres + SSE. Tool calls within a single hop run in parallel; hops are
    sequential. Hop-level timeout via ``asyncio.wait_for`` (3.10-safe).
    """
    model = get_model(provider, model_name)
    if not model.supports_tools:
        return Trajectory(
            prompt_id=prompt_id,
            provider=provider,
            model=model_name,
            user_prompt=user_prompt,
            status="model_error",
            error=f"{provider}/{model_name} has supports_tools=False",
        )

    ctx = Context(
        system_prompt=system_prompt,
        messages=[UserMessage(content=user_prompt)],
        tools=tools.schemas(),
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
    )

    traj = Trajectory(
        prompt_id=prompt_id,
        provider=provider,
        model=model_name,
        user_prompt=user_prompt,
    )

    t0 = time.perf_counter()
    accumulated_text_parts: list[str] = []
    accumulated_thinking_parts: list[str] = []
    try:
        for hop in range(max_hops):
            try:
                hop_text, final_msg, stream_error = await asyncio.wait_for(
                    _consume_stream(
                        wrapper,
                        model,
                        ctx,
                        session_id="eval-agent",
                        conversation_id=f"{provider}-{model_name}-{prompt_id}",
                    ),
                    timeout=per_hop_timeout_s,
                )
            except asyncio.TimeoutError:
                traj.status = "model_error"
                traj.error = f"hop {hop} timed out after {per_hop_timeout_s}s"
                traj.hops_used = hop
                return _finalize(traj, t0, accumulated_text_parts, accumulated_thinking_parts)

            if stream_error:
                traj.status = "model_error"
                traj.error = stream_error
                traj.hops_used = hop + 1
                accumulated_text_parts.append(hop_text)
                if final_msg and final_msg.thinking:
                    accumulated_thinking_parts.append(final_msg.thinking)
                return _finalize(traj, t0, accumulated_text_parts, accumulated_thinking_parts)

            if hop_text:
                accumulated_text_parts.append(hop_text)

            if final_msg is None:
                # Stream ended without a `done` event — treat as terminal.
                traj.hops_used = hop + 1
                return _finalize(traj, t0, accumulated_text_parts, accumulated_thinking_parts)

            if final_msg.thinking:
                accumulated_thinking_parts.append(final_msg.thinking)

            usage = final_msg.usage
            traj.input_tokens += usage.input_tokens
            traj.output_tokens += usage.output_tokens
            traj.cost_usd += usage.cost_usd(model)

            if not final_msg.tool_calls:
                traj.hops_used = hop + 1
                return _finalize(traj, t0, accumulated_text_parts, accumulated_thinking_parts)

            steps = await asyncio.gather(
                *[
                    _execute_one_call(tools, hop, tc.id, tc.name, tc.arguments)
                    for tc in final_msg.tool_calls
                ]
            )
            traj.tool_calls.extend(steps)

            tool_results = [
                ToolResult(
                    tool_call_id=step.call_id,
                    name=step.name,
                    content=json.dumps(step.result, default=str),
                    is_error=step.is_error,
                )
                for step in steps
            ]
            ctx.messages.append(final_msg)
            ctx.messages.append(ToolResultMessage(results=tool_results))
        else:
            traj.hit_max_hops = True
            traj.status = "max_hops"
            traj.hops_used = max_hops
    except Exception as e:  # noqa: BLE001
        logger.exception("run_trajectory crashed for %s", prompt_id)
        traj.status = "model_error"
        traj.error = str(e)

    return _finalize(traj, t0, accumulated_text_parts, accumulated_thinking_parts)


def _finalize(
    traj: Trajectory, t0: float, text_parts: list[str], thinking_parts: list[str]
) -> Trajectory:
    traj.final_text = "\n\n".join(p for p in text_parts if p.strip())
    traj.thinking = "\n\n---\n\n".join(p for p in thinking_parts if p and p.strip())
    traj.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return traj
