"""Llama Guard 3 (1B) safety classifier — input + output filter.

Llama Guard takes a conversation and replies with a token-level safety
verdict whose first line is either ``safe`` or ``unsafe``. The Ollama
modelfile for ``llama-guard3:1b`` bundles the official chat template, so
we just send standard role/content messages — Ollama applies the wrap.

Input check: send the user message in a single-turn conversation; Llama
Guard evaluates that last turn.

Output check: send both the user message and the assistant reply; Llama
Guard evaluates the assistant's turn against the same policy.

Returns the same ``GuardrailResult`` shape as the deterministic regex
filters so the dispatch in the chat router is a one-line swap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sdk import AssistantMessage, Context, TextContent, UserMessage, get_model
from services.api.guardrails.filters import GuardrailResult

if TYPE_CHECKING:
    from sdk import LLMWrapper

    from services.api.config import Settings

logger = logging.getLogger(__name__)


# Categories Llama Guard 3 emits (S1-S14 in the model card). We don't act on
# them differently — any "unsafe" verdict blocks — but we surface the codes
# in the GuardrailResult.reason so the report / SSE event can attribute the
# refusal.
_CATEGORY_LABELS: dict[str, str] = {
    "S1": "violent crimes",
    "S2": "non-violent crimes",
    "S3": "sex-related crimes",
    "S4": "child sexual exploitation",
    "S5": "defamation",
    "S6": "specialized advice",
    "S7": "privacy",
    "S8": "intellectual property",
    "S9": "indiscriminate weapons",
    "S10": "hate",
    "S11": "suicide & self-harm",
    "S12": "sexual content",
    "S13": "elections",
    "S14": "code interpreter abuse",
}


def _parse_verdict(text: str) -> GuardrailResult:
    """Parse a Llama Guard response into a GuardrailResult.

    Expected shapes (per the Llama Guard 3 model card):
        "safe"
        "unsafe\\nS1"
        "unsafe\\nS1,S10"
    Anything else is treated as ``safe`` (permissive on malformed output —
    the deterministic regex layer is still in front when guardrails="regex").
    """
    if not text:
        return GuardrailResult.ok()
    first, _, rest = text.strip().partition("\n")
    if first.strip().lower() == "safe":
        return GuardrailResult.ok()
    if first.strip().lower() != "unsafe":
        logger.warning("llama-guard returned unparseable verdict: %r", text[:80])
        return GuardrailResult.ok()
    codes = [c.strip() for c in rest.replace("\n", ",").split(",") if c.strip()]
    if not codes:
        return GuardrailResult.block("llamaguard: unsafe (no category)")
    labels = ", ".join(
        f"{c} ({_CATEGORY_LABELS.get(c, 'unknown')})" for c in codes
    )
    return GuardrailResult.block(f"llamaguard: unsafe — {labels}")


async def _classify(
    wrapper: "LLMWrapper",
    settings: "Settings",
    messages: list,
) -> GuardrailResult:
    """Run Llama Guard against the supplied messages and parse the verdict."""
    try:
        model = get_model("ollama", settings.ollama_guard_model)
    except KeyError as e:
        logger.warning("llama-guard model not registered: %s", e)
        return GuardrailResult.ok()

    ctx = Context(
        system_prompt="",  # Llama Guard's chat template handles framing
        messages=messages,
        temperature=0.0,
        max_tokens=64,
        stream=False,
    )
    try:
        assistant: AssistantMessage = await wrapper.complete(
            model,
            ctx,
            session_id="guardrails",
            conversation_id="llamaguard",
        )
    except Exception as e:  # noqa: BLE001
        # Permissive on classifier failure — alternative is to block all
        # traffic when Ollama is briefly unreachable, which is worse than
        # falling back to the regex layer the dispatcher applies in front.
        logger.warning("llama-guard classifier call failed: %s", e)
        return GuardrailResult.ok()

    text = "".join(b.text for b in assistant.content if isinstance(b, TextContent))
    return _parse_verdict(text)


async def check_input(
    wrapper: "LLMWrapper", settings: "Settings", user_text: str
) -> GuardrailResult:
    """Classify the user message before the chat model sees it."""
    if not user_text:
        return GuardrailResult.block("empty input")
    return await _classify(
        wrapper, settings, [UserMessage(content=user_text)]
    )


async def check_output(
    wrapper: "LLMWrapper",
    settings: "Settings",
    user_text: str,
    assistant_text: str,
) -> GuardrailResult:
    """Classify the assistant turn in the context of the user message."""
    if not assistant_text:
        return GuardrailResult.ok()
    assistant_msg = AssistantMessage(content=[TextContent(text=assistant_text)])
    return await _classify(
        wrapper,
        settings,
        [UserMessage(content=user_text), assistant_msg],
    )


__all__ = ["check_input", "check_output"]
