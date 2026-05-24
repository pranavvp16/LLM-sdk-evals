"""G-Eval leaf scorer.

The G-Eval pattern (Liu et al. 2023):
1. Generate explicit chain-of-thought evaluation steps from a high-level
   ``criteria`` string via one LLM call.
2. Send those steps + the rubric + the case to the judge, request an integer
   score in [0, 10] with a one-paragraph reason.
3. Normalize the score to [0, 5] for parity with the rest of the eval.

The eval-steps generation is cached per (judge_id, criteria) for the run —
callers pass a shared cache dict into ``score_leaf``. Cache miss costs one
extra LLM call per axis per judge per run.

Logprob-weighted normalization (the original paper's contribution) requires
the judge to expose logprobs. None of our three judges do that through
``LLMWrapper`` today (Anthropic does not support logprobs at all; OpenAI's
chat completions API supports them but our wrapper does not surface them).
We therefore use the raw integer; this is documented in the report's
"judge panel methodology" page so readers know the difference.
"""

from __future__ import annotations

import json
import logging
import re
import time

from sdk import Context, LLMWrapper, UserMessage
from sdk.types import AssistantMessage, ModelDef, TextContent

from eval.dag.nodes import GEvalLeaf

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_MAX_RETRIES = 3


_EVAL_STEPS_PROMPT = """\
Given the following evaluation criteria, write 3 to 5 concrete evaluation
steps a human grader would apply to score an LLM response from 0 to 10 on
this criterion. The steps should be ordered, atomic, and verifiable.

## Criterion
{criteria}

## Score key (anchors)
{rubric}

Respond ONLY with a JSON list of strings:
["step 1", "step 2", ...]
"""


_SCORE_PROMPT = """\
You are scoring an LLM response on one specific criterion.

## Criterion
{criteria}

## Score key (use these anchors)
{rubric}

## Evaluation steps to follow in order
{steps}

## Case
- User prompt: {prompt}
- Expected behavior: {expected}
- Assistant response: {response}

Apply the evaluation steps in order, then assign an integer score 0-10
strictly following the score key. Respond ONLY with JSON:
{{
  "score": <integer 0-10>,
  "reason": "<1-3 sentences citing which steps drove the score>"
}}
"""


def _extract_text(msg: AssistantMessage) -> str:
    return "".join(b.text for b in msg.content if isinstance(b, TextContent))


def _rubric_block(rubric: dict[int, str]) -> str:
    return "\n".join(f"  {k} = {v}" for k, v in sorted(rubric.items()))


def _judge_id(model: ModelDef) -> str:
    return f"{model.provider}/{model.id}"


def _judge_temperature(model: ModelDef) -> float | None:
    # gpt-5 family rejects temperature != 1.0 — match executor._judge_temperature.
    if model.provider == "openai" and model.id.startswith("gpt-5"):
        return None
    return 0.0


async def _generate_eval_steps(
    wrapper: LLMWrapper,
    model: ModelDef,
    criteria: str,
    rubric: dict[int, str],
) -> list[str] | None:
    """One CoT call to materialize ordered evaluation steps."""
    prompt = _EVAL_STEPS_PROMPT.format(criteria=criteria, rubric=_rubric_block(rubric))
    ctx = Context(
        system_prompt="You design evaluation rubrics. Respond ONLY with a JSON list.",
        messages=[UserMessage(content=prompt)],
        temperature=_judge_temperature(model),
        max_tokens=800,
    )
    for attempt in range(_MAX_RETRIES):
        try:
            msg = await wrapper.complete(
                model, ctx, session_id="eval-dag", conversation_id="geval-steps"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("eval-steps call failed (attempt %d): %s", attempt, e)
            continue
        raw = _extract_text(msg)
        # JSON list — match [...] specifically, not the generic { } regex
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            logger.warning("eval-steps non-list on attempt %d: %r", attempt, raw[:160])
            continue
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and all(isinstance(s, str) for s in data):
            return data[:6]
    return None


async def score_leaf(
    leaf: GEvalLeaf,
    wrapper: LLMWrapper,
    model: ModelDef,
    case: dict,
    *,
    cache: dict[tuple[str, str], list[str]] | None = None,
):
    """Score one G-Eval leaf. Returns a NodeOutput-compatible object."""
    from eval.dag.executor import NodeOutput  # avoid circular import at module load

    judge = _judge_id(model)
    key = (judge, leaf.criteria)
    t0 = time.perf_counter()

    steps: list[str] | None = None
    steps_in = steps_out = 0
    steps_cost = 0.0
    if cache is not None and key in cache:
        steps = cache[key]
    else:
        steps = await _generate_eval_steps(wrapper, model, leaf.criteria, leaf.rubric)
        if steps is None:
            return NodeOutput(
                node=leaf.name,
                score=None,
                reason="eval-steps generation failed",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        if cache is not None:
            cache[key] = steps

    steps_block = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
    scoring_prompt = _SCORE_PROMPT.format(
        criteria=leaf.criteria,
        rubric=_rubric_block(leaf.rubric),
        steps=steps_block,
        prompt=case.get("prompt", "(none)"),
        expected=case.get("expected", "(none)"),
        response=case.get("response", "(none)"),
    )
    ctx = Context(
        system_prompt="You are an impartial evaluator. Respond ONLY with JSON.",
        messages=[UserMessage(content=scoring_prompt)],
        temperature=_judge_temperature(model),
        max_tokens=600,
    )

    last_raw = ""
    for attempt in range(_MAX_RETRIES):
        try:
            msg = await wrapper.complete(
                model, ctx, session_id="eval-dag", conversation_id="geval-score"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("geval score call failed (attempt %d): %s", attempt, e)
            continue
        last_raw = _extract_text(msg)
        m = _JSON_RE.search(last_raw)
        if not m:
            logger.warning("geval score non-JSON on attempt %d: %r", attempt, last_raw[:160])
            continue
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        try:
            raw_score = int(parsed["score"])
        except (KeyError, ValueError, TypeError):
            continue
        raw_score = max(0, min(10, raw_score))
        score_05 = raw_score / 2.0  # 0-10 → 0-5
        usage = msg.usage
        return NodeOutput(
            node=leaf.name,
            score=score_05,
            reason=str(parsed.get("reason", "")),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            input_tokens=usage.input_tokens + steps_in,
            output_tokens=usage.output_tokens + steps_out,
            cost_usd=usage.cost_usd(model) + steps_cost,
        )

    return NodeOutput(
        node=leaf.name,
        score=None,
        reason=f"geval score parse failed: {last_raw[:120]!r}",
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )
