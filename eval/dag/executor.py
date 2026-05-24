"""Per-judge DAG traversal.

Walks an ``AxisDAG`` from its entry node, calling the supplied judge model
at every Binary/NonBinary node and at every GEval leaf. Returns a
``JudgeAxisResult`` capturing the full path, per-node outputs, and roll-up
cost/latency.

A node prompt is a Python format-string rendered against the ``case`` dict
(typically `{prompt, response, expected, ...}`). The judge is asked for
JSON: `{"verdict": ..., "reason": ...}` for branch nodes, `{"score": int,
"reason": ...}` for G-Eval leaves. Up to 3 retries on JSON parse failure;
on terminal failure the traversal halts and the partial path is returned
with ``status="judge_failed"``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from sdk import Context, LLMWrapper, UserMessage
from sdk.types import AssistantMessage, ModelDef, TextContent

from eval.dag.geval import score_leaf
from eval.dag.nodes import AxisDAG, BinaryNode, GEvalLeaf, NonBinaryNode

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_MAX_RETRIES = 3


@dataclass
class NodeOutput:
    node: str
    verdict: bool | str | None = None
    score: float | None = None
    reason: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class JudgeAxisResult:
    judge_id: str
    score: float | None
    verdict_path: list[str] = field(default_factory=list)
    node_outputs: list[NodeOutput] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    status: str = "success"           # success | judge_failed
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_id": self.judge_id,
            "score": self.score,
            "verdict_path": self.verdict_path,
            "node_outputs": [
                {
                    "node": n.node,
                    "verdict": n.verdict,
                    "score": n.score,
                    "reason": n.reason,
                    "latency_ms": n.latency_ms,
                    "cost_usd": n.cost_usd,
                }
                for n in self.node_outputs
            ],
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "status": self.status,
            "error": self.error,
        }


def _extract_text(msg: AssistantMessage) -> str:
    return "".join(b.text for b in msg.content if isinstance(b, TextContent))


def _parse_json(text: str) -> dict | None:
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _render(prompt: str, case: dict) -> str:
    """Format the node prompt with the case dict, tolerating missing keys."""
    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "(none)"
    return prompt.format_map(_SafeDict(case))


def _judge_id(model: ModelDef) -> str:
    return f"{model.provider}/{model.id}"


def _judge_temperature(model: ModelDef) -> float | None:
    """Most providers happily accept ``temperature=0.0`` for deterministic judging.

    OpenAI's gpt-5 family rejects any value other than the default of 1.0
    (returns HTTP 400). For those models we omit the temperature param and
    accept slightly higher response variance.
    """
    if model.provider == "openai" and model.id.startswith("gpt-5"):
        return None
    return 0.0


async def _call_judge(
    wrapper: LLMWrapper,
    model: ModelDef,
    prompt: str,
    *,
    max_tokens: int,
    system: str = "You are an impartial evaluator. Respond ONLY with JSON.",
) -> tuple[AssistantMessage | None, float]:
    """Single judge call, returns (message_or_None_on_error, latency_ms)."""
    ctx = Context(
        system_prompt=system,
        messages=[UserMessage(content=prompt)],
        temperature=_judge_temperature(model),
        max_tokens=max_tokens,
    )
    t0 = time.perf_counter()
    try:
        msg = await wrapper.complete(
            model, ctx, session_id="eval-dag", conversation_id=f"axis-{model.id}"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("judge call failed: %s/%s: %s", model.provider, model.id, e)
        return None, (time.perf_counter() - t0) * 1000.0
    return msg, (time.perf_counter() - t0) * 1000.0


async def _resolve_branch_node(
    node: BinaryNode | NonBinaryNode,
    wrapper: LLMWrapper,
    model: ModelDef,
    case: dict,
) -> tuple[NodeOutput | None, str | None]:
    """Run one branch node, return (NodeOutput, next_target). next_target is None on failure."""
    rendered = _render(node.prompt, case)
    for attempt in range(_MAX_RETRIES):
        msg, latency_ms = await _call_judge(wrapper, model, rendered, max_tokens=600)
        if msg is None:
            continue
        raw = _extract_text(msg)
        parsed = _parse_json(raw)
        if parsed is None or "verdict" not in parsed:
            logger.warning(
                "node %s: non-JSON on attempt %d (%r)", node.name, attempt, raw[:160]
            )
            continue

        verdict = parsed["verdict"]
        reason = str(parsed.get("reason", ""))
        usage = msg.usage
        out = NodeOutput(
            node=node.name,
            verdict=verdict,
            reason=reason,
            latency_ms=latency_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd(model),
        )

        if isinstance(node, BinaryNode):
            target = node.on_true if bool(verdict) else node.on_false
            return out, target
        # NonBinary: verdict is a label string
        label = str(verdict).strip().lower()
        target = node.branches.get(label)
        if target is None:
            # Unknown label — try once more
            logger.warning(
                "node %s: unknown branch label %r (allowed: %s)",
                node.name, label, list(node.branches.keys()),
            )
            continue
        return out, target

    return None, None


async def run_axis(
    dag: AxisDAG,
    judge_model: ModelDef,
    case: dict,
    wrapper: LLMWrapper,
    geval_cache: dict[tuple[str, str], list[str]] | None = None,
) -> JudgeAxisResult:
    """Traverse the DAG for a single judge; return per-judge result."""
    result = JudgeAxisResult(judge_id=_judge_id(judge_model), score=None)
    current = dag.entry

    while True:
        if current.startswith("verdict:"):
            verdict_key = current.split(":", 1)[1]
            verdict = dag.verdict(verdict_key)
            result.verdict_path.append(f"verdict:{verdict_key}={verdict.score:.1f}")
            result.score = verdict.score
            return result

        node = dag.get(current)

        if isinstance(node, GEvalLeaf):
            leaf_out = await score_leaf(
                node, wrapper, judge_model, case, cache=geval_cache,
            )
            result.node_outputs.append(leaf_out)
            result.latency_ms += leaf_out.latency_ms
            result.input_tokens += leaf_out.input_tokens
            result.output_tokens += leaf_out.output_tokens
            result.cost_usd += leaf_out.cost_usd
            if leaf_out.score is None:
                result.status = "judge_failed"
                result.error = f"geval leaf {node.name} failed"
                return result
            result.verdict_path.append(f"{node.name}:{leaf_out.score:.1f}")
            result.score = leaf_out.score
            return result

        # Branch node (Binary / NonBinary)
        out, target = await _resolve_branch_node(node, wrapper, judge_model, case)
        if out is None or target is None:
            result.status = "judge_failed"
            result.error = f"branch node {node.name} failed"
            return result
        result.node_outputs.append(out)
        result.latency_ms += out.latency_ms
        result.input_tokens += out.input_tokens
        result.output_tokens += out.output_tokens
        result.cost_usd += out.cost_usd
        verdict_label = (
            "T" if isinstance(node, BinaryNode) and bool(out.verdict) else
            "F" if isinstance(node, BinaryNode) else str(out.verdict)
        )
        result.verdict_path.append(f"{node.name}:{verdict_label}")
        current = target
