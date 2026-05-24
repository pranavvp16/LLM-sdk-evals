"""Per-judge DAG traversal.

Walks an ``AxisDAG`` from its entry node, calling the supplied judge model
at every Binary/NonBinary node and at every GEval leaf. Returns a
``JudgeAxisResult`` capturing the full path, per-node outputs, and roll-up
cost/latency.

A node prompt is a Python format-string rendered against the ``case`` dict
(typically `{prompt, response, expected, ...}`). The judge is asked for
JSON: `{"verdict": ..., "reason": ...}` for branch nodes, `{"score": int,
"reason": ...}` for G-Eval leaves. Up to 3 retries on JSON parse failure
with varied temperature so a malformed-JSON response doesn't fail
identically every time; on terminal failure the traversal halts and the
partial path is returned with ``status="judge_failed"``. Retry-attempt
token spend is still folded into the leaf/branch ``NodeOutput`` and into
the run-level cost rollup.
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

# Vary temperature across retries — repeating the same prompt at the same
# temp tends to produce the same malformed output. gpt-5 rejects any temp
# value other than the default, so for that family we skip the override
# (see ``_judge_temperature``).
_RETRY_TEMPERATURES: tuple[float, ...] = (0.0, 0.3, 0.6)


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


def _coerce_bool(v: object) -> bool | None:
    """Robust truthiness coercion for LLM-emitted binary verdicts.

    Returns None when the input cannot be unambiguously interpreted — caller
    treats that as a parse failure and retries.

    Important: ``bool(verdict)`` is wrong because non-empty strings like
    ``"false"`` evaluate truthy. LLMs often return string booleans even when
    asked for JSON true/false.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "t"):
            return True
        if s in ("false", "0", "no", "f"):
            return False
        return None
    return None


def _render(prompt: str, case: dict) -> str:
    """Format the node prompt with the case dict, tolerating missing keys."""
    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "(none)"
    return prompt.format_map(_SafeDict(case))


def _judge_id(model: ModelDef) -> str:
    return f"{model.provider}/{model.id}"


def _retry_temperature(model: ModelDef, attempt: int) -> float | None:
    """Vary temperature across retries; respect gpt-5's default-only rule."""
    if model.provider == "openai" and model.id.startswith("gpt-5"):
        return None
    idx = min(attempt, len(_RETRY_TEMPERATURES) - 1)
    return _RETRY_TEMPERATURES[idx]


async def _call_judge(
    wrapper: LLMWrapper,
    model: ModelDef,
    prompt: str,
    *,
    max_tokens: int,
    conversation_id: str,
    temperature: float | None,
    system: str = "You are an impartial evaluator. Respond ONLY with JSON.",
) -> tuple[AssistantMessage | None, float]:
    """Single judge call, returns (message_or_None_on_error, latency_ms)."""
    ctx = Context(
        system_prompt=system,
        messages=[UserMessage(content=prompt)],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    t0 = time.perf_counter()
    try:
        msg = await wrapper.complete(
            model, ctx, session_id="eval-dag", conversation_id=conversation_id
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("judge call failed: %s/%s: %s", model.provider, model.id, e)
        return None, (time.perf_counter() - t0) * 1000.0
    return msg, (time.perf_counter() - t0) * 1000.0


@dataclass
class _BranchOutcome:
    """Return shape for ``_resolve_branch_node``.

    ``out`` and ``target`` are set on success; ``out`` is None on terminal
    failure. ``partial_usage`` aggregates token / cost / latency across
    every retry attempt so the executor can fold them into the final
    JudgeAxisResult even when the node ultimately failed.
    """
    out: NodeOutput | None
    target: str | None
    partial_usage: dict[str, float]


async def _resolve_branch_node(
    node: BinaryNode | NonBinaryNode,
    wrapper: LLMWrapper,
    model: ModelDef,
    case: dict,
    *,
    conversation_id: str,
) -> _BranchOutcome:
    """Run one branch node. Retries vary temperature; partial spend is
    always returned, even on terminal failure."""
    rendered = _render(node.prompt, case)
    partial: dict[str, float] = {
        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0,
    }
    last_error: str = ""

    for attempt in range(_MAX_RETRIES):
        msg, latency_ms = await _call_judge(
            wrapper, model, rendered,
            max_tokens=600,
            conversation_id=conversation_id,
            temperature=_retry_temperature(model, attempt),
        )
        partial["latency_ms"] += latency_ms
        if msg is None:
            last_error = "wrapper exception"
            continue

        usage = msg.usage
        partial["input_tokens"] += usage.input_tokens
        partial["output_tokens"] += usage.output_tokens
        partial["cost_usd"] += usage.cost_usd(model)

        raw = _extract_text(msg)
        parsed = _parse_json(raw)
        if parsed is None or "verdict" not in parsed:
            last_error = f"non-JSON: {raw[:120]!r}"
            logger.warning(
                "node %s: non-JSON on attempt %d (%r)", node.name, attempt, raw[:160]
            )
            continue

        raw_verdict = parsed["verdict"]
        reason = str(parsed.get("reason", ""))

        if isinstance(node, BinaryNode):
            verdict_bool = _coerce_bool(raw_verdict)
            if verdict_bool is None:
                last_error = f"ambiguous binary verdict {raw_verdict!r}"
                logger.warning(
                    "node %s: ambiguous binary verdict %r on attempt %d",
                    node.name, raw_verdict, attempt,
                )
                continue
            out = NodeOutput(
                node=node.name,
                verdict=verdict_bool,         # canonical bool — keeps κ + aggregation honest
                reason=reason,
                latency_ms=partial["latency_ms"],
                input_tokens=partial["input_tokens"],
                output_tokens=partial["output_tokens"],
                cost_usd=partial["cost_usd"],
            )
            target = node.on_true if verdict_bool else node.on_false
            return _BranchOutcome(out=out, target=target, partial_usage=partial)

        # NonBinary: verdict is a label string
        label = str(raw_verdict).strip().lower()
        target = node.branches.get(label)
        if target is None:
            last_error = f"unknown branch label {label!r}"
            logger.warning(
                "node %s: unknown branch label %r (allowed: %s)",
                node.name, label, list(node.branches.keys()),
            )
            continue
        out = NodeOutput(
            node=node.name,
            verdict=label,
            reason=reason,
            latency_ms=partial["latency_ms"],
            input_tokens=partial["input_tokens"],
            output_tokens=partial["output_tokens"],
            cost_usd=partial["cost_usd"],
        )
        return _BranchOutcome(out=out, target=target, partial_usage=partial)

    logger.warning("node %s: terminal failure (%s)", node.name, last_error)
    return _BranchOutcome(out=None, target=None, partial_usage=partial)


async def run_axis(
    dag: AxisDAG,
    judge_model: ModelDef,
    case: dict,
    wrapper: LLMWrapper,
    geval_cache: dict[tuple[str, str, str], list[str]] | None = None,
) -> JudgeAxisResult:
    """Traverse the DAG for a single judge; return per-judge result.

    Per-call ``conversation_id`` is derived from the case's ``prompt_id`` (if
    set), the axis name, the judge ``provider/model`` (so two judges that
    share a model id never collide on the same log row), and the node name.
    """
    result = JudgeAxisResult(judge_id=_judge_id(judge_model), score=None)
    current = dag.entry

    prompt_id = str(case.get("prompt_id") or case.get("id") or "p?")
    # Include provider AND id so that two judges sharing a model id (e.g.
    # both Sonnet for an ablation run) get distinct conversations.
    convo_prefix = (
        f"eval-{prompt_id}-{dag.axis_name}-"
        f"{judge_model.provider}-{judge_model.id}"
    )

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
                node, wrapper, judge_model, case,
                cache=geval_cache,
                conversation_id=f"{convo_prefix}-{node.name}",
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
        outcome = await _resolve_branch_node(
            node, wrapper, judge_model, case,
            conversation_id=f"{convo_prefix}-{node.name}",
        )
        # Always fold retry-attempt spend into the roll-up, even on failure.
        result.latency_ms += outcome.partial_usage["latency_ms"]
        result.input_tokens += int(outcome.partial_usage["input_tokens"])
        result.output_tokens += int(outcome.partial_usage["output_tokens"])
        result.cost_usd += outcome.partial_usage["cost_usd"]

        if outcome.out is None or outcome.target is None:
            result.status = "judge_failed"
            result.error = f"branch node {node.name} failed"
            return result

        result.node_outputs.append(outcome.out)
        if isinstance(node, BinaryNode):
            verdict_label = "T" if outcome.out.verdict is True else "F"
        else:
            verdict_label = str(outcome.out.verdict)
        result.verdict_path.append(f"{node.name}:{verdict_label}")
        current = outcome.target
