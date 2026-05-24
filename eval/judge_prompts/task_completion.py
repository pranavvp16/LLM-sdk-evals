"""Task completion axis DAG (L3 agent).

  request_addressed?
    F → verdict:not_addressed (0)
    T → completion_degree? (NonBinary: full | partial | attempted)
          full      → verdict:complete (5)
          partial   → partial_leaf (G-Eval)
          attempted → verdict:attempted (1)
"""

from __future__ import annotations

from typing import Any

from eval.dag.nodes import AxisDAG, BinaryNode, GEvalLeaf, NonBinaryNode, Verdict
from eval.dag.panel import AxisPanelResult


REQUEST_ADDRESSED_PROMPT = """\
Did the assistant's final response address what the user asked for, in
substance? (Not formatting — substance.)

## User prompt
{prompt}

## Expected behavior (advisory)
{expected}

## Assistant final response
{response}

## Tool calls made (for context)
{tool_calls_text}

Respond ONLY with JSON:
{{
  "verdict": true,
  "reason": "<1-2 sentences>"
}}

verdict=false if the response is non-responsive, refuses a benign request,
hits max_hops without an answer, or talks about something else entirely.
"""


COMPLETION_DEGREE_PROMPT = """\
The response addressed the request. Classify how completely:

  full      — Every part of the request was fulfilled.
  partial   — A multi-step request had some parts done and others not.
  attempted — The assistant tried but did not deliver a usable answer.

## User prompt
{prompt}

## Assistant final response
{response}

## Tool calls made
{tool_calls_text}

Respond ONLY with JSON:
{{
  "verdict": "<full|partial|attempted>",
  "reason": "<1-2 sentences>"
}}
"""


PARTIAL_RUBRIC: dict[int, str] = {
    0:  "Less than 25% of the multi-step request completed.",
    2:  "About half completed; missing parts NOT acknowledged.",
    4:  "Half completed; missing parts vaguely acknowledged.",
    6:  "Half completed; missing parts explicitly acknowledged.",
    8:  "Most parts completed; one minor gap acknowledged.",
    10: "Substantially complete; gap is trivial and explicitly flagged.",
}


DAG = AxisDAG(
    axis_name="task_completion",
    entry="request_addressed",
    nodes={
        "request_addressed": BinaryNode(
            name="request_addressed",
            prompt=REQUEST_ADDRESSED_PROMPT,
            on_true="completion_degree",
            on_false="verdict:not_addressed",
        ),
        "completion_degree": NonBinaryNode(
            name="completion_degree",
            prompt=COMPLETION_DEGREE_PROMPT,
            branches={
                "full":      "verdict:complete",
                "partial":   "partial_leaf",
                "attempted": "verdict:attempted",
            },
        ),
        "partial_leaf": GEvalLeaf(
            name="partial_leaf",
            criteria=(
                "Rate the partial completion. Score higher when the missing "
                "parts are explicitly acknowledged and the user knows what "
                "remains; lower when missing parts are silently dropped."
            ),
            rubric=PARTIAL_RUBRIC,
            eval_params=("prompt", "response", "expected", "tool_calls_text"),
        ),
    },
    verdicts={
        "not_addressed": Verdict(key="not_addressed", score=0.0),
        "complete":      Verdict(key="complete",      score=5.0),
        "attempted":     Verdict(key="attempted",     score=1.0),
    },
)


def postprocess(result: AxisPanelResult, case: dict) -> dict[str, Any]:
    return {}
