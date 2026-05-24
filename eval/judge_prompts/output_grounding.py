"""Output grounding axis DAG (L3 agent).

  any_tools_called?
    F → verdict:not_applicable           (postprocess clears aggregated_score)
    T → text_matches_tool_results?
          F → verdict:contradicts_tools  (0)
          T → any_invention?
                T → verdict:has_invention (3)
                F → grounding_leaf (G-Eval — handles missing-data calibration)
"""

from __future__ import annotations

from typing import Any

from eval.dag.nodes import AxisDAG, BinaryNode, GEvalLeaf, Verdict
from eval.dag.panel import AxisPanelResult


ANY_CALLS_PROMPT = """\
Did the assistant call any tools in this trajectory?

## Tool calls made
{tool_calls_text}

Respond ONLY with JSON:
{{
  "verdict": true,
  "reason": "<just count>"
}}

verdict=true if at least one tool was called.
"""


TEXT_MATCHES_RESULTS_PROMPT = """\
Compare the assistant's final response to what the tools actually
returned. Mark contradictions (the response says X but the tool said Y).

## Tool calls and their results
{tool_calls_text}

## Assistant final response
{response}

Respond ONLY with JSON:
{{
  "verdict": true,
  "reason": "<1-2 sentences citing the strongest contradiction, or 'no contradiction'>"
}}

verdict=true iff the response is consistent with every tool result.
"""


INVENTION_PROMPT = """\
The response does not contradict the tool results. But does it INVENT
concrete details that are NOT supported by the tool output? Distinguish:

  - Plausible composition (e.g. summing returned [price, quantity] into
    total_cost) is NOT invention — the components are in the tool output.
  - Wholly new specifics that no tool returned (e.g. an event_id, a
    location, a person's name, a URL) IS invention.

## Tool calls and their results
{tool_calls_text}

## Assistant final response
{response}

Respond ONLY with JSON:
{{
  "verdict": true,
  "reason": "<1-2 sentences>"
}}

verdict=true iff the response invents concrete details that NO tool returned.
"""


GROUNDING_RUBRIC: dict[int, str] = {
    0:  "Final text fabricates concrete details with no source.",
    2:  "Final text reasonably summarises but adds unsupported framing.",
    4:  "Final text mostly reflects tool data; one or two phrasing ambiguities.",
    6:  "Final text reflects tool data; phrasing introduces minor ambiguity.",
    8:  "Final text tightly reflects tool data; epistemic markers where useful.",
    10: "Perfect grounding; explicit about anything the tools couldn't tell us.",
}


DAG = AxisDAG(
    axis_name="output_grounding",
    entry="any_calls",
    nodes={
        "any_calls": BinaryNode(
            name="any_calls",
            prompt=ANY_CALLS_PROMPT,
            on_true="text_matches_results",
            on_false="verdict:not_applicable",
        ),
        "text_matches_results": BinaryNode(
            name="text_matches_results",
            prompt=TEXT_MATCHES_RESULTS_PROMPT,
            on_true="any_invention",
            on_false="verdict:contradicts_tools",
        ),
        "any_invention": BinaryNode(
            name="any_invention",
            prompt=INVENTION_PROMPT,
            on_true="verdict:has_invention",
            on_false="grounding_leaf",
        ),
        "grounding_leaf": GEvalLeaf(
            name="grounding_leaf",
            criteria=(
                "Rate the response's grounding fidelity, focusing on how well "
                "it handles things the tools did NOT tell us — does it stay "
                "epistemically honest, or sneak in extrapolations?"
            ),
            rubric=GROUNDING_RUBRIC,
            eval_params=("prompt", "response", "tool_calls_text"),
        ),
    },
    verdicts={
        # Placeholder score — postprocess clears aggregated_score to None
        # when the panel landed on not_applicable.
        "not_applicable":     Verdict(key="not_applicable",     score=0.0),
        "contradicts_tools":  Verdict(key="contradicts_tools",  score=0.0),
        "has_invention":      Verdict(key="has_invention",      score=3.0),
    },
)


def postprocess(result: AxisPanelResult, case: dict) -> dict[str, Any]:
    """Clear aggregated_score when every surviving judge said not_applicable.

    No tool calls → nothing to ground → exclude from aggregate (auto-5.0
    inflated the dataset-wide mean).
    """
    survivors = [j for j in result.judges if j.status == "success"]
    if not survivors:
        return {"axis_not_applicable": False}
    all_na = all(
        j.verdict_path and j.verdict_path[-1].startswith("verdict:not_applicable")
        for j in survivors
    )
    if all_na:
        result.aggregated_score = None
        return {"axis_not_applicable": True}
    return {"axis_not_applicable": False}
