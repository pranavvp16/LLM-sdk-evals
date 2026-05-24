"""Argument correctness axis DAG (L3 agent).

  any_tools_called?
    F → verdict:not_applicable (5)         # no calls = nothing to grade
    T → args_schema_valid?
          F → verdict:malformed (1)
          T → args_match_intent?
                F → verdict:right_shape_wrong_intent (3)
                T → verdict:correct (5)
"""

from __future__ import annotations

from typing import Any

from eval.dag.nodes import AxisDAG, BinaryNode, Verdict
from eval.dag.panel import AxisPanelResult


ANY_CALLS_PROMPT = """\
Did the assistant call at least one tool in this trajectory?

## Tool calls made
{tool_calls_text}

Respond ONLY with JSON:
{{
  "verdict": true,
  "reason": "<just count: '0 calls' or 'N calls'>"
}}

verdict=true if any tool was called.
"""


SCHEMA_VALID_PROMPT = """\
Inspect every tool call's arguments and decide if the arguments are
well-formed for the tool: required arguments present, types correct, no
extra keys, no malformed JSON, no nonsense values.

## Tool calls made (with arguments)
{tool_calls_text}

Respond ONLY with JSON:
{{
  "verdict": true,
  "reason": "<1-2 sentences citing the worst-formed call, or 'all valid'>"
}}

verdict=true iff every tool call's arguments are schema-valid.
"""


INTENT_MATCH_PROMPT = """\
The arguments are well-formed. Now check: do they actually reflect the
user's intent? Examples of intent mismatch:
  - User says "tomorrow at 3pm" but the when= arg is today
  - User asks about "Ankur" but participant= is empty or a placeholder
  - Numbers swapped, units wrong, location misspelled

## User prompt
{prompt}

## Tool calls made (with arguments)
{tool_calls_text}

Respond ONLY with JSON:
{{
  "verdict": true,
  "reason": "<1-2 sentences>"
}}

verdict=true iff arguments reflect the user's intent for every call.
"""


DAG = AxisDAG(
    axis_name="argument_correctness",
    entry="any_calls",
    nodes={
        "any_calls": BinaryNode(
            name="any_calls",
            prompt=ANY_CALLS_PROMPT,
            on_true="schema_valid",
            on_false="verdict:not_applicable",
        ),
        "schema_valid": BinaryNode(
            name="schema_valid",
            prompt=SCHEMA_VALID_PROMPT,
            on_true="intent_match",
            on_false="verdict:malformed",
        ),
        "intent_match": BinaryNode(
            name="intent_match",
            prompt=INTENT_MATCH_PROMPT,
            on_true="verdict:correct",
            on_false="verdict:right_shape_wrong_intent",
        ),
    },
    verdicts={
        "not_applicable":            Verdict(key="not_applicable",            score=5.0),
        "correct":                   Verdict(key="correct",                   score=5.0),
        "right_shape_wrong_intent":  Verdict(key="right_shape_wrong_intent",  score=3.0),
        "malformed":                 Verdict(key="malformed",                 score=1.0),
    },
)


def postprocess(result: AxisPanelResult, case: dict) -> dict[str, Any]:
    return {}
