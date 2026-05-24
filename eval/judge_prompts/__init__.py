"""Per-axis DAG + judge prompt bundles for the eval ensemble.

Each module exports:
  * ``DAG: AxisDAG``                          — the per-axis decision graph
  * ``postprocess(result, case) -> dict``     — axis-specific extras to merge

L2 (static): hallucination, bias, safety, role_violation
L3 (agent):  tool_selection, argument_correctness, task_completion,
              output_grounding, safety_with_tools
"""

from eval.judge_prompts.argument_correctness import (
    DAG as ARGUMENT_CORRECTNESS_DAG,
    postprocess as argument_correctness_postprocess,
)
from eval.judge_prompts.bias import DAG as BIAS_DAG, postprocess as bias_postprocess
from eval.judge_prompts.hallucination import (
    DAG as HALLUCINATION_DAG,
    postprocess as hallucination_postprocess,
)
from eval.judge_prompts.output_grounding import (
    DAG as OUTPUT_GROUNDING_DAG,
    postprocess as output_grounding_postprocess,
)
from eval.judge_prompts.role_violation import (
    DAG as ROLE_VIOLATION_DAG,
    postprocess as role_violation_postprocess,
)
from eval.judge_prompts.safety import DAG as SAFETY_DAG, postprocess as safety_postprocess
from eval.judge_prompts.safety_with_tools import (
    DAG as SAFETY_WITH_TOOLS_DAG,
    postprocess as safety_with_tools_postprocess,
)
from eval.judge_prompts.task_completion import (
    DAG as TASK_COMPLETION_DAG,
    postprocess as task_completion_postprocess,
)
from eval.judge_prompts.tool_selection import (
    DAG as TOOL_SELECTION_DAG,
    postprocess as tool_selection_postprocess,
)


L2_AXES: list[tuple[str, "AxisDAG", callable]] = [  # type: ignore[name-defined]
    ("hallucination",  HALLUCINATION_DAG,   hallucination_postprocess),
    ("bias",           BIAS_DAG,            bias_postprocess),
    ("safety",         SAFETY_DAG,          safety_postprocess),
    ("role_violation", ROLE_VIOLATION_DAG,  role_violation_postprocess),
]

L3_AXES: list[tuple[str, "AxisDAG", callable]] = [  # type: ignore[name-defined]
    ("tool_selection",       TOOL_SELECTION_DAG,       tool_selection_postprocess),
    ("argument_correctness", ARGUMENT_CORRECTNESS_DAG, argument_correctness_postprocess),
    ("task_completion",      TASK_COMPLETION_DAG,      task_completion_postprocess),
    ("output_grounding",     OUTPUT_GROUNDING_DAG,     output_grounding_postprocess),
    ("safety_with_tools",    SAFETY_WITH_TOOLS_DAG,    safety_with_tools_postprocess),
]

__all__ = [
    "HALLUCINATION_DAG", "hallucination_postprocess",
    "BIAS_DAG", "bias_postprocess",
    "SAFETY_DAG", "safety_postprocess",
    "ROLE_VIOLATION_DAG", "role_violation_postprocess",
    "TOOL_SELECTION_DAG", "tool_selection_postprocess",
    "ARGUMENT_CORRECTNESS_DAG", "argument_correctness_postprocess",
    "TASK_COMPLETION_DAG", "task_completion_postprocess",
    "OUTPUT_GROUNDING_DAG", "output_grounding_postprocess",
    "SAFETY_WITH_TOOLS_DAG", "safety_with_tools_postprocess",
    "L2_AXES",
    "L3_AXES",
]
