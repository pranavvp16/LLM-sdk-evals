"""Three-judge fan-out + aggregation for one axis.

``score_axis_panel`` runs every judge in the panel against the same DAG in
parallel via ``asyncio.gather`` and produces an ``AxisPanelResult`` with:

  - aggregated_score          : majority verdict if path-unanimous; else mean
  - verdict_path_majority     : the verdict path the majority took (or first
                                judge's path on full disagreement)
  - judges                    : per-judge JudgeAxisResult (dict form)
  - agreement                 : {binary_unanimous, geval_stdev, kappa_avg}

Aggregation rules:
  * If all surviving (non-failed) judges land on the same final verdict key,
    use that verdict's score; mark ``binary_unanimous = True``.
  * Otherwise take the mean of per-judge scores; ``binary_unanimous = False``.
  * G-Eval leaves emit a numeric score directly; their stdev across surviving
    judges is reported as ``geval_stdev`` (None when no G-Eval leaf was hit).
  * ``kappa_avg`` is the mean pairwise Cohen's κ over binary node verdicts
    that all surviving judges traversed (NaN → None).
"""

from __future__ import annotations

import asyncio
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sdk import LLMWrapper
from sdk.registry import get_model

from eval.dag.executor import JudgeAxisResult, run_axis
from eval.dag.kappa import mean_pairwise_kappa
from eval.dag.nodes import AxisDAG


JudgeSpec = tuple[str, str]


@dataclass
class AxisPanelResult:
    aggregated_score: float | None
    verdict_path_majority: list[str]
    judges: list[JudgeAxisResult]
    binary_unanimous: bool
    geval_stdev: float | None
    kappa_avg: float | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregated_score": self.aggregated_score,
            "verdict_path_majority": self.verdict_path_majority,
            "judges": [j.to_dict() for j in self.judges],
            "agreement": {
                "binary_unanimous": self.binary_unanimous,
                "geval_stdev": self.geval_stdev,
                "kappa_avg": self.kappa_avg,
            },
            **self.extra,
        }


def _verdict_key_of(path: list[str]) -> str:
    """The terminal segment of a verdict path uniquely identifies the leaf reached."""
    return path[-1] if path else ""


def _binary_node_verdicts(judges: list[JudgeAxisResult]) -> dict[str, list[bool]]:
    """For each binary node visited by ALL surviving judges, collect their booleans."""
    by_node: dict[str, list[bool]] = {}
    if not judges:
        return by_node
    node_sets = [
        {n.node: n.verdict for n in j.node_outputs if isinstance(n.verdict, bool)}
        for j in judges
    ]
    common = set(node_sets[0].keys())
    for s in node_sets[1:]:
        common &= set(s.keys())
    for node in common:
        verdicts = [s[node] for s in node_sets]
        if all(isinstance(v, bool) for v in verdicts):
            by_node[node] = list(verdicts)
    return by_node


def _aggregate(judges: list[JudgeAxisResult]) -> AxisPanelResult:
    survivors = [j for j in judges if j.status == "success" and j.score is not None]

    if not survivors:
        return AxisPanelResult(
            aggregated_score=None,
            verdict_path_majority=[],
            judges=judges,
            binary_unanimous=False,
            geval_stdev=None,
            kappa_avg=None,
        )

    # Path-unanimity check
    final_keys = [_verdict_key_of(j.verdict_path) for j in survivors]
    unanimous = len(set(final_keys)) == 1

    if unanimous:
        aggregated = survivors[0].score
        path = survivors[0].verdict_path
    else:
        # Majority verdict path; tie-break by lowest judge index
        counter = Counter(final_keys)
        top_key, _ = counter.most_common(1)[0]
        majority = next(j for j in survivors if _verdict_key_of(j.verdict_path) == top_key)
        path = majority.verdict_path
        aggregated = statistics.mean(j.score for j in survivors)

    # G-Eval stdev: do any judges have G-Eval leaf scores?
    geval_scores: list[float] = []
    for j in survivors:
        for n in j.node_outputs:
            if n.verdict is None and n.score is not None:
                geval_scores.append(n.score)
    geval_stdev = (
        statistics.pstdev(geval_scores) if len(geval_scores) >= 2 else None
    )

    # κ over commonly-traversed binary nodes
    by_node = _binary_node_verdicts(survivors)
    if by_node:
        # Transpose to per-judge boolean lists across nodes
        node_names = list(by_node.keys())
        per_judge = [[by_node[n][i] for n in node_names] for i in range(len(survivors))]
        k = mean_pairwise_kappa(per_judge)
        kappa_avg = None if (k is None or math.isnan(k)) else k
    else:
        kappa_avg = None

    return AxisPanelResult(
        aggregated_score=aggregated,
        verdict_path_majority=path,
        judges=judges,
        binary_unanimous=unanimous,
        geval_stdev=geval_stdev,
        kappa_avg=kappa_avg,
    )


async def score_axis_panel(
    dag: AxisDAG,
    case: dict,
    wrapper: LLMWrapper,
    judges: list[JudgeSpec],
    *,
    geval_cache: dict[tuple[str, str], list[str]] | None = None,
) -> AxisPanelResult:
    """Fan out the same DAG to every judge in parallel; aggregate."""
    tasks = [
        run_axis(dag, get_model(prov, mid), case, wrapper, geval_cache)
        for prov, mid in judges
    ]
    judge_results = await asyncio.gather(*tasks)
    return _aggregate(list(judge_results))
