"""Three-judge fan-out + aggregation for one axis.

``score_axis_panel`` runs every judge in the panel against the same DAG in
parallel via ``asyncio.gather`` and produces an ``AxisPanelResult`` with:

  - aggregated_score          : majority verdict if path-unanimous; else mean
  - verdict_path_majority     : the verdict path the majority took (or first
                                surviving judge's path on full disagreement)
  - judges                    : per-judge JudgeAxisResult (dict form)
  - panel_status              : "success" | "partial" | "all_failed"
  - agreement                 : {binary_unanimous, geval_stdev, kappa_avg, kappa_status}

Aggregation rules:
  * Only judges with ``status=='success'`` contribute (failed judges keep
    their entry in ``judges`` for the UI but are excluded from aggregation
    AND from path-unanimity AND from κ).
  * If all surviving judges land on the same final verdict key, use that
    verdict's score; mark ``binary_unanimous = True``.
  * Otherwise take the mean of per-judge scores.
  * G-Eval leaves emit a numeric score directly; their stdev across
    surviving judges is reported as ``geval_stdev`` (None when no G-Eval
    leaf was hit by ≥2 survivors).
  * ``kappa_avg`` is the mean pairwise Cohen's κ over binary node verdicts
    that all surviving judges traversed (None when undefined). With only
    3 judges per axis you get 3 pairs, and κ over <5 binary verdicts per
    pair is statistical noise — we surface ``kappa_status`` so the UI can
    show "insufficient samples" rather than rendering a number that means
    nothing.
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

# Minimum number of binary-verdict observations per judge pair to report
# a Cohen's κ value. Below this we report kappa_status="insufficient_samples"
# and kappa_avg=None — Cohen's κ on n<5 is dominated by sampling noise.
_KAPPA_MIN_SAMPLES = 5


@dataclass
class AxisPanelResult:
    aggregated_score: float | None
    verdict_path_majority: list[str]
    judges: list[JudgeAxisResult]
    binary_unanimous: bool
    geval_stdev: float | None
    kappa_avg: float | None
    kappa_status: str = "ok"           # ok | insufficient_samples | undefined
    panel_status: str = "success"      # success | partial | all_failed
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregated_score": self.aggregated_score,
            "verdict_path_majority": self.verdict_path_majority,
            "judges": [j.to_dict() for j in self.judges],
            "panel_status": self.panel_status,
            "agreement": {
                "binary_unanimous": self.binary_unanimous,
                "geval_stdev": self.geval_stdev,
                "kappa_avg": self.kappa_avg,
                "kappa_status": self.kappa_status,
            },
            **self.extra,
        }


def _verdict_key_of(path: list[str]) -> str:
    """The terminal segment of a verdict path uniquely identifies the leaf reached."""
    return path[-1] if path else ""


def _binary_node_verdicts(judges: list[JudgeAxisResult]) -> dict[str, list[bool]]:
    """For each binary node visited by ALL surviving judges, collect their booleans.

    Defensive: the executor normalises verdicts to ``bool`` before storing
    them, but coerce here too so the κ rollup doesn't silently drop nodes
    if the executor ever returns ``"true"``/``"false"`` / ``1``/``0``.
    """
    def _as_bool(v: object) -> bool | None:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v != 0
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "1", "yes", "t"):
                return True
            if s in ("false", "0", "no", "f"):
                return False
        return None

    by_node: dict[str, list[bool]] = {}
    if not judges:
        return by_node
    node_sets: list[dict[str, bool]] = []
    for j in judges:
        per_judge: dict[str, bool] = {}
        for n in j.node_outputs:
            b = _as_bool(n.verdict)
            if b is not None:
                per_judge[n.node] = b
        node_sets.append(per_judge)
    common = set(node_sets[0].keys())
    for s in node_sets[1:]:
        common &= set(s.keys())
    for node in common:
        by_node[node] = [s[node] for s in node_sets]
    return by_node


def _classify_panel_status(judges: list[JudgeAxisResult]) -> str:
    """Three-state panel health summary based on per-judge status."""
    n_ok = sum(1 for j in judges if j.status == "success" and j.score is not None)
    if n_ok == 0:
        return "all_failed"
    if n_ok < len(judges):
        return "partial"
    return "success"


def _aggregate(judges: list[JudgeAxisResult]) -> AxisPanelResult:
    panel_status = _classify_panel_status(judges)
    survivors = [j for j in judges if j.status == "success" and j.score is not None]

    if not survivors:
        # All judges failed — aggregated_score is None and binary_unanimous is
        # meaningless. Consumers MUST check panel_status to distinguish
        # "everyone failed" from "everyone disagreed".
        return AxisPanelResult(
            aggregated_score=None,
            verdict_path_majority=[],
            judges=judges,
            binary_unanimous=False,
            geval_stdev=None,
            kappa_avg=None,
            kappa_status="undefined",
            panel_status=panel_status,
        )

    # Path-unanimity check (only across survivors — failed judges have no path)
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
    if not by_node:
        kappa_avg, kappa_status = None, "undefined"
    elif len(by_node) < _KAPPA_MIN_SAMPLES:
        # Per-pair observation count == len(by_node); below threshold κ is noise.
        kappa_avg, kappa_status = None, "insufficient_samples"
    else:
        node_names = list(by_node.keys())
        per_judge = [[by_node[n][i] for n in node_names] for i in range(len(survivors))]
        k = mean_pairwise_kappa(per_judge)
        if k is None or math.isnan(k):
            kappa_avg, kappa_status = None, "undefined"
        else:
            kappa_avg, kappa_status = k, "ok"

    return AxisPanelResult(
        aggregated_score=aggregated,
        verdict_path_majority=path,
        judges=judges,
        binary_unanimous=unanimous,
        geval_stdev=geval_stdev,
        kappa_avg=kappa_avg,
        kappa_status=kappa_status,
        panel_status=panel_status,
    )


async def score_axis_panel(
    dag: AxisDAG,
    case: dict,
    wrapper: LLMWrapper,
    judges: list[JudgeSpec],
    *,
    geval_cache: dict[tuple[str, str, str], list[str]] | None = None,
) -> AxisPanelResult:
    """Fan out the same DAG to every judge in parallel; aggregate."""
    tasks = [
        run_axis(dag, get_model(prov, mid), case, wrapper, geval_cache)
        for prov, mid in judges
    ]
    judge_results = await asyncio.gather(*tasks)
    return _aggregate(list(judge_results))
