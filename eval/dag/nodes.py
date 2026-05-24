"""DAG node dataclasses.

An ``AxisDAG`` is a small directed acyclic graph for scoring one axis (e.g.
hallucination). Edges are encoded inline on each node — Binary nodes have
``on_true`` / ``on_false`` targets, NonBinary nodes have a ``branches`` dict.
A target string is either another node name or ``"verdict:<key>"`` to terminate
at a Verdict in ``AxisDAG.verdicts``.

GEvalLeaf is a terminal node that runs a G-Eval-style rubric-anchored score
(0-10 normalized to 0-5) instead of branching further.

Verdicts produce a numeric score in [0, 5]; aggregation across judges
happens in ``eval.dag.panel``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class BinaryNode:
    name: str
    prompt: str
    on_true: str
    on_false: str


@dataclass(frozen=True)
class NonBinaryNode:
    name: str
    prompt: str
    branches: dict[str, str]


@dataclass(frozen=True)
class GEvalLeaf:
    name: str
    criteria: str
    rubric: dict[int, str]
    eval_params: tuple[str, ...] = field(default_factory=tuple)


Node = Union[BinaryNode, NonBinaryNode, GEvalLeaf]


@dataclass(frozen=True)
class Verdict:
    key: str
    score: float


@dataclass(frozen=True)
class AxisDAG:
    axis_name: str
    entry: str
    nodes: dict[str, Node]
    verdicts: dict[str, Verdict]

    def get(self, name: str) -> Node:
        return self.nodes[name]

    def verdict(self, key: str) -> Verdict:
        return self.verdicts[key]
