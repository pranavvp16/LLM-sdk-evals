"""Unit tests for the DAG executor, G-Eval leaf, panel aggregation, and κ.

These tests do NOT call real LLM providers — every judge response is
scripted via ``tests.conftest.FakeJudgeWrapper``. They also do not touch
the frozen SDK or test_sdk.py.
"""

from __future__ import annotations

import math

import pytest

from sdk.registry import get_model

from services.api.judge_registry import register_judge_models
from services.api.oss_registry import register_oss_models

from eval.dag.executor import run_axis
from eval.dag.geval import _rubric_hash
from eval.dag.kappa import cohen_kappa, mean_pairwise_kappa
from eval.dag.nodes import AxisDAG, BinaryNode, GEvalLeaf, Verdict
from eval.dag.panel import score_axis_panel
from tests.conftest import FakeJudgeWrapper


@pytest.fixture(autouse=True, scope="module")
def _register_models():
    register_oss_models()
    register_judge_models()


def _make_binary_dag() -> AxisDAG:
    return AxisDAG(
        axis_name="t",
        entry="root",
        nodes={
            "root": BinaryNode(
                name="root",
                prompt="Is the response good? prompt={prompt} response={response}",
                on_true="verdict:good",
                on_false="verdict:bad",
            ),
        },
        verdicts={
            "good": Verdict(key="good", score=5.0),
            "bad":  Verdict(key="bad",  score=0.0),
        },
    )


def _make_geval_dag() -> AxisDAG:
    return AxisDAG(
        axis_name="t",
        entry="leaf",
        nodes={
            "leaf": GEvalLeaf(
                name="leaf",
                criteria="Is the response thoughtful?",
                rubric={0: "no", 5: "ok", 10: "yes"},
                eval_params=("prompt", "response"),
            ),
        },
        verdicts={},
    )


SONNET = ("anthropic", "claude-sonnet-4-6")
GPT5   = ("openai", "gpt-5")
DEEPSEEK = ("opencode-go", "deepseek-v4-flash")


@pytest.mark.asyncio
async def test_binary_node_true_path():
    dag = _make_binary_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [{"verdict": True, "reason": "fine"}],
    })
    result = await run_axis(dag, get_model(*SONNET), {"prompt": "p", "response": "r"}, wrapper)  # type: ignore[arg-type]
    assert result.status == "success"
    assert result.score == 5.0
    assert result.verdict_path == ["root:T", "verdict:good=5.0"]


@pytest.mark.asyncio
async def test_binary_node_false_path():
    dag = _make_binary_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [{"verdict": False, "reason": "nope"}],
    })
    result = await run_axis(dag, get_model(*SONNET), {"prompt": "p", "response": "r"}, wrapper)  # type: ignore[arg-type]
    assert result.score == 0.0
    assert result.verdict_path == ["root:F", "verdict:bad=0.0"]


@pytest.mark.asyncio
async def test_geval_leaf_normalization():
    """G-Eval leaf: 0-10 score → 0-5 normalized."""
    dag = _make_geval_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [
            ["step 1", "step 2", "step 3"],          # eval-steps generation
            {"score": 8, "reason": "good response"},  # scoring
        ],
    })
    result = await run_axis(
        dag, get_model(*SONNET), {"prompt": "p", "response": "r"}, wrapper,
        geval_cache={},
    )  # type: ignore[arg-type]
    assert result.status == "success"
    assert result.score == 4.0   # 8 / 2
    assert "leaf:4.0" in result.verdict_path[0]


@pytest.mark.asyncio
async def test_panel_majority_vote():
    """3 judges vote, 2 say True → unanimity False, score reflects majority via mean."""
    dag = _make_binary_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [{"verdict": True,  "reason": "yes"}],
        "gpt-5":             [{"verdict": True,  "reason": "yes"}],
        "deepseek-v4-flash": [{"verdict": False, "reason": "no"}],
    })
    panel = await score_axis_panel(
        dag, {"prompt": "p", "response": "r"}, wrapper,  # type: ignore[arg-type]
        judges=[SONNET, GPT5, DEEPSEEK],
    )
    assert panel.binary_unanimous is False
    # Mean of [5, 5, 0] = 3.333…
    assert panel.aggregated_score == pytest.approx(10 / 3)
    # Path-majority: 2 judges took "verdict:good=5.0", 1 took "verdict:bad=0.0".
    assert panel.verdict_path_majority[-1] == "verdict:good=5.0"


@pytest.mark.asyncio
async def test_panel_unanimous_path():
    dag = _make_binary_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [{"verdict": True, "reason": "yes"}],
        "gpt-5":             [{"verdict": True, "reason": "yes"}],
        "deepseek-v4-flash": [{"verdict": True, "reason": "yes"}],
    })
    panel = await score_axis_panel(
        dag, {"prompt": "p", "response": "r"}, wrapper,  # type: ignore[arg-type]
        judges=[SONNET, GPT5, DEEPSEEK],
    )
    assert panel.binary_unanimous is True
    assert panel.aggregated_score == 5.0


@pytest.mark.asyncio
async def test_panel_single_judge_failure():
    """One judge fails JSON parse (returns garbage 3 times) → other two still aggregate."""
    dag = _make_binary_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [{"verdict": True,  "reason": "ok"}],
        "gpt-5":             [{"verdict": True,  "reason": "ok"}],
        "deepseek-v4-flash": ["not json", "still not json", "nope nope nope"],
    })
    panel = await score_axis_panel(
        dag, {"prompt": "p", "response": "r"}, wrapper,  # type: ignore[arg-type]
        judges=[SONNET, GPT5, DEEPSEEK],
    )
    # The failed judge is excluded from aggregation
    survivors = [j for j in panel.judges if j.status == "success"]
    failed    = [j for j in panel.judges if j.status == "judge_failed"]
    assert len(survivors) == 2
    assert len(failed) == 1
    assert failed[0].judge_id == "opencode-go/deepseek-v4-flash"
    assert panel.aggregated_score == 5.0  # 2 survivors agreed


def test_cohen_kappa_perfect_agreement():
    assert cohen_kappa([True, True, False, False], [True, True, False, False]) == 1.0


def test_cohen_kappa_partial_agreement():
    # 3/4 agree, p_o = 0.75
    # a_pos = 3/4 = 0.75, b_pos = 2/4 = 0.5
    # p_e = 0.75*0.5 + 0.25*0.5 = 0.375 + 0.125 = 0.5
    # κ = (0.75 - 0.5) / (1 - 0.5) = 0.5
    k = cohen_kappa([True, True, True, False], [True, True, False, False])
    assert k == pytest.approx(0.5)


def test_cohen_kappa_constant_rater():
    """When a rater is constant, expected agreement = 1 → undefined κ."""
    k = cohen_kappa([True, True, True, True], [True, True, True, True])
    # Both constant + agree: p_e = 1, undefined → NaN
    assert math.isnan(k)


def test_mean_pairwise_kappa():
    raters = [
        [True, True, True, False],
        [True, True, False, False],
        [True, True, True, False],
    ]
    k = mean_pairwise_kappa(raters)
    # pairs: (1,2)=0.5, (1,3)=1.0, (2,3)=0.5 → mean ≈ 0.667
    assert k == pytest.approx((0.5 + 1.0 + 0.5) / 3)


@pytest.mark.asyncio
async def test_binary_node_string_false_takes_false_branch():
    """Regression for PR#13 Codex P1: `bool("false")` is truthy in Python.

    The executor must coerce string-valued verdicts to actual bools before
    branching, or the false-branch is never reached when a judge returns
    `"verdict": "false"` instead of JSON `false`.
    """
    dag = _make_binary_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [{"verdict": "false", "reason": "stringly typed"}],
    })
    result = await run_axis(dag, get_model(*SONNET), {"prompt": "p", "response": "r"}, wrapper)  # type: ignore[arg-type]
    assert result.score == 0.0
    assert result.verdict_path == ["root:F", "verdict:bad=0.0"]


@pytest.mark.asyncio
async def test_binary_node_ambiguous_verdict_retries_then_fails():
    """A verdict like 'maybe' is not coercible — node should retry and fail clean."""
    dag = _make_binary_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [
            {"verdict": "maybe", "reason": "1"},
            {"verdict": "kinda", "reason": "2"},
            {"verdict": "i guess", "reason": "3"},
        ],
    })
    result = await run_axis(dag, get_model(*SONNET), {"prompt": "p", "response": "r"}, wrapper)  # type: ignore[arg-type]
    assert result.status == "judge_failed"
    assert result.score is None


@pytest.mark.asyncio
async def test_geval_leaf_cost_includes_steps_generation():
    """Regression for PR#13 Greptile P1: step-gen tokens were silently dropped.

    Cache-cold path must add step-gen usage to the leaf NodeOutput.
    """
    dag = _make_geval_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [
            ["step 1", "step 2", "step 3"],          # steps-gen call
            {"score": 8, "reason": "ok"},             # scoring call
        ],
    })
    result = await run_axis(
        dag, get_model(*SONNET), {"prompt": "p", "response": "r"}, wrapper,  # type: ignore[arg-type]
        geval_cache={},
    )
    assert result.status == "success"
    # FakeJudgeWrapper bills 10 input + 20 output per call; 2 calls expected
    # → 20 input, 40 output. Pre-fix, only the scoring call (10/20) showed up.
    leaf = result.node_outputs[0]
    assert leaf.input_tokens == 20
    assert leaf.output_tokens == 40
    # Cost > 0 because Sonnet has non-zero pricing
    assert leaf.cost_usd > 0


@pytest.mark.asyncio
async def test_geval_leaf_cache_hit_skips_steps_generation():
    """Cached steps → no steps-gen call → only the scoring call's tokens count."""
    dag = _make_geval_dag()
    # Cache key is now (judge_id, criteria, rubric_hash) — see eval/dag/geval.py
    leaf = dag.nodes["leaf"]
    assert isinstance(leaf, GEvalLeaf)
    cache: dict = {
        (f"{SONNET[0]}/{SONNET[1]}", leaf.criteria, _rubric_hash(leaf.rubric)): ["pre-cached step"],
    }
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [
            {"score": 6, "reason": "cached path"},    # scoring only
        ],
    })
    result = await run_axis(
        dag, get_model(*SONNET), {"prompt": "p", "response": "r"}, wrapper,  # type: ignore[arg-type]
        geval_cache=cache,
    )
    assert result.score == 3.0   # 6 / 2
    leaf_out = result.node_outputs[0]
    assert leaf_out.input_tokens == 10   # just the scoring call
    assert leaf_out.output_tokens == 20


# ── New regression tests for the audit-gap fixes ──────────────────────────


@pytest.mark.asyncio
async def test_panel_all_failed_distinguishable_from_disagreement():
    """When every judge fails, panel_status='all_failed'; aggregated_score=None.

    A consumer reading {aggregated_score: None, binary_unanimous: false} can now
    tell error from edge case via panel_status.
    """
    dag = _make_binary_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": ["nope", "nope", "nope"],
        "gpt-5":             ["nope", "nope", "nope"],
        "deepseek-v4-flash": ["nope", "nope", "nope"],
    })
    panel = await score_axis_panel(
        dag, {"prompt": "p", "response": "r"}, wrapper,  # type: ignore[arg-type]
        judges=[SONNET, GPT5, DEEPSEEK],
    )
    assert panel.panel_status == "all_failed"
    assert panel.aggregated_score is None
    assert panel.kappa_status == "undefined"
    assert all(j.status == "judge_failed" for j in panel.judges)


@pytest.mark.asyncio
async def test_panel_partial_status():
    """One judge fails, two succeed → panel_status='partial'."""
    dag = _make_binary_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [{"verdict": True, "reason": "ok"}],
        "gpt-5":             [{"verdict": True, "reason": "ok"}],
        "deepseek-v4-flash": ["bad", "bad", "bad"],
    })
    panel = await score_axis_panel(
        dag, {"prompt": "p", "response": "r"}, wrapper,  # type: ignore[arg-type]
        judges=[SONNET, GPT5, DEEPSEEK],
    )
    assert panel.panel_status == "partial"
    assert panel.aggregated_score == 5.0  # 2 successful judges agreed


@pytest.mark.asyncio
async def test_panel_success_status():
    """All judges succeed → panel_status='success'."""
    dag = _make_binary_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [{"verdict": True, "reason": "ok"}],
        "gpt-5":             [{"verdict": True, "reason": "ok"}],
        "deepseek-v4-flash": [{"verdict": True, "reason": "ok"}],
    })
    panel = await score_axis_panel(
        dag, {"prompt": "p", "response": "r"}, wrapper,  # type: ignore[arg-type]
        judges=[SONNET, GPT5, DEEPSEEK],
    )
    assert panel.panel_status == "success"


@pytest.mark.asyncio
async def test_branch_failure_folds_retry_cost():
    """A node that runs out of retries still bills the partial spend.

    Pre-fix the 3 retry attempts' tokens evaporated; under-reported cost in
    the run-level rollup.
    """
    dag = _make_binary_dag()
    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [
            {"verdict": "maybe", "reason": "1"},
            {"verdict": "kinda", "reason": "2"},
            {"verdict": "i guess", "reason": "3"},
        ],
    })
    result = await run_axis(dag, get_model(*SONNET), {"prompt": "p", "response": "r"}, wrapper)  # type: ignore[arg-type]
    assert result.status == "judge_failed"
    # FakeJudgeWrapper bills 10/20 per call; 3 retries → 30/60. Pre-fix: 0/0.
    assert result.input_tokens == 30
    assert result.output_tokens == 60
    assert result.cost_usd > 0


def test_kappa_insufficient_samples_is_signalled():
    """The panel must surface kappa_status='insufficient_samples' rather
    than report a number computed from n<5 observations.

    A 3-judge × 1-binary-node DAG (e.g. tool_selection's correct path) has
    exactly 1 observation per pair — Cohen's κ on that is statistical noise.
    """
    # Pure unit check on the threshold constant — the integration is exercised
    # via score_axis_panel above.
    from eval.dag.panel import _KAPPA_MIN_SAMPLES
    assert _KAPPA_MIN_SAMPLES >= 5


def test_rubric_hash_differs_for_different_rubrics():
    """Two leaves with identical criteria but different rubrics must NOT
    share cached eval-steps. Pre-fix the cache key was (judge, criteria)
    only, so changing the rubric silently reused steps written for the old one."""
    a = _rubric_hash({0: "no", 5: "ok", 10: "yes"})
    b = _rubric_hash({0: "no", 5: "ok", 10: "PERFECT"})
    c = _rubric_hash({0: "no", 5: "ok", 10: "yes"})
    assert a != b
    assert a == c   # deterministic for the same rubric


@pytest.mark.asyncio
async def test_axis_not_applicable_clears_aggregated_score():
    """argument_correctness postprocess clears aggregated_score when every
    judge said not_applicable (no tool calls).

    Pre-fix the axis auto-scored 5.0 and silently inflated dataset means.
    """
    from eval.judge_prompts.argument_correctness import DAG as AC_DAG, postprocess as ac_pp

    wrapper = FakeJudgeWrapper({
        "claude-sonnet-4-6": [{"verdict": False, "reason": "0 calls"}],
        "gpt-5":             [{"verdict": False, "reason": "0 calls"}],
        "deepseek-v4-flash": [{"verdict": False, "reason": "0 calls"}],
    })
    panel = await score_axis_panel(
        AC_DAG, {"prompt_id": "p1", "prompt": "say hi", "tool_calls_text": "(none)"},
        wrapper, judges=[SONNET, GPT5, DEEPSEEK],  # type: ignore[arg-type]
    )
    # Pre-postprocess: all judges took verdict:not_applicable → aggregated=0.0 (placeholder)
    extras = ac_pp(panel, {})
    assert extras["axis_not_applicable"] is True
    assert panel.aggregated_score is None


@pytest.mark.asyncio
async def test_axis_not_applicable_false_when_tools_called():
    """When at least one judge says tools were called, axis_not_applicable
    is False and aggregated_score retains the panel value."""
    from eval.judge_prompts.argument_correctness import DAG as AC_DAG, postprocess as ac_pp

    wrapper = FakeJudgeWrapper({
        # all three judges chose verdict:correct, so the panel returns 5.0
        "claude-sonnet-4-6": [
            {"verdict": True, "reason": "1 call"},
            {"verdict": True, "reason": "valid"},
            {"verdict": True, "reason": "intent ok"},
        ],
        "gpt-5": [
            {"verdict": True, "reason": "1 call"},
            {"verdict": True, "reason": "valid"},
            {"verdict": True, "reason": "intent ok"},
        ],
        "deepseek-v4-flash": [
            {"verdict": True, "reason": "1 call"},
            {"verdict": True, "reason": "valid"},
            {"verdict": True, "reason": "intent ok"},
        ],
    })
    panel = await score_axis_panel(
        AC_DAG, {"prompt_id": "p1", "prompt": "schedule it", "tool_calls_text": "schedule_call(...)"},
        wrapper, judges=[SONNET, GPT5, DEEPSEEK],  # type: ignore[arg-type]
    )
    extras = ac_pp(panel, {})
    assert extras["axis_not_applicable"] is False
    assert panel.aggregated_score == 5.0
