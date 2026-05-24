"""Run static + agent evaluations, score with the 3-judge DAG panel,
write ``eval/results.json``.

Usage:
    python eval/run_eval.py                # both static and agent
    python eval/run_eval.py --static       # only the 40 L2 prompts (4 axes per prompt)
    python eval/run_eval.py --agent        # only the 20 tool-use trajectories
    python eval/run_eval.py --all --limit 3  # smoke run, 3 per section

Both models (OSS + Frontier) receive the SAME structured system prompt
(``eval.heuristics.AGENT_SYSTEM_PROMPT``). Every response is scored on:
  - deterministic heuristics (persona signature, banned phrases, markdown), and
  - a 4-axis (L2) or 5-axis (L3) DAG ensemble across 3 judges from
    different model families (see JUDGE_PANEL below).

``run_all()`` is the public entrypoint shared between this CLI and the
FastAPI ``POST /eval/run`` background task.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from sdk import AssistantMessage, Context, LLMWrapper, UserMessage, get_model
from sdk.types import TextContent

from eval.agent_prompts import AGENT_PROMPTS
from eval.agent_runner import build_default_registry, run_trajectory
from eval.dag.panel import AxisPanelResult, JudgeSpec, score_axis_panel
from eval.heuristics import AGENT_SYSTEM_PROMPT, check_response, system_prompt_hash
from eval.judge_prompts import L2_AXES, L3_AXES
from eval.prompts import ALL_PROMPTS, Prompt
from services.api.config import Settings, get_settings
from services.api.guardrails import (
    GuardrailResult,
    dispatch_input_check,
    dispatch_output_check,
)
from services.api.judge_registry import register_judge_models
from services.api.oss_registry import register_oss_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


FRONTIER_PROVIDER = "anthropic"
FRONTIER_MODEL = "claude-sonnet-4-6"

# Three-judge DAG ensemble. One per model family so that no single judge can
# self-prefer when scoring its own family's responses.
JUDGE_PANEL: list[JudgeSpec] = [
    ("anthropic",   "claude-sonnet-4-6"),
    ("openai",      "gpt-5"),
    ("opencode-go", "deepseek-v4-flash"),
]

# Stock refusal text the eval surfaces when Llama Guard blocks. The judge
# still sees this text and scores it.
_GUARDRAIL_REFUSAL = "This request was blocked by guardrails."

OUTPUT_PATH = Path(__file__).resolve().parent / "results.json"

ProgressCallback = Optional[Callable[[str, dict], None]]


# ── Helpers ──────────────────────────────────────────────────────────────


def _extract_text(message: AssistantMessage) -> str:
    """Concatenate all TextContent blocks into a single string."""
    parts = []
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
    return "".join(parts)


def _failed_axis(axis_name: str) -> dict[str, Any]:
    """Sentinel axis result for rows where the model failed before scoring."""
    return {
        "aggregated_score": None,
        "verdict_path_majority": [],
        "judges": [],
        "agreement": {
            "binary_unanimous": False,
            "geval_stdev": None,
            "kappa_avg": None,
        },
    }


def _format_tool_calls_text(traj: dict[str, Any]) -> str:
    """Compact rendering of a trajectory's tool calls for the judge prompts."""
    calls = traj.get("tool_calls", [])
    if not calls:
        return "(none — model answered directly)"
    lines: list[str] = []
    for tc in calls:
        args = tc.get("arguments")
        result = tc.get("result")
        args_str = json.dumps(args, default=str) if not isinstance(args, str) else args
        result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
        if len(result_str) > 500:
            result_str = result_str[:500] + "…"
        err = " [ERROR]" if tc.get("is_error") else ""
        lines.append(
            f"  hop {tc.get('hop', '?')}: {tc.get('name', '?')}({args_str}){err}\n"
            f"    → {result_str}"
        )
    return "\n".join(lines)


async def _llamaguard_pre_signal(
    prompt_text: str, wrapper: LLMWrapper, settings: Settings
) -> str:
    """Return a one-line Llama Guard label for the user prompt.

    Used by the L2 safety DAG as a pre-signal. Falls back to "unknown" when
    Ollama / Llama Guard is not wired up so the judge can still proceed.
    """
    if not settings.ollama_base_url:
        return "unknown (llamaguard not configured for this run)"
    try:
        result: GuardrailResult = await dispatch_input_check(
            "llamaguard", text=prompt_text, wrapper=wrapper, settings=settings
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("llamaguard pre-signal failed: %s", e)
        return f"unknown ({e})"
    return "safe" if result.allowed else f"unsafe: {result.reason}"


async def _call_static(
    wrapper: LLMWrapper,
    provider: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    thinking: bool | int | None = None,
    *,
    guardrails: str = "off",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run a single prompt through one model.

    ``guardrails`` controls whether we wrap the call with input / output
    safety checks. ``"off"`` is the historical raw-model path; ``"regex"``
    applies the deterministic filters; ``"llamaguard"`` calls Llama Guard
    on Ollama for both directions.
    """
    model = get_model(provider, model_name)
    ctx = Context(
        system_prompt=system_prompt,
        messages=[UserMessage(content=user_prompt)],
        max_tokens=max_tokens,
        temperature=0.2,
        thinking=thinking,
    )
    t0 = time.perf_counter()

    if guardrails != "off":
        in_check: GuardrailResult = await dispatch_input_check(
            guardrails, text=user_prompt, wrapper=wrapper, settings=settings
        )
        if not in_check.allowed:
            return {
                "response": _GUARDRAIL_REFUSAL,
                "thinking": "",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "status": "success",
                "guardrail_block": {"stage": "input", "reason": in_check.reason},
            }

    try:
        assistant = await wrapper.complete(
            model,
            ctx,
            session_id="eval",
            conversation_id=f"static-{provider}-{model_name}-{guardrails}",
        )
    except Exception as e:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.warning("%s/%s static call failed: %s", provider, model_name, e)
        return {
            "response": "",
            "thinking": "",
            "latency_ms": round(latency_ms, 1),
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "status": "error",
            "error": str(e),
        }
    text = _extract_text(assistant)
    out: dict[str, Any] = {
        "response": text,
        "thinking": assistant.thinking or "",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "input_tokens": assistant.usage.input_tokens,
        "output_tokens": assistant.usage.output_tokens,
        "cost_usd": assistant.usage.cost_usd(model),
        "status": "success",
    }

    if guardrails != "off" and text:
        out_check = await dispatch_output_check(
            guardrails,
            user_text=user_prompt,
            assistant_text=text,
            wrapper=wrapper,
            settings=settings,
        )
        if not out_check.allowed:
            out["response"] = _GUARDRAIL_REFUSAL
            out["guardrail_block"] = {
                "stage": "output",
                "reason": out_check.reason,
            }
            out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    return out


# ── Scoring (3-judge DAG ensemble) ───────────────────────────────────────


GEvalCache = dict[tuple[str, str], list[str]]


async def _score_axis(
    axis_name: str,
    dag,
    postprocess,
    case: dict,
    wrapper: LLMWrapper,
    geval_cache: GEvalCache,
) -> tuple[str, dict[str, Any]]:
    panel: AxisPanelResult = await score_axis_panel(
        dag, case, wrapper, JUDGE_PANEL, geval_cache=geval_cache,
    )
    panel.extra.update(postprocess(panel, case))
    return axis_name, panel.to_dict()


async def _score_side_l2(
    prompt: Prompt,
    side: dict[str, Any],
    wrapper: LLMWrapper,
    settings: Settings,
    geval_cache: GEvalCache,
    pre_signal: str,
) -> dict[str, Any]:
    if side.get("status") == "error":
        return {axis: _failed_axis(axis) for axis, _, _ in L2_AXES}
    case = {
        "prompt": prompt["prompt"],
        "expected": prompt["expected_behavior"],
        "response": side.get("response") or "(empty)",
        "category": prompt["category"],
        "llamaguard_pre_signal": pre_signal,
    }
    tasks = [_score_axis(name, dag, pp, case, wrapper, geval_cache) for name, dag, pp in L2_AXES]
    results = await asyncio.gather(*tasks)
    return dict(results)


async def _score_side_l3(
    prompt: dict,
    traj_dict: dict[str, Any],
    wrapper: LLMWrapper,
    geval_cache: GEvalCache,
) -> dict[str, Any]:
    if traj_dict.get("status") == "model_error":
        return {axis: _failed_axis(axis) for axis, _, _ in L3_AXES}
    case = {
        "prompt": prompt["prompt"],
        "expected": prompt["expected_behavior"],
        "response": traj_dict.get("final_text") or "(empty)",
        "tool_calls_text": _format_tool_calls_text(traj_dict),
        "expected_tools": ", ".join(prompt.get("expected_tools") or []) or "(none)",
        "category": prompt["category"],
    }
    tasks = [_score_axis(name, dag, pp, case, wrapper, geval_cache) for name, dag, pp in L3_AXES]
    results = await asyncio.gather(*tasks)
    return dict(results)


# ── Static prompts (hallucination / bias / safety / role_violation) ──────


async def _evaluate_static(
    wrapper: LLMWrapper,
    prompt: Prompt,
    oss_provider: str,
    oss_model: str,
    system_prompt: str,
    *,
    settings: Settings,
    run_guardrails_column: bool,
    geval_cache: GEvalCache,
) -> dict[str, Any]:
    oss_task = _call_static(
        wrapper, oss_provider, oss_model, system_prompt, prompt["prompt"], 512, thinking=None
    )
    frontier_task = _call_static(
        wrapper,
        FRONTIER_PROVIDER,
        FRONTIER_MODEL,
        system_prompt,
        prompt["prompt"],
        1024,
        thinking=True,
    )
    if run_guardrails_column:
        guarded_task = _call_static(
            wrapper,
            oss_provider,
            oss_model,
            system_prompt,
            prompt["prompt"],
            512,
            thinking=None,
            guardrails="llamaguard",
            settings=settings,
        )
        oss_out, frontier_out, guarded_out = await asyncio.gather(
            oss_task, frontier_task, guarded_task
        )
    else:
        oss_out, frontier_out = await asyncio.gather(oss_task, frontier_task)
        guarded_out = None

    sides = [oss_out, frontier_out] + ([guarded_out] if guarded_out else [])
    for side in sides:
        side["heuristic"] = check_response(side["response"]).to_dict()

    pre_signal = await _llamaguard_pre_signal(prompt["prompt"], wrapper, settings)

    oss_out["scores"] = await _score_side_l2(
        prompt, oss_out, wrapper, settings, geval_cache, pre_signal
    )
    frontier_out["scores"] = await _score_side_l2(
        prompt, frontier_out, wrapper, settings, geval_cache, pre_signal
    )
    if guarded_out is not None:
        guarded_out["scores"] = await _score_side_l2(
            prompt, guarded_out, wrapper, settings, geval_cache, pre_signal
        )

    row: dict[str, Any] = {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt": prompt["prompt"],
        "expected_behavior": prompt["expected_behavior"],
        "kind": "static",
        "oss": oss_out,
        "frontier": frontier_out,
    }
    if guarded_out is not None:
        row["oss_guarded"] = guarded_out
    return row


def _log_l2_row(row: dict[str, Any], run_guardrails_column: bool) -> None:
    def s(side: str, axis: str) -> str:
        v = row.get(side, {}).get("scores", {}).get(axis, {}).get("aggregated_score")
        return f"{v:.1f}" if isinstance(v, (int, float)) else "—"

    fr = "F: " + " ".join(f"{a[0]}={s('frontier', a)}" for a in ("hallucination", "bias", "safety", "role_violation"))
    os_ = "O: " + " ".join(f"{a[0]}={s('oss', a)}" for a in ("hallucination", "bias", "safety", "role_violation"))
    if run_guardrails_column:
        g = "G: " + " ".join(f"{a[0]}={s('oss_guarded', a)}" for a in ("hallucination", "bias", "safety", "role_violation"))
        logger.info("[%s] %s — %s | %s | %s", row["prompt_id"], row["category"], os_, g, fr)
    else:
        logger.info("[%s] %s — %s | %s", row["prompt_id"], row["category"], os_, fr)


async def run_static(
    wrapper: LLMWrapper,
    oss_provider: str,
    oss_model: str,
    system_prompt: str,
    limit: int | None = None,
    on_progress: ProgressCallback = None,
    *,
    settings: Settings,
    run_guardrails_column: bool = False,
    geval_cache: GEvalCache | None = None,
) -> list[dict]:
    prompts = ALL_PROMPTS[:limit] if limit else ALL_PROMPTS
    rows: list[dict] = []
    cache = geval_cache if geval_cache is not None else {}
    for p in prompts:
        row = await _evaluate_static(
            wrapper, p, oss_provider, oss_model, system_prompt,
            settings=settings, run_guardrails_column=run_guardrails_column,
            geval_cache=cache,
        )
        rows.append(row)
        _log_l2_row(row, run_guardrails_column)
        if on_progress:
            on_progress(row["prompt_id"], row)
    return rows


# ── Agent prompts (tool-use trajectories) ────────────────────────────────


async def _run_guarded_trajectory(
    wrapper: LLMWrapper,
    prompt: dict,
    oss_provider: str,
    oss_model: str,
    system_prompt: str,
    tools,
    settings: Settings,
) -> dict[str, Any]:
    """Llama-Guard-wrapped agent trajectory.

    Input-side: if Llama Guard blocks the user message, we return a refusal
    trajectory without running any tools.

    Output-side: if Llama Guard blocks the final assistant text after the
    trajectory finishes, we replace it with the standard refusal string.
    """
    t0 = time.perf_counter()
    in_check = await dispatch_input_check(
        "llamaguard", text=prompt["prompt"], wrapper=wrapper, settings=settings
    )
    if not in_check.allowed:
        return {
            "prompt_id": prompt["id"],
            "provider": oss_provider,
            "model": oss_model,
            "user_prompt": prompt["prompt"],
            "tool_calls": [],
            "final_text": _GUARDRAIL_REFUSAL,
            "thinking": "",
            "hops_used": 0,
            "hit_max_hops": False,
            "status": "success",
            "error": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "guardrail_block": {"stage": "input", "reason": in_check.reason},
        }

    traj = await run_trajectory(
        wrapper,
        oss_provider,
        oss_model,
        tools,
        prompt["prompt"],
        prompt["id"],
        system_prompt,
    )
    d = traj.to_dict()

    if d.get("final_text"):
        out_check = await dispatch_output_check(
            "llamaguard",
            user_text=prompt["prompt"],
            assistant_text=d["final_text"],
            wrapper=wrapper,
            settings=settings,
        )
        if not out_check.allowed:
            d["final_text"] = _GUARDRAIL_REFUSAL
            d["guardrail_block"] = {
                "stage": "output",
                "reason": out_check.reason,
            }
    return d


async def _evaluate_agent(
    wrapper: LLMWrapper,
    prompt: dict,
    oss_provider: str,
    oss_model: str,
    system_prompt: str,
    tools,
    *,
    settings: Settings,
    run_guardrails_column: bool,
    geval_cache: GEvalCache,
) -> dict[str, Any]:
    oss_task = run_trajectory(
        wrapper,
        oss_provider,
        oss_model,
        tools,
        prompt["prompt"],
        prompt["id"],
        system_prompt,
    )
    frontier_task = run_trajectory(
        wrapper,
        FRONTIER_PROVIDER,
        FRONTIER_MODEL,
        tools,
        prompt["prompt"],
        prompt["id"],
        system_prompt,
    )
    if run_guardrails_column:
        guarded_task = _run_guarded_trajectory(
            wrapper, prompt, oss_provider, oss_model, system_prompt, tools, settings
        )
        oss_traj, frontier_traj, guarded_dict = await asyncio.gather(
            oss_task, frontier_task, guarded_task
        )
    else:
        oss_traj, frontier_traj = await asyncio.gather(oss_task, frontier_task)
        guarded_dict = None
    oss_dict = oss_traj.to_dict()
    frontier_dict = frontier_traj.to_dict()

    oss_dict["heuristic"] = check_response(oss_dict.get("final_text", "")).to_dict()
    frontier_dict["heuristic"] = check_response(frontier_dict.get("final_text", "")).to_dict()
    if guarded_dict is not None:
        guarded_dict["heuristic"] = check_response(
            guarded_dict.get("final_text", "")
        ).to_dict()

    oss_dict["scores"] = await _score_side_l3(prompt, oss_dict, wrapper, geval_cache)
    frontier_dict["scores"] = await _score_side_l3(prompt, frontier_dict, wrapper, geval_cache)
    if guarded_dict is not None:
        guarded_dict["scores"] = await _score_side_l3(prompt, guarded_dict, wrapper, geval_cache)

    row: dict[str, Any] = {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt": prompt["prompt"],
        "expected_behavior": prompt["expected_behavior"],
        "expected_tools": prompt.get("expected_tools"),
        "kind": "agent",
        "oss": oss_dict,
        "frontier": frontier_dict,
    }
    if guarded_dict is not None:
        row["oss_guarded"] = guarded_dict
    return row


def _log_l3_row(row: dict[str, Any], run_guardrails_column: bool) -> None:
    def s(side: str, axis: str) -> str:
        v = row.get(side, {}).get("scores", {}).get(axis, {}).get("aggregated_score")
        return f"{v:.1f}" if isinstance(v, (int, float)) else "—"

    axes_short = ("tool_selection", "argument_correctness", "task_completion", "output_grounding", "safety_with_tools")
    short = ["ts", "ac", "tc", "og", "sf"]
    fr = "F: " + " ".join(f"{sh}={s('frontier', a)}" for sh, a in zip(short, axes_short))
    os_ = "O: " + " ".join(f"{sh}={s('oss', a)}" for sh, a in zip(short, axes_short))
    if run_guardrails_column:
        g = "G: " + " ".join(f"{sh}={s('oss_guarded', a)}" for sh, a in zip(short, axes_short))
        logger.info("[%s] %s — %s | %s | %s", row["prompt_id"], row["category"], os_, g, fr)
    else:
        logger.info("[%s] %s — %s | %s", row["prompt_id"], row["category"], os_, fr)


async def run_agent(
    wrapper: LLMWrapper,
    oss_provider: str,
    oss_model: str,
    system_prompt: str,
    limit: int | None = None,
    on_progress: ProgressCallback = None,
    *,
    settings: Settings,
    run_guardrails_column: bool = False,
    geval_cache: GEvalCache | None = None,
) -> list[dict]:
    oss_model_def = get_model(oss_provider, oss_model)
    if not oss_model_def.supports_tools:
        raise ValueError(
            f"agent eval requires supports_tools=True; "
            f"{oss_provider}/{oss_model} has it disabled. "
            f"Set OSS_PROVIDER=opencode-go (or another tool-capable provider)."
        )

    tools = build_default_registry()
    prompts = AGENT_PROMPTS[:limit] if limit else AGENT_PROMPTS
    rows: list[dict] = []
    cache = geval_cache if geval_cache is not None else {}
    for p in prompts:
        row = await _evaluate_agent(
            wrapper, p, oss_provider, oss_model, system_prompt, tools,
            settings=settings, run_guardrails_column=run_guardrails_column,
            geval_cache=cache,
        )
        rows.append(row)
        _log_l3_row(row, run_guardrails_column)
        if on_progress:
            on_progress(row["prompt_id"], row)
    return rows


# ── Run-level rollups (judge cost, κ-by-axis) ────────────────────────────


def _rollup_judge_panel(static_rows: list[dict], agent_rows: list[dict]) -> dict[str, Any]:
    """Sum judge cost + latency across every panel call in the run."""
    total_cost = 0.0
    total_latency_ms = 0.0
    per_judge: dict[str, dict[str, float]] = {}
    for row in list(static_rows) + list(agent_rows):
        for side in ("oss", "frontier", "oss_guarded"):
            scores = row.get(side, {}).get("scores", {})
            if not isinstance(scores, dict):
                continue
            for axis_data in scores.values():
                if not isinstance(axis_data, dict):
                    continue
                for j in axis_data.get("judges", []):
                    cost = j.get("cost_usd", 0.0) or 0.0
                    lat = j.get("latency_ms", 0.0) or 0.0
                    total_cost += cost
                    total_latency_ms += lat
                    bucket = per_judge.setdefault(
                        j.get("judge_id", "?"), {"cost_usd": 0.0, "latency_ms": 0.0, "calls": 0}
                    )
                    bucket["cost_usd"] += cost
                    bucket["latency_ms"] += lat
                    bucket["calls"] += 1
    return {
        "judge_total_cost_usd": round(total_cost, 6),
        "judge_total_latency_ms": round(total_latency_ms, 1),
        "judge_per_judge": per_judge,
    }


# ── Public entrypoint shared between CLI and API ─────────────────────────


async def run_all(
    *,
    sections: list[str],
    limit: int | None = None,
    on_progress: ProgressCallback = None,
    guardrails_ablation: bool | None = None,
) -> dict:
    """Run the requested sections end-to-end and return the payload."""
    register_oss_models()
    register_judge_models()
    settings = get_settings()
    oss_provider, oss_model = settings.resolve_oss()
    wrapper = LLMWrapper(
        api_keys=settings.api_keys(),
        base_urls=settings.base_urls(),
        ingestion_url=settings.ingestion_url,
    )
    system_prompt = AGENT_SYSTEM_PROMPT

    if guardrails_ablation is None:
        run_guard = (oss_provider == "ollama") and bool(settings.ollama_base_url)
    else:
        run_guard = bool(guardrails_ablation)

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    # One G-Eval steps cache for the whole run — eliminates duplicate
    # CoT-step calls per (judge, criteria) across rows.
    geval_cache: GEvalCache = {}

    static_rows: list[dict] = []
    agent_rows: list[dict] = []
    if "static" in sections:
        static_rows = await run_static(
            wrapper, oss_provider, oss_model, system_prompt, limit, on_progress,
            settings=settings, run_guardrails_column=run_guard,
            geval_cache=geval_cache,
        )
    if "agent" in sections:
        agent_rows = await run_agent(
            wrapper, oss_provider, oss_model, system_prompt, limit, on_progress,
            settings=settings, run_guardrails_column=run_guard,
            geval_cache=geval_cache,
        )

    completed_at = datetime.now(timezone.utc).isoformat()
    panel_rollup = _rollup_judge_panel(static_rows, agent_rows)

    return {
        "metadata": {
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "sections": sections,
            "limit": limit,
            "oss_provider": oss_provider,
            "oss_model": oss_model,
            "frontier_model": FRONTIER_MODEL,
            "judge_panel": [f"{p}/{m}" for p, m in JUDGE_PANEL],
            "judge_model": "panel",                                 # legacy field kept for compatibility
            "static_prompts_total": len(static_rows),
            "agent_prompts_total": len(agent_rows),
            "system_prompt_hash": system_prompt_hash(),
            "guardrails_ablation": run_guard,
            "guard_model": settings.ollama_guard_model if run_guard else None,
            **panel_rollup,
        },
        "static_results": static_rows,
        "agent_results": agent_rows,
    }


# ── CLI ──────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the personal-assistant eval suite.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--static", action="store_true", help="only the static prompts")
    group.add_argument("--agent", action="store_true", help="only the agent prompts")
    group.add_argument("--all", action="store_true", help="both sections (default)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap prompts per section (smoke runs)",
    )
    guard_group = parser.add_mutually_exclusive_group()
    guard_group.add_argument(
        "--with-llamaguard",
        action="store_true",
        help="force the 3-column ablation even if OSS isn't Ollama",
    )
    guard_group.add_argument(
        "--no-llamaguard",
        action="store_true",
        help="skip the OSS+LlamaGuard column",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    if args.static:
        sections = ["static"]
    elif args.agent:
        sections = ["agent"]
    else:
        sections = ["static", "agent"]

    ablation: bool | None = None
    if args.with_llamaguard:
        ablation = True
    elif args.no_llamaguard:
        ablation = False

    payload = await run_all(
        sections=sections, limit=args.limit, guardrails_ablation=ablation,
    )
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    n_static = payload["metadata"]["static_prompts_total"]
    n_agent = payload["metadata"]["agent_prompts_total"]
    cols = "3 columns" if payload["metadata"].get("guardrails_ablation") else "2 columns"
    cost = payload["metadata"].get("judge_total_cost_usd", 0.0)
    print(
        f"\nWrote {n_static} static + {n_agent} agent rows ({cols}) → {OUTPUT_PATH}\n"
        f"Judge panel total cost: ${cost:.4f}"
    )


if __name__ == "__main__":
    asyncio.run(_main())
