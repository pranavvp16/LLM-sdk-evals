"""Run static + agent evaluations, score with the judges, write
``eval/results.json``.

Usage:
    python eval/run_eval.py                # both static and agent
    python eval/run_eval.py --static       # only the 30 hallucination/bias/safety
    python eval/run_eval.py --agent        # only the 20 tool-use trajectories
    python eval/run_eval.py --all --limit 3  # smoke run, 3 per section

Both models (OSS + Frontier) receive the SAME structured system prompt
(``eval.heuristics.AGENT_SYSTEM_PROMPT``). Every response is also scored on
deterministic heuristic compliance (persona, banned phrases, markdown
structure) alongside the LLM-judge scores.

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

from eval.agent_judge import score_agent
from eval.agent_prompts import AGENT_PROMPTS
from eval.agent_runner import build_default_registry, run_trajectory
from eval.heuristics import AGENT_SYSTEM_PROMPT, check_response, system_prompt_hash
from eval.judge import score
from eval.prompts import ALL_PROMPTS, Prompt
from services.api.config import get_settings
from services.api.oss_registry import register_oss_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


FRONTIER_PROVIDER = "anthropic"
FRONTIER_MODEL = "claude-sonnet-4-6"

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


async def _call_static(
    wrapper: LLMWrapper,
    provider: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    thinking: bool | int | None = None,
) -> dict[str, Any]:
    model = get_model(provider, model_name)
    ctx = Context(
        system_prompt=system_prompt,
        messages=[UserMessage(content=user_prompt)],
        max_tokens=max_tokens,
        temperature=0.2,
        thinking=thinking,
    )
    t0 = time.perf_counter()
    try:
        assistant = await wrapper.complete(
            model,
            ctx,
            session_id="eval",
            conversation_id=f"static-{provider}-{model_name}",
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = _extract_text(assistant)
        return {
            "response": text,
            "thinking": assistant.thinking or "",
            "latency_ms": round(latency_ms, 1),
            "input_tokens": assistant.usage.input_tokens,
            "output_tokens": assistant.usage.output_tokens,
            "cost_usd": assistant.usage.cost_usd(model),
            "status": "success",
        }
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


# ── Static prompts (hallucination / bias / safety) ───────────────────────


async def _evaluate_static(
    wrapper: LLMWrapper,
    prompt: Prompt,
    oss_provider: str,
    oss_model: str,
    system_prompt: str,
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
    oss_out, frontier_out = await asyncio.gather(oss_task, frontier_task)

    for side in (oss_out, frontier_out):
        side["heuristic"] = check_response(side["response"]).to_dict()

    if oss_out["status"] == "error":
        oss_scores = {
            "hallucination": -1, "bias": -1, "safety": -1, "rationale": "model_error"
        }
    else:
        oss_scores = await score(
            wrapper, prompt["prompt"], prompt["expected_behavior"], oss_out["response"]
        )
    oss_out["scores"] = oss_scores

    if frontier_out["status"] == "error":
        frontier_scores = {
            "hallucination": -1, "bias": -1, "safety": -1, "rationale": "model_error"
        }
    else:
        frontier_scores = await score(
            wrapper,
            prompt["prompt"],
            prompt["expected_behavior"],
            frontier_out["response"],
        )
    frontier_out["scores"] = frontier_scores

    return {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt": prompt["prompt"],
        "expected_behavior": prompt["expected_behavior"],
        "kind": "static",
        "oss": oss_out,
        "frontier": frontier_out,
    }


async def run_static(
    wrapper: LLMWrapper,
    oss_provider: str,
    oss_model: str,
    system_prompt: str,
    limit: int | None = None,
    on_progress: ProgressCallback = None,
) -> list[dict]:
    prompts = ALL_PROMPTS[:limit] if limit else ALL_PROMPTS
    rows: list[dict] = []
    for p in prompts:
        row = await _evaluate_static(wrapper, p, oss_provider, oss_model, system_prompt)
        rows.append(row)
        o = row["oss"]["scores"]
        f = row["frontier"]["scores"]
        logger.info(
            "[%s] %s — OSS h=%s b=%s s=%s | Frontier h=%s b=%s s=%s",
            row["prompt_id"], row["category"],
            o["hallucination"], o["bias"], o["safety"],
            f["hallucination"], f["bias"], f["safety"],
        )
        if on_progress:
            on_progress(row["prompt_id"], row)
    return rows


# ── Agent prompts (tool-use trajectories) ────────────────────────────────


async def _evaluate_agent(
    wrapper: LLMWrapper,
    prompt: dict,
    oss_provider: str,
    oss_model: str,
    system_prompt: str,
    tools,
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
    oss_traj, frontier_traj = await asyncio.gather(oss_task, frontier_task)
    oss_dict = oss_traj.to_dict()
    frontier_dict = frontier_traj.to_dict()

    oss_dict["heuristic"] = check_response(oss_dict.get("final_text", "")).to_dict()
    frontier_dict["heuristic"] = check_response(frontier_dict.get("final_text", "")).to_dict()

    expected_tools = prompt.get("expected_tools")

    if oss_dict.get("status") == "model_error":
        oss_scores = {
            "tool_selection": -1, "argument_correctness": -1, "task_completion": -1,
            "output_grounding": -1, "safety_with_tools": -1, "rationale": "model_error",
        }
    else:
        oss_scores = await score_agent(
            wrapper, prompt["prompt"], prompt["expected_behavior"], expected_tools, oss_dict
        )
    oss_dict["scores"] = oss_scores

    if frontier_dict.get("status") == "model_error":
        frontier_scores = {
            "tool_selection": -1, "argument_correctness": -1, "task_completion": -1,
            "output_grounding": -1, "safety_with_tools": -1, "rationale": "model_error",
        }
    else:
        frontier_scores = await score_agent(
            wrapper,
            prompt["prompt"],
            prompt["expected_behavior"],
            expected_tools,
            frontier_dict,
        )
    frontier_dict["scores"] = frontier_scores

    return {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt": prompt["prompt"],
        "expected_behavior": prompt["expected_behavior"],
        "expected_tools": expected_tools,
        "kind": "agent",
        "oss": oss_dict,
        "frontier": frontier_dict,
    }


async def run_agent(
    wrapper: LLMWrapper,
    oss_provider: str,
    oss_model: str,
    system_prompt: str,
    limit: int | None = None,
    on_progress: ProgressCallback = None,
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
    for p in prompts:
        row = await _evaluate_agent(
            wrapper, p, oss_provider, oss_model, system_prompt, tools
        )
        rows.append(row)
        o = row["oss"]["scores"]
        f = row["frontier"]["scores"]
        logger.info(
            "[%s] %s — OSS ts=%s ac=%s tc=%s og=%s sf=%s | Frontier ts=%s ac=%s tc=%s og=%s sf=%s",
            row["prompt_id"], row["category"],
            o["tool_selection"], o["argument_correctness"], o["task_completion"],
            o["output_grounding"], o["safety_with_tools"],
            f["tool_selection"], f["argument_correctness"], f["task_completion"],
            f["output_grounding"], f["safety_with_tools"],
        )
        if on_progress:
            on_progress(row["prompt_id"], row)
    return rows


# ── Public entrypoint shared between CLI and API ─────────────────────────


async def run_all(
    *,
    sections: list[str],
    limit: int | None = None,
    on_progress: ProgressCallback = None,
) -> dict:
    """Run the requested sections end-to-end and return the payload.

    ``sections`` is any non-empty subset of {"static", "agent"}. The caller
    is responsible for persisting the return value (e.g. to results.json).
    """
    register_oss_models()
    settings = get_settings()
    oss_provider, oss_model = settings.resolve_oss()
    wrapper = LLMWrapper(
        api_keys=settings.api_keys(),
        base_urls=settings.base_urls(),
        ingestion_url=settings.ingestion_url,
    )
    system_prompt = AGENT_SYSTEM_PROMPT

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    static_rows: list[dict] = []
    agent_rows: list[dict] = []
    if "static" in sections:
        static_rows = await run_static(
            wrapper, oss_provider, oss_model, system_prompt, limit, on_progress
        )
    if "agent" in sections:
        agent_rows = await run_agent(
            wrapper, oss_provider, oss_model, system_prompt, limit, on_progress
        )

    completed_at = datetime.now(timezone.utc).isoformat()

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
            "judge_model": "claude-sonnet-4-6",
            "static_prompts_total": len(static_rows),
            "agent_prompts_total": len(agent_rows),
            "system_prompt_hash": system_prompt_hash(),
        },
        "static_results": static_rows,
        "agent_results": agent_rows,
    }


# ── CLI ──────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the personal-assistant eval suite.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--static", action="store_true", help="only the 30 static prompts")
    group.add_argument("--agent", action="store_true", help="only the 20 agent prompts")
    group.add_argument("--all", action="store_true", help="both sections (default)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap prompts per section (smoke runs)",
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

    payload = await run_all(sections=sections, limit=args.limit)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    n_static = payload["metadata"]["static_prompts_total"]
    n_agent = payload["metadata"]["agent_prompts_total"]
    print(f"\nWrote {n_static} static + {n_agent} agent rows → {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(_main())
