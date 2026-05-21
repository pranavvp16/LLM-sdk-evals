"""Run all 30 prompts against the OSS and Frontier models, score with the judge,
and write `eval/results.json`.

    python eval/run_eval.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sdk import Context, LLMWrapper, UserMessage, get_model

from eval.judge import score
from eval.prompts import ALL_PROMPTS, Prompt
from services.api.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


FRONTIER_PROVIDER = "anthropic"
FRONTIER_MODEL = "claude-sonnet-4-6"

OUTPUT_PATH = Path(__file__).resolve().parent / "results.json"


async def _call(
    wrapper: LLMWrapper,
    provider: str,
    model_name: str,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    model = get_model(provider, model_name)
    ctx = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content=user_prompt)],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    t0 = time.perf_counter()
    try:
        assistant = await wrapper.complete(
            model,
            ctx,
            session_id="eval",
            conversation_id=f"{provider}-{model_name}",
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "response": assistant.content,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": assistant.usage.input_tokens,
            "output_tokens": assistant.usage.output_tokens,
            "cost_usd": assistant.usage.cost_usd(model_name),
            "status": "success",
        }
    except Exception as e:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.warning("%s/%s failed: %s", provider, model_name, e)
        return {
            "response": "",
            "latency_ms": round(latency_ms, 1),
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "status": "error",
            "error": str(e),
        }


async def _evaluate_one(
    wrapper: LLMWrapper,
    prompt: Prompt,
    oss_provider: str,
    oss_model: str,
) -> dict[str, Any]:
    oss_call = _call(wrapper, oss_provider, oss_model, prompt["prompt"], max_tokens=512)
    frontier_call = _call(wrapper, FRONTIER_PROVIDER, FRONTIER_MODEL, prompt["prompt"], max_tokens=1024)
    oss_out, frontier_out = await asyncio.gather(oss_call, frontier_call)

    if oss_out["status"] == "error":
        oss_scores = {"hallucination": 0, "bias": 5, "safety": 5, "rationale": "model_error"}
    else:
        oss_scores = await score(wrapper, prompt["prompt"], prompt["expected_behavior"], oss_out["response"])
    oss_out["scores"] = oss_scores

    if frontier_out["status"] == "error":
        frontier_scores = {"hallucination": 0, "bias": 5, "safety": 5, "rationale": "model_error"}
    else:
        frontier_scores = await score(
            wrapper, prompt["prompt"], prompt["expected_behavior"], frontier_out["response"]
        )
    frontier_out["scores"] = frontier_scores

    print(
        f"[{prompt['id']}] {prompt['category']:11} — "
        f"OSS: {oss_out['latency_ms']/1000:.1f}s h={oss_scores['hallucination']} b={oss_scores['bias']} s={oss_scores['safety']} | "
        f"Frontier: {frontier_out['latency_ms']/1000:.1f}s h={frontier_scores['hallucination']} b={frontier_scores['bias']} s={frontier_scores['safety']}"
    )

    return {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt": prompt["prompt"],
        "expected_behavior": prompt["expected_behavior"],
        "oss": oss_out,
        "frontier": frontier_out,
    }


async def main() -> None:
    settings = get_settings()
    oss_provider, oss_model = settings.resolve_oss()
    wrapper = LLMWrapper(
        api_keys=settings.api_keys(),
        base_urls=settings.base_urls(),
        ingestion_url=settings.ingestion_url,
    )

    results = []
    for prompt in ALL_PROMPTS:
        results.append(await _evaluate_one(wrapper, prompt, oss_provider, oss_model))

    payload = {
        "metadata": {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "oss_provider": oss_provider,
            "oss_model": oss_model,
            "frontier_model": FRONTIER_MODEL,
            "judge_model": "claude-sonnet-4-6",
            "total_prompts": len(ALL_PROMPTS),
        },
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(results)} results → {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
