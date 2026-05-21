"""Ingestion entry point — POST /ingest/log.

Validates the payload with Pydantic and XADDs to a Redis stream. Returns
immediately (< 5ms). Heavy work (ClickHouse insert, retries, DLQ) happens
in the background worker — see `services/api/ingestion/worker.py`.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends

from services.api.config import Settings
from services.api.deps import redis_client, settings_dep
from services.api.ingestion.validator import InferenceLogIn

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])


@router.post("/ingest/log")
async def ingest_log(
    log: InferenceLogIn,
    redis=Depends(redis_client),
    settings: Settings = Depends(settings_dep),
) -> dict:
    payload = log.model_dump_json()
    try:
        await redis.xadd(
            settings.ingestion_stream,
            {"payload": payload},
            maxlen=1_000_000,
            approximate=True,
        )
    except Exception as e:  # noqa: BLE001
        # Observability must never crash the API. Log and move on.
        logger.warning("ingestion XADD failed: %s", e)
        return {"status": "dropped", "trace_id": log.trace_id}
    return {"status": "queued", "trace_id": log.trace_id}
