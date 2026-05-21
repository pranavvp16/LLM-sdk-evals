"""Background ingestion worker.

Reads validated logs from a Redis stream, batches them, and flushes to
ClickHouse. Started as an asyncio task during FastAPI startup (see main.py).

Loop:
  1. XREADGROUP up to `batch_size` entries (block until `batch_flush_ms`).
  2. Validate + enrich each entry.
  3. Insert as a single ClickHouse batch.
  4. XACK on success.
  5. On failure, retry up to `max_retries` with exponential backoff, then push
     to the DLQ stream and ack the original to stop replay.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from services.api.config import Settings
from services.api.db.clickhouse import insert_inference_logs
from services.api.ingestion.enricher import enrich
from services.api.ingestion.validator import InferenceLogIn

logger = logging.getLogger(__name__)


async def _ensure_group(redis: Redis, settings: Settings) -> None:
    try:
        await redis.xgroup_create(
            settings.ingestion_stream,
            settings.ingestion_consumer_group,
            id="0",
            mkstream=True,
        )
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def _to_row(payload: bytes | str) -> dict:
    data = json.loads(payload)
    log = InferenceLogIn.model_validate(data)
    row = enrich(log)
    # Convert epoch-ms ints into datetimes for ClickHouse DateTime64(3).
    row["started_at"] = datetime.fromtimestamp(row["started_at"] / 1000, tz=timezone.utc)
    row["ended_at"] = datetime.fromtimestamp(row["ended_at"] / 1000, tz=timezone.utc)
    return row


async def _flush(settings: Settings, rows: list[dict], attempt: int = 0) -> bool:
    try:
        await insert_inference_logs(settings, rows)
        return True
    except Exception as e:  # noqa: BLE001
        if attempt >= settings.max_retries:
            logger.error("clickhouse insert failed after %d attempts: %s", attempt, e)
            return False
        backoff = 0.25 * (2 ** attempt)
        logger.warning("clickhouse insert failed (attempt %d): %s — backoff %.2fs", attempt, e, backoff)
        await asyncio.sleep(backoff)
        return await _flush(settings, rows, attempt + 1)


async def run_worker(redis: Redis, settings: Settings) -> None:
    await _ensure_group(redis, settings)
    stream = settings.ingestion_stream
    group = settings.ingestion_consumer_group
    consumer = settings.ingestion_consumer_name
    logger.info("ingestion worker consuming stream=%s group=%s", stream, group)

    while True:
        try:
            entries = await redis.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={stream: ">"},
                count=settings.batch_size,
                block=settings.batch_flush_ms,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("xreadgroup failed: %s", e)
            await asyncio.sleep(1.0)
            continue

        if not entries:
            continue

        rows: list[dict] = []
        ids: list[bytes] = []
        bad_ids: list[bytes] = []

        for _stream_name, msgs in entries:
            for msg_id, fields in msgs:
                raw = fields.get(b"payload") or fields.get("payload")
                try:
                    rows.append(_to_row(raw))
                    ids.append(msg_id)
                except Exception as e:  # noqa: BLE001
                    logger.warning("dropping malformed log %s: %s", msg_id, e)
                    bad_ids.append(msg_id)

        if rows:
            ok = await _flush(settings, rows)
            if ok:
                await redis.xack(stream, group, *ids)
            else:
                # dead-letter and ack to prevent infinite replay
                for msg_id, row in zip(ids, rows):
                    await redis.xadd(
                        settings.ingestion_dlq_stream,
                        {"payload": json.dumps(row, default=str)},
                    )
                await redis.xack(stream, group, *ids)

        if bad_ids:
            await redis.xack(stream, group, *bad_ids)
