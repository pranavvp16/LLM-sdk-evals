"""ClickHouse access wrapped in `asyncio.to_thread` so the blocking driver
does not stall the asyncio event loop.

We use the synchronous `clickhouse-driver` (native protocol) because the async
forks are less mature. The batch worker is the only hot caller, so wrapping it
once at the boundary is enough.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable

from clickhouse_driver import Client

from services.api.config import Settings

logger = logging.getLogger(__name__)

INFERENCE_COLUMNS: tuple[str, ...] = (
    "trace_id",
    "session_id",
    "conversation_id",
    "provider",
    "model",
    "api_protocol",
    "started_at",
    "ended_at",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_tokens",
    "cost_usd",
    "input_preview",
    "output_preview",
    "status",
    "error_message",
    "stop_reason",
    "stream",
    "temperature",
    "max_tokens",
)


def _build_client(settings: Settings) -> Client:
    return Client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        database=settings.clickhouse_db,
        user=settings.clickhouse_user,
        password=settings.clickhouse_password,
    )


async def insert_inference_logs(settings: Settings, rows: Iterable[dict]) -> int:
    """Batch insert into `inference_logs`. Returns the number of rows written."""
    rows_list = list(rows)
    if not rows_list:
        return 0

    def _insert() -> int:
        client = _build_client(settings)
        try:
            client.execute(
                f"INSERT INTO inference_logs ({', '.join(INFERENCE_COLUMNS)}) VALUES",
                rows_list,
                types_check=True,
            )
        finally:
            client.disconnect()
        return len(rows_list)

    return await asyncio.to_thread(_insert)


async def execute(settings: Settings, sql: str, params: dict | None = None) -> list[tuple]:
    """Run an arbitrary query (used by health checks and migrations)."""

    def _exec() -> list[tuple]:
        client = _build_client(settings)
        try:
            return client.execute(sql, params or {})
        finally:
            client.disconnect()

    return await asyncio.to_thread(_exec)
