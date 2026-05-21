"""Read-only ClickHouse aggregates used by the dashboard UI.

These endpoints are *not* `/metrics` (that path is owned by Prometheus). They
read from `inference_logs` to feed the in-app dashboard cards and the recent-
calls table.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from services.api.config import Settings
from services.api.db.clickhouse import execute
from services.api.deps import settings_dep

router = APIRouter(tags=["metrics"])


@router.get("/metrics/summary")
async def metrics_summary(
    hours: Annotated[int, Query(ge=1, le=24 * 30)] = 24,
    settings: Settings = Depends(settings_dep),
) -> dict:
    """Aggregates over the trailing `hours` window."""
    where = f"started_at >= now() - INTERVAL {hours} HOUR"

    totals_sql = f"""
        SELECT
            count() AS total_calls,
            sum(status = 'success') AS success_calls,
            sum(status = 'error') AS error_calls,
            sum(input_tokens) AS input_tokens,
            sum(output_tokens) AS output_tokens,
            sum(cost_usd) AS cost_usd,
            quantile(0.5)(latency_ms) AS p50_ms,
            quantile(0.95)(latency_ms) AS p95_ms,
            quantile(0.99)(latency_ms) AS p99_ms
        FROM inference_logs
        WHERE {where}
    """
    by_provider_sql = f"""
        SELECT
            provider,
            count() AS calls,
            sum(cost_usd) AS cost_usd,
            quantile(0.95)(latency_ms) AS p95_ms,
            sum(status = 'error') / count() AS error_rate
        FROM inference_logs
        WHERE {where}
        GROUP BY provider
        ORDER BY calls DESC
    """

    totals_rows = await execute(settings, totals_sql)
    by_rows = await execute(settings, by_provider_sql)

    if totals_rows and totals_rows[0][0]:
        t = totals_rows[0]
        totals = {
            "total_calls": int(t[0]),
            "success_calls": int(t[1]),
            "error_calls": int(t[2]),
            "input_tokens": int(t[3]),
            "output_tokens": int(t[4]),
            "cost_usd": float(t[5]),
            "p50_ms": float(t[6]),
            "p95_ms": float(t[7]),
            "p99_ms": float(t[8]),
        }
    else:
        totals = {
            "total_calls": 0,
            "success_calls": 0,
            "error_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }

    by_provider = [
        {
            "provider": str(r[0]),
            "calls": int(r[1]),
            "cost_usd": float(r[2]),
            "p95_ms": float(r[3]),
            "error_rate": float(r[4]),
        }
        for r in by_rows
    ]

    return {"window_hours": hours, "totals": totals, "by_provider": by_provider}


@router.get("/metrics/recent")
async def metrics_recent(
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
    settings: Settings = Depends(settings_dep),
) -> list[dict]:
    sql = f"""
        SELECT
            started_at,
            provider,
            model,
            status,
            latency_ms,
            total_tokens,
            cost_usd,
            substring(output_preview, 1, 80) AS preview,
            error_message
        FROM inference_logs
        ORDER BY started_at DESC
        LIMIT {limit}
    """
    rows = await execute(settings, sql)
    return [
        {
            "started_at": (r[0].isoformat() if isinstance(r[0], datetime) else str(r[0])),
            "provider": str(r[1]),
            "model": str(r[2]),
            "status": str(r[3]),
            "latency_ms": float(r[4]),
            "total_tokens": int(r[5]),
            "cost_usd": float(r[6]),
            "preview": str(r[7] or ""),
            "error_message": (str(r[8]) if r[8] is not None else None),
        }
        for r in rows
    ]
