"""Liveness and readiness endpoints.

`/health` is a cheap liveness probe.
`/ready` checks that Postgres and Redis are reachable.
Prometheus metrics live at `/metrics`, mounted by `Instrumentator` in main.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from services.api.deps import pg_pool, redis_client

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(
    pool=Depends(pg_pool),
    redis=Depends(redis_client),
) -> dict[str, str]:
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"postgres: {e}") from e
    try:
        await redis.ping()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"redis: {e}") from e
    return {"status": "ready"}
