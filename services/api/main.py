"""FastAPI application factory and lifespan management.

Owns the lifecycle of:
- The asyncpg Postgres pool
- The async Redis client
- The shared LLMWrapper (one per process, reused across requests)
- The Redis → ClickHouse ingestion worker task

These are attached to `app.state` and pulled into routes via `deps.py`.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from redis.asyncio import Redis

from sdk import LLMWrapper

from services.api.config import get_settings
from services.api.oss_registry import register_oss_models
from services.api.routers import chat, health, ingest, metrics
from services.api.tools.builtins import register_builtins
from services.api.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    register_oss_models()

    app.state.pg_pool = await asyncpg.create_pool(
        dsn=settings.asyncpg_dsn(),
        min_size=1,
        max_size=10,
    )
    logger.info("postgres pool ready")

    app.state.redis = Redis.from_url(
        settings.redis_url,
        decode_responses=False,
    )
    await app.state.redis.ping()
    logger.info("redis ready")

    app.state.llm_wrapper = LLMWrapper(
        api_keys=settings.api_keys(),
        base_urls=settings.base_urls(),
        ingestion_url=settings.ingestion_url,
    )
    logger.info("llm wrapper ready (providers: %s)", list(settings.api_keys().keys()))

    tools = ToolRegistry()
    register_builtins(tools)
    app.state.tools = tools
    logger.info("tool registry ready (%d tools: %s)", len(tools.names()), tools.names())

    # Ingestion worker — Redis stream consumer → ClickHouse batch writer.
    # Imported lazily so the API can start even before ClickHouse is reachable.
    from services.api.ingestion.worker import run_worker

    app.state.worker_task = asyncio.create_task(
        run_worker(app.state.redis, settings),
        name="ingestion-worker",
    )
    logger.info("ingestion worker started")

    try:
        yield
    finally:
        task: asyncio.Task = app.state.worker_task
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        await app.state.redis.aclose()
        await app.state.pg_pool.close()
        logger.info("shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Ollive Inference Platform",
        description="Multi-provider chatbot + inference logging pipeline",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Session-ID"],
    )

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(ingest.router)
    app.include_router(metrics.router)

    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


app = create_app()
