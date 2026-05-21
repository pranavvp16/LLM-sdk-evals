"""FastAPI dependency-injection helpers.

Resources (Postgres pool, Redis client, LLMWrapper) are created once in
`main.py`'s lifespan handler and stashed on `app.state`. Routes pull them out
via these dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, Request

from services.api.config import Settings, get_settings

if TYPE_CHECKING:
    import asyncpg
    from redis.asyncio import Redis

    from sdk import LLMWrapper

    from services.api.tools.registry import ToolRegistry


def settings_dep() -> Settings:
    return get_settings()


def pg_pool(request: Request) -> "asyncpg.Pool":
    pool = getattr(request.app.state, "pg_pool", None)
    if pool is None:
        raise RuntimeError("Postgres pool not initialised — check lifespan handler")
    return pool


def redis_client(request: Request) -> "Redis":
    client = getattr(request.app.state, "redis", None)
    if client is None:
        raise RuntimeError("Redis client not initialised — check lifespan handler")
    return client


def llm_wrapper(request: Request) -> "LLMWrapper":
    wrapper = getattr(request.app.state, "llm_wrapper", None)
    if wrapper is None:
        raise RuntimeError("LLMWrapper not initialised — check lifespan handler")
    return wrapper


def tools_registry(request: Request) -> "ToolRegistry":
    registry = getattr(request.app.state, "tools", None)
    if registry is None:
        raise RuntimeError("ToolRegistry not initialised — check lifespan handler")
    return registry


__all__ = [
    "Settings",
    "settings_dep",
    "pg_pool",
    "redis_client",
    "llm_wrapper",
    "tools_registry",
    "Depends",
]
