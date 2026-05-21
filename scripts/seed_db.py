"""Seed a few sample conversations + messages for local development.

Run after `run_migrations.py`:
    python scripts/seed_db.py
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

import asyncpg

from services.api.config import get_settings
from services.api.db import postgres as pg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


SEEDS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "anthropic",
        "claude-sonnet-4-6",
        [
            ("user", "What's the capital of France?"),
            ("assistant", "The capital of France is Paris."),
        ],
    ),
    (
        "openai",
        "gpt-4o-mini",
        [("user", "Write a haiku about observability.")],
    ),
]


async def main() -> None:
    settings = get_settings()
    pool = await asyncpg.create_pool(dsn=settings.asyncpg_dsn())
    try:
        session_id = uuid4()
        await pg.upsert_session(pool, session_id, user_agent="seed-script", ip=None)
        for provider, model, msgs in SEEDS:
            conv = await pg.create_conversation(
                pool,
                session_id=session_id,
                provider=provider,
                model=model,
                title=msgs[0][1][:80],
            )
            for role, content in msgs:
                await pg.add_message(pool, conv["id"], role=role, content=content)
            logger.info("seeded conversation %s (%s/%s)", conv["id"], provider, model)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
