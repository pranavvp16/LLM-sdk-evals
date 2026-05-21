"""Apply Postgres and ClickHouse SQL migrations.

Runs every `*_postgres.sql` and `*_clickhouse.sql` file under
`services/api/db/migrations/` in lexicographic order. Idempotent: each file
should use `IF NOT EXISTS` so re-running is a no-op.

Called by the api container's entrypoint before uvicorn starts.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import asyncpg
from clickhouse_driver import Client

from services.api.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "services" / "api" / "db" / "migrations"


def _find(suffix: str) -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob(f"*_{suffix}.sql"))


async def run_postgres() -> None:
    settings = get_settings()
    conn = await asyncpg.connect(dsn=settings.asyncpg_dsn())
    try:
        for path in _find("postgres"):
            sql = path.read_text()
            logger.info("postgres ← %s", path.name)
            try:
                await conn.execute(sql)
            except asyncpg.exceptions.DuplicateTableError:
                logger.info("  (already applied — skipping)")
    finally:
        await conn.close()


def run_clickhouse() -> None:
    settings = get_settings()
    client = Client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        database=settings.clickhouse_db,
        user=settings.clickhouse_user,
        password=settings.clickhouse_password,
    )
    try:
        for path in _find("clickhouse"):
            logger.info("clickhouse ← %s", path.name)
            sql = path.read_text()
            # ClickHouse driver runs one statement per execute call.
            for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                client.execute(stmt)
    finally:
        client.disconnect()


async def main() -> None:
    await run_postgres()
    run_clickhouse()
    logger.info("migrations complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:  # noqa: BLE001
        logger.error("migration failed: %s", e)
        sys.exit(1)
