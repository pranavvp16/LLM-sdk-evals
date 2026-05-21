"""Postgres data access for sessions / conversations / messages.

All queries use asyncpg parameter binding ($1, $2, …) — never f-string SQL.
Functions take a pool or a connection so callers can compose them inside a
transaction when they need to.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional
from uuid import UUID

import asyncpg


# ── sessions ──────────────────────────────────────────────────────────────


def hash_ip(ip: str | None) -> str | None:
    """SHA-256 the IP before storing — we never persist raw addresses."""
    if not ip:
        return None
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()


async def upsert_session(
    pool: asyncpg.Pool,
    session_id: UUID,
    user_agent: str | None,
    ip: str | None,
) -> None:
    """Insert a session if it doesn't exist. Idempotent."""
    await pool.execute(
        """
        INSERT INTO sessions (id, user_agent, ip_hash)
        VALUES ($1, $2, $3)
        ON CONFLICT (id) DO NOTHING
        """,
        session_id,
        user_agent,
        hash_ip(ip),
    )


# ── conversations ─────────────────────────────────────────────────────────


async def create_conversation(
    pool: asyncpg.Pool,
    session_id: UUID,
    provider: str,
    model: str,
    title: str = "",
) -> dict[str, Any]:
    row = await pool.fetchrow(
        """
        INSERT INTO conversations (session_id, provider, model, title)
        VALUES ($1, $2, $3, $4)
        RETURNING id, session_id, title, provider, model, status, created_at, updated_at
        """,
        session_id,
        provider,
        model,
        title,
    )
    return dict(row)


async def get_conversation(pool: asyncpg.Pool, conv_id: UUID) -> Optional[dict[str, Any]]:
    row = await pool.fetchrow(
        "SELECT * FROM conversations WHERE id = $1",
        conv_id,
    )
    return dict(row) if row else None


async def list_conversations(
    pool: asyncpg.Pool,
    session_id: UUID | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    if session_id is None:
        rows = await pool.fetch(
            """
            SELECT * FROM conversations
            ORDER BY updated_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT * FROM conversations
            WHERE session_id = $1
            ORDER BY updated_at DESC
            LIMIT $2 OFFSET $3
            """,
            session_id,
            limit,
            offset,
        )
    return [dict(r) for r in rows]


async def set_conversation_status(
    pool: asyncpg.Pool,
    conv_id: UUID,
    status: str,
) -> None:
    cancelled_at = "now()" if status == "cancelled" else "NULL"
    await pool.execute(
        f"""
        UPDATE conversations
        SET status = $1,
            updated_at = now(),
            cancelled_at = {cancelled_at}
        WHERE id = $2
        """,
        status,
        conv_id,
    )


# ── messages ──────────────────────────────────────────────────────────────


async def add_message(
    pool: asyncpg.Pool,
    conv_id: UUID,
    role: str,
    content: str,
    token_count: int = 0,
    tool_call_id: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    is_error: bool = False,
) -> dict[str, Any]:
    import json as _json
    row = await pool.fetchrow(
        """
        INSERT INTO messages (
            conversation_id, role, content, token_count,
            tool_call_id, tool_calls, is_error
        )
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
        RETURNING id, conversation_id, role, content,
                  tool_call_id, tool_calls, is_error,
                  token_count, created_at
        """,
        conv_id,
        role,
        content,
        token_count,
        tool_call_id,
        _json.dumps(tool_calls) if tool_calls is not None else None,
        is_error,
    )
    # touch the parent conversation so list ordering reflects activity
    await pool.execute(
        "UPDATE conversations SET updated_at = now() WHERE id = $1",
        conv_id,
    )
    return dict(row)


async def list_messages(pool: asyncpg.Pool, conv_id: UUID) -> list[dict[str, Any]]:
    import json as _json
    rows = await pool.fetch(
        """
        SELECT id, conversation_id, role, content,
               tool_call_id, tool_calls, is_error,
               token_count, created_at
        FROM messages
        WHERE conversation_id = $1
        ORDER BY created_at ASC
        """,
        conv_id,
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        # asyncpg returns jsonb as a string — parse so callers get a list[dict].
        if isinstance(d.get("tool_calls"), str):
            d["tool_calls"] = _json.loads(d["tool_calls"])
        out.append(d)
    return out
