"""Pydantic v2 schema mirroring `sdk.types.InferenceLog`.

We validate inbound logs at the HTTP boundary (POST /ingest/log) before
pushing to Redis — keeps malformed rows out of the stream.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Status = Literal["success", "error", "cancelled"]


class InferenceLogIn(BaseModel):
    trace_id: str
    session_id: str
    conversation_id: str
    provider: str
    model: str
    api_protocol: str
    started_at: int = Field(ge=0)
    ended_at: int = Field(ge=0)
    latency_ms: float = Field(gt=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    input_preview: str = Field(max_length=200)
    output_preview: str = Field(max_length=200)
    status: Status
    error_message: Optional[str] = None
    stop_reason: str = ""
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    @field_validator("input_preview", "output_preview", mode="before")
    @classmethod
    def _clip(cls, v: object) -> str:
        s = "" if v is None else str(v)
        return s[:200]
