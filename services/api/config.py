"""Application settings loaded from environment variables.

All secrets and URLs come from env vars. Never hardcode.
See `.env.example` for the full list of supported variables.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Postgres ───────────────────────────────────────────────────────────
    postgres_url: str = Field(
        default="postgresql://ollive:ollive@localhost:5432/ollive",
        description="asyncpg DSN. The 'postgresql+asyncpg://' prefix is stripped at runtime.",
    )

    # ── ClickHouse ─────────────────────────────────────────────────────────
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 9000
    clickhouse_db: str = "ollive"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""

    # ── Redis ──────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    ingestion_stream: str = "inference_logs"
    ingestion_dlq_stream: str = "inference_logs_dlq"
    ingestion_consumer_group: str = "ingest"
    ingestion_consumer_name: str = "worker-1"

    # ── API ────────────────────────────────────────────────────────────────
    api_port: int = 8000
    ingestion_url: str = "http://localhost:8000/ingest/log"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # ── Ingestion worker ───────────────────────────────────────────────────
    batch_size: int = 100
    batch_flush_ms: int = 1000
    max_retries: int = 3

    # ── LLM provider keys ──────────────────────────────────────────────────
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    huggingface_api_key: str = ""
    groq_api_key: str = ""
    deepseek_api_key: str = ""

    def api_keys(self) -> dict[str, str]:
        """Provider → key mapping for `LLMWrapper(api_keys=...)`."""
        keys = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "google": self.google_api_key,
            "huggingface": self.huggingface_api_key,
            "groq": self.groq_api_key,
            "deepseek": self.deepseek_api_key,
        }
        return {k: v for k, v in keys.items() if v}

    def asyncpg_dsn(self) -> str:
        """asyncpg does not accept the SQLAlchemy '+asyncpg' suffix."""
        return self.postgres_url.replace("postgresql+asyncpg://", "postgresql://", 1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
