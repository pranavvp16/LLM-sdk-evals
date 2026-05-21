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

    # ── OSS backends (vLLM, OpenCode Go, HF fallback) ─────────────────────
    vllm_base_url: str = ""
    vllm_api_key: str = "EMPTY"
    vllm_model: str = "qwen2.5-0.5b-instruct"
    opencode_api_key: str = ""
    opencode_zen_base_url: str = "https://opencode.ai/zen/v1"
    opencode_zen_model: str = "kimi-k2.5"
    opencode_go_base_url: str = "https://opencode.ai/zen/go/v1"
    opencode_go_model: str = "glm-5"
    oss_provider: str = ""
    oss_model: str = ""

    def api_keys(self) -> dict[str, str]:
        """Provider → key mapping for `LLMWrapper(api_keys=...)`."""
        keys: dict[str, str] = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "google": self.google_api_key,
            "huggingface": self.huggingface_api_key,
            "groq": self.groq_api_key,
            "deepseek": self.deepseek_api_key,
        }
        if self.vllm_base_url:
            keys["vllm"] = self.vllm_api_key or "EMPTY"
        if self.opencode_api_key:
            # pi-ai: one OPENCODE_API_KEY for both Zen and Go provider ids
            keys["opencode"] = self.opencode_api_key
            keys["opencode-go"] = self.opencode_api_key
        return {k: v for k, v in keys.items() if v}

    def base_urls(self) -> dict[str, str]:
        """Provider → base URL overrides for `LLMWrapper(base_urls=...)`."""
        urls: dict[str, str] = {}
        if self.vllm_base_url:
            urls["vllm"] = self.vllm_base_url.rstrip("/")
        if self.opencode_zen_base_url:
            urls["opencode"] = self.opencode_zen_base_url.rstrip("/")
        if self.opencode_go_base_url:
            urls["opencode-go"] = self.opencode_go_base_url.rstrip("/")
        return urls

    def resolve_oss(self) -> tuple[str, str]:
        """Return (provider, model_id) for CLI eval OSS side."""
        defaults: dict[str, str] = {
            "vllm": self.vllm_model,
            "opencode": self.opencode_zen_model,
            "opencode-go": self.opencode_go_model,
            "huggingface": "qwen2.5-0.5b-instruct",
        }
        if self.oss_provider:
            return self.oss_provider, self.oss_model or defaults.get(
                self.oss_provider, self.vllm_model
            )
        if self.vllm_base_url:
            return "vllm", self.vllm_model
        return "huggingface", "qwen2.5-0.5b-instruct"

    def asyncpg_dsn(self) -> str:
        """asyncpg does not accept the SQLAlchemy '+asyncpg' suffix."""
        return self.postgres_url.replace("postgresql+asyncpg://", "postgresql://", 1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
