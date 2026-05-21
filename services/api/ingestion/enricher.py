"""Post-validation enrichment.

Cheap row-level transforms applied between Redis and ClickHouse. Today this is
just a placeholder — if pricing tables drift or session lookups are needed
later, do them here, not in the validator or the worker.
"""

from __future__ import annotations

from services.api.ingestion.validator import InferenceLogIn


def enrich(log: InferenceLogIn) -> dict:
    """Convert a validated log into the ClickHouse row dict."""
    row = log.model_dump()
    # Defensive: never write negative numbers.
    for k in ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens"):
        row[k] = max(0, int(row.get(k) or 0))
    row["cost_usd"] = max(0.0, float(row.get("cost_usd") or 0.0))
    return row
