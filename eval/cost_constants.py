"""Cost constants for the OSS provider used in the eval.

This deployment runs the app stack on a CPU-only Azure VM and uses OpenCode-Go
(hosted) for OSS inference, because the Azure subscription has no GPU quota.
The amortized-per-VM-hour figure therefore doesn't apply — OpenCode bills
per token, and the eval report falls back to the per-call cost the SDK
records (which is currently $0 since OpenCode-Go's `ModelCost` is unpriced
in `sdk/registry.py`).

If you later self-host an OSS model (e.g. swap to a GPU VM and run vLLM),
set `OSS_SELF_HOSTED = True` and refresh the SKU + hourly figures — the
eval report will recompute amortized $/1M output.
"""

from __future__ import annotations

OSS_SELF_HOSTED: bool = False
OSS_HARDWARE_COST_PER_HOUR_USD: float = 0.75    # Standard_L8aos_v4 on-demand
OSS_SKU: str = "Standard_L8aos_v4 (CPU only — vLLM not running)"
OSS_PRICING_NOTE: str = (
    "Deployment is CPU-only; OSS inference uses hosted OpenCode-Go. "
    "Per-token cost is taken from OpenCode pricing when populated in the SDK registry."
)
