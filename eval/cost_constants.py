"""Cost constants for the OSS provider used in the eval.

OSS inference runs on the same Azure VM that hosts the rest of the stack,
via Ollama serving `qwen2.5:1.5b` (Q4_K_M, ~1 GB). Because the model is
self-hosted, per-call `ModelCost` in the SDK registry is zero and the
real cost is the amortized VM hour, computed from these constants by
`eval/report.py`.
"""

from __future__ import annotations

OSS_SELF_HOSTED: bool = True
OSS_HARDWARE_COST_PER_HOUR_USD: float = 0.504   # Standard_E8s_v5 on-demand, eastus
OSS_SKU: str = "Standard_E8s_v5 (8 vCPU / 64 GB, CPU only)"
OSS_PRICING_NOTE: str = (
    "OSS endpoint is Ollama serving qwen2.5:1.5b (Q4_K_M) on the same VM "
    "as the rest of the stack. Per-call cost is $0; amortized $/1M output "
    "tokens is hardware-hours × $0.504 / total OSS output tokens × 1e6."
)
