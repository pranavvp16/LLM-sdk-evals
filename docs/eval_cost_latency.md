# Eval — Cost & Latency

OSS: `opencode-go/deepseek-v4-flash` on `Standard_E8s_v5 (8 vCPU / 64 GB, CPU only)` ($0.504/hr)

Frontier: `claude-sonnet-4-6`

Wall-clock for this run: **3.8 min**

| Metric | OSS | Frontier |
|---|---|---|
| Median latency | 4709 ms | 4389 ms |
| p95 latency | 4624 ms | 3261 ms |
| Total output tokens | 461 | 254 |
| Effective throughput | 48.9 tok/s | n/a (hosted) |
| Hardware cost | $0.504/hr | n/a |
| This run cost | $0.0319 | $0.0130 |
| $ / 1M output tokens (amortized for self-hosted) | $2.86 | $51.21 |

_OSS endpoint is Ollama serving qwen2.5:1.5b (Q4_K_M) on the same VM as the rest of the stack. Per-call cost is $0; amortized $/1M output tokens is hardware-hours × $0.504 / total OSS output tokens × 1e6._
