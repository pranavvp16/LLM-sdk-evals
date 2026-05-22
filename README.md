# Ollive Inference Platform

Multi-provider LLM chatbot + inference logging pipeline. Submission for the
Ollive AI assignment (`work@ollive.ai`).

A single `docker compose up` brings up the full stack: chatbot UI, FastAPI
backend, ingestion worker, Postgres, ClickHouse, Redis, Prometheus, and Grafana.

> **Authoritative spec:** [`CLAUDE.md`](./CLAUDE.md) — full architecture
> diagram, schema decisions, tool-calling design, and the eval rubric.

---

## 1. Quick start

You need **Docker Desktop** (or Docker Engine + Compose v2) and **at least
one LLM API key**.

```bash
git clone https://github.com/pranavvp16/LLM-sdk-evals.git
cd LLM-sdk-evals

cp .env.example .env
# open .env and fill in ANTHROPIC_API_KEY (and/or OPENAI_API_KEY / GOOGLE_API_KEY)

docker compose up --build
```

That's it. First build takes ~3 minutes (pulls Postgres, ClickHouse, Redis,
Grafana, Prometheus images; builds the api + chatbot images). Subsequent
runs start in ~10 seconds.

When you see the api log line `Uvicorn running on http://0.0.0.0:8000`,
everything is ready.

| Service        | URL                                    | Notes                              |
|----------------|----------------------------------------|------------------------------------|
| Chatbot UI     | http://localhost:3000                  | Chat, conversation list, compare   |
| Compare view   | http://localhost:3000/compare          | Side-by-side OSS vs Frontier       |
| Dashboard      | http://localhost:3000/dashboard        | Embedded Grafana panels            |
| FastAPI docs   | http://localhost:8000/docs             | Swagger UI                         |
| Prometheus     | http://localhost:9090                  | Raw metrics                        |
| Grafana        | http://localhost:3001                  | `admin / admin`                    |
| Postgres       | `localhost:5432`                       | `ollive / ollive / ollive`         |
| ClickHouse     | http://localhost:8123                  | HTTP interface                     |
| Redis          | `localhost:6379`                       |                                    |

### Shut down

```bash
docker compose down            # stop containers, keep data
docker compose down -v         # also drop Postgres/ClickHouse/Redis/Grafana volumes
```

---

## 2. Required environment variables

Only `ANTHROPIC_API_KEY` is strictly required to chat. The rest unlock
additional providers and the eval comparison.

| Var                    | Required | Purpose                                          |
|------------------------|----------|--------------------------------------------------|
| `ANTHROPIC_API_KEY`    | yes      | Claude Sonnet / Haiku — default chat + judge     |
| `OPENAI_API_KEY`       | no       | GPT models + OpenAI-compatible vLLM endpoint     |
| `GOOGLE_API_KEY`       | no       | Gemini                                           |
| `HUGGINGFACE_API_KEY`  | no       | OSS model for eval + HF Inference API fallback   |
| `VLLM_BASE_URL`        | no       | Self-hosted Qwen2.5-0.5B (OpenAI-compatible)     |
| `VLLM_API_KEY`         | no       | Bearer token for vLLM (`EMPTY` works by default) |
| `OPENCODE_API_KEY`     | no       | OpenCode Zen (`opencode`) and Go (`opencode-go`); same key as pi |
| `OPENCODE_ZEN_BASE_URL`| no       | Zen API default `https://opencode.ai/zen/v1`     |
| `OPENCODE_GO_BASE_URL` | no       | Go API default `https://opencode.ai/zen/go/v1`   |
| `OSS_PROVIDER`         | no       | CLI eval: `vllm`, `opencode`, `opencode-go`, or `huggingface` |
| `POSTGRES_*`           | preset   | Defaults work out of the box                     |
| `CLICKHOUSE_*`         | preset   | Defaults work out of the box                     |
| `REDIS_URL`            | preset   | Defaults work out of the box                     |

Everything else in `.env.example` is preset to sane defaults that match the
service hostnames inside the Compose network — don't change them unless you
know what you're doing.

---

## 3. First chat (smoke test)

1. Open http://localhost:3000.
2. Pick a provider/model from the dropdown.
3. Type a message — you should see the response stream token-by-token.
4. Try `Schedule a 30-minute call with Priya at 10am tomorrow` — the model
   will call the `schedule_call` tool and the UI will render a tool card
   inline with the returned `event_id`.
5. Open http://localhost:3000/compare — pick any two models (vLLM Qwen, OpenCode
   Go GLM-5, HF Qwen, or Claude) and run the same prompt side-by-side.
6. Open http://localhost:3001 (Grafana, `admin/admin`) → the latency,
   throughput, and error dashboards populate within a couple of seconds.

---

## 4. Running tests (host, not Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -v
```

Tests use a fake provider (`tests/conftest.py`) — no real LLM calls, no API
keys needed. The 18-test SDK suite (`tests/test_sdk.py`) is frozen and must
keep passing.

---

## 5. Running the evaluation (Part A deliverable)

The eval pits an **OSS model** (DeepSeek-V4-Flash via OpenCode Go by default,
or vLLM-hosted Qwen / HF Inference) against **Claude Sonnet 4.6** across three
layers — judged by Sonnet:

- **L1 heuristic** — deterministic checks against the shared system prompt
  (persona signature, required headers, banned phrases, length bounds).
- **L2 LLM-as-judge static** — 30 plain-text prompts (10 factual / 10
  adversarial / 10 bias) scored 0–5 on hallucination, bias, safety.
- **L3 LLM-as-judge agent** — 20 tool-use prompts, 5-axis trajectory scoring
  (tool selection, arg correctness, task completion, output grounding, tool
  safety).

Output: `eval/results.json` + a 9-page `docs/eval_report.pdf` (when both
static and agent sections are present): summary, static methodology /
per-category / failures, then agent methodology / summary / per-category /
walkthrough / worst trajectories. Also writes `docs/eval_cost_latency.md`.
Browse interactively at
[`http://localhost:3000/compare?view=benchmark`](http://localhost:3000/compare?view=benchmark).

```bash
# Both layers (default). Needs ANTHROPIC_API_KEY plus an OSS backend (OPENCODE_API_KEY
# for L3 tool-use, or HUGGINGFACE_API_KEY for static-only).
python eval/run_eval.py

python eval/run_eval.py --static            # just the 30 static prompts
python eval/run_eval.py --agent             # just the 20 agent trajectories
python eval/run_eval.py --all --limit 3     # smoke test
python eval/report.py                       # regenerate PDF from results.json
```

The OSS backend is selected by `Settings.resolve_oss()` (see CLAUDE.md §5):
explicit `OSS_PROVIDER` → `opencode-go` if `OPENCODE_API_KEY` is set →
`huggingface` if only `HUGGINGFACE_API_KEY` is set. `huggingface` is
static-only (no tool support); the L3 agent eval needs `opencode-go` or vLLM.

---

## 6. Architecture

```
┌──────────── frontend ─────────────────────────────────────────┐
│  Next.js chat UI       Eval dashboard       Side-by-side eval │
└──────────────────────────────┬────────────────────────────────┘
                               │ HTTP + SSE
┌──────────────────────────── SDK (Python) ─────────────────────┐
│  LLMWrapper: multi-provider, streaming, PII redact, metadata  │
└──────────────────────────────┬────────────────────────────────┘
                               │ fire-and-forget POST /ingest/log
┌──────────────────────────── ingestion ────────────────────────┐
│  FastAPI /ingest/log   →  Redis Streams  →  async worker      │
└────────────────┬─────────────────────────────┬────────────────┘
                 │                             │
        PostgreSQL                       ClickHouse
        (conversations,                  (inference_logs,
         messages, sessions)              90-day TTL)
                 │                             │
                 ▼                             ▼
              Grafana (latency / throughput / errors)
```

The full diagram (ASCII), schema definitions, and design rationale live in
[`CLAUDE.md`](./CLAUDE.md) §2 and §7.

---

## 7. Repo layout

```
sdk/                  Multi-provider LLM wrapper (frozen)
services/api/         FastAPI backend + ingestion worker + tool registry
services/chatbot/     Next.js 14 chatbot UI
eval/                 LLM-as-judge runner + PDF report
infra/                Prometheus + Grafana provisioning
scripts/              run_migrations.py, seed_db.py
tests/                pytest-asyncio
docs/                 Eval report PDF + cost/latency markdown
```

---

## 8. Troubleshooting

**`api` container restarts in a loop** — usually means `ANTHROPIC_API_KEY` is
empty or your `.env` was not picked up. `docker compose config` to verify,
then `docker compose up --build` again.

**Chatbot shows "fetch failed"** — the api isn't ready yet. Wait 5-10 seconds
after `Uvicorn running on…` appears in the logs.

**Grafana dashboards are empty** — they only populate once at least one chat
message has flowed through. Send a message in the chatbot, then refresh.

**Tool calls don't render in the UI but work via curl** — the chatbot image
is baked at build time (Next.js production build), so frontend changes need
`docker compose build chatbot` to take effect.

**Port already in use** — change `API_PORT` / `CHATBOT_PORT` /
`GRAFANA_PORT` / `PROMETHEUS_PORT` in `.env`.

**Reset everything from scratch** —
`docker compose down -v && docker compose up --build`.
