# CLAUDE.md

The single authoritative spec for this repo. Read top-to-bottom before writing
any code. Replaces the older `AGENTS.md` (now removed). If anything in code
disagrees with this file, this file wins until updated.

---

## 1. What we're building

An **LLM inference observability platform** plus the **AI personal assistant**
that runs on top of it. Two deliverables, one repo, one `docker compose up`.

**Part A — AI Personal Assistant + Evaluation**
A multi-turn chatbot with short-term memory and a small set of mock daily-life
tools (schedule call, update calendar, set reminder, etc.). The same UI talks
to two models:

- **OSS:** `deepseek-v4-flash` (OpenCode Go) by default — a reasoning model
  with `supports_tools=True`. The previously-considered `glm-5` is also
  registered on the same provider for comparison. Other OpenAI-compatible backends are still selectable
  per request: **vLLM** (`vllm`), **OpenCode Zen** (`opencode`), or
  **HuggingFace Inference** (`huggingface`, used as a fallback when no
  OpenCode key is set). The chat UI and `/compare` pick provider per request;
  CLI eval picks via `Settings.resolve_oss()` (see §5).
- **Frontier:** `claude-sonnet-4-6` via Anthropic API.

The assistants are evaluated on three layers (see §10):
1. **Heuristic / instruction-following** — deterministic checks against a
   shared structured system prompt (persona signature, required headers,
   banned phrases, length bounds).
2. **LLM-as-judge static** — hallucination, bias, safety on 30 plain-text
   prompts.
3. **LLM-as-judge agent** — 5-axis trajectory scoring on 20 tool-use
   prompts (tool selection, argument correctness, task completion, output
   grounding, safety with tools).

Results browse at `http://localhost:3000/compare?view=benchmark` with
per-prompt drill-down, and as a generated PDF at `docs/eval_report.pdf`.

**Part B — Inference logging pipeline**
Every LLM call the chatbot makes flows through `LLMWrapper`, which captures an
`InferenceLog` and fires it at `POST /ingest/log`. From there: Redis Stream →
async worker → ClickHouse → Grafana. Prometheus scrapes FastAPI for
service-level metrics.

The chatbot is the application being observed. The pipeline is the
observability. They share the same Postgres, the same Compose file, and the
same SDK.

---

## 2. Architecture

```
┌──────────── frontend ─────────────────────────────────────────┐
│  Next.js chat UI       Eval dashboard       Side-by-side eval │
│  (list/resume/cancel)  (live metrics)       (OSS vs Frontier) │
└──────────────────────────────┬────────────────────────────────┘
                               │ HTTP + SSE
┌──────────────────────────── SDK (Python, DONE) ───────────────┐
│  LLMWrapper                                                   │
│  ├─ Multi-provider  (Anthropic, OpenAI/vLLM/Ollama, Google)   │
│  ├─ Streaming SSE   (token-by-token + tool_call events)       │
│  ├─ Metadata        (latency, tokens, cost, stop_reason)      │
│  └─ PII redact      (email/phone/SSN/CC before logging)       │
└──────────────────────────────┬────────────────────────────────┘
                               │ fire-and-forget POST /ingest/log
┌──────────────────────────── ingestion ────────────────────────┐
│  FastAPI /ingest/log   →  Redis Streams (buffer/queue)        │
│                              │                                │
│                              ▼                                │
│                       Async worker (validate, enrich, retry)  │
└────────────────┬─────────────────────────────┬────────────────┘
                 │                             │
┌── storage ─────┼─────────────────────────────┼────────────────┐
│  PostgreSQL                          ClickHouse               │
│  conversations / messages / sessions inference_logs (OLAP)    │
└────────────────┬─────────────────────────────┬────────────────┘
                 │                             │
┌── observe ─────▼─────────────────────────────▼────────────────┐
│  Grafana            Prometheus           Eval report PDF      │
│  latency/throughput error rate / p95     hallucination / bias │
│                     alerts               / safety             │
└───────────────────────────────────────────────────────────────┘
```

**Why this shape:**

- **Redis between API and ClickHouse:** `POST /ingest/log` returns in microseconds
  no matter what ClickHouse is doing. If ClickHouse is slow / restarting, logs
  queue up and replay. The chat path is never affected.
- **Postgres vs ClickHouse split:** conversations/messages have foreign keys,
  joins, soft-deletes — Postgres. Inference logs are append-only time-series
  with billion-row aggregation queries (`p95 latency by provider`) — ClickHouse.
- **PII boundary at the SDK:** nothing downstream of `sdk/pii.py` ever sees raw
  user input. Ingestion, ClickHouse, dashboards all trust the log is clean.

---

## 3. Hard rules

- **Treat `sdk/` as frozen, with one narrow exception.** The 18/18 SDK test
  suite must keep passing — no API changes, no refactors, no feature additions.
  The single exception is **provider bug-fixes that are required to make real
  LLM calls work**: the `tests/conftest.py` fake provider can mask wire-protocol
  mistakes against real APIs (the Anthropic `stream=True` bug was one such
  case). Such fixes must be minimal, must keep the existing tests green, and
  must be called out in the commit message. Never modify `tests/test_sdk.py`.
  All LLM calls go through `LLMWrapper`. Never import `anthropic`, `openai`, or
  `google-generativeai` outside `sdk/providers/`.
- **Async everywhere.** FastAPI routes are `async def`. Use `asyncpg` for
  Postgres, `redis.asyncio` for Redis, `asyncio.to_thread(...)` to wrap the
  synchronous `clickhouse-driver`. Never use `requests` — use `httpx` or
  `aiohttp`.
- **Parameterised SQL only.** No f-string SQL. asyncpg uses `$1, $2, ...`.
- **Secrets from env vars only.** `services/api/config.py` (pydantic-settings)
  for Python, `process.env` / `NEXT_PUBLIC_*` for TypeScript. Never hardcode.
- **PII redacted in the SDK, never downstream.** `InferenceLog.input_preview`
  and `.output_preview` are already clean. Don't redact again, don't log raw
  user input elsewhere.
- **Ingestion must never block the chat path.** `POST /ingest/log` returns in
  < 5 ms by `XADD`ing to Redis and returning. Heavy work happens in the worker.
  If Redis or ClickHouse is down, the chat keeps working.
- **IPs are SHA-256 hashed before storage.** Never store raw IPs.
- **No MCP.** Tool calling is in-process via a small `ToolRegistry` (see §6).

---

## 4. Repo layout

```
ollive-assignment/
├── CLAUDE.md                ← this file (single spec)
├── docker-compose.yml
├── .env.example
├── requirements.txt
│
├── sdk/                     ← DONE — DO NOT MODIFY
│   ├── types.py             canonical: StreamEvent, InferenceLog, Message, ModelDef, ToolCall
│   ├── registry.py          model catalog (Anthropic, OpenAI, Google, Groq, DeepSeek, Ollama, HF, vLLM)
│   ├── normalize.py         Context → provider wire format, incl. tool schemas
│   ├── pii.py               email/phone/SSN/CC redaction
│   ├── wrapper.py           LLMWrapper.stream() / .complete(), fires InferenceLog
│   └── providers/{anthropic,openai,google}_provider.py
│
├── services/
│   ├── api/                                FastAPI backend
│   │   ├── main.py                         app factory + lifespan
│   │   ├── config.py                       pydantic-settings
│   │   ├── deps.py                         pool / redis / wrapper / registry deps
│   │   ├── routers/
│   │   │   ├── chat.py                     SSE streaming + conversation CRUD + tool loop
│   │   │   ├── ingest.py                   POST /ingest/log → Redis XADD
│   │   │   ├── health.py
│   │   │   └── eval.py                     GET /eval/summary, /eval/compare for the side-by-side UI
│   │   ├── tools/                          in-process tool calling (no MCP)
│   │   │   ├── registry.py                 Tool dataclass + ToolRegistry
│   │   │   ├── executor.py                 run tool_calls, build ToolResultMessage
│   │   │   └── builtins.py                 mock daily-assistant tools
│   │   ├── ingestion/
│   │   │   ├── validator.py                Pydantic v2 mirror of InferenceLog
│   │   │   ├── enricher.py                 cost recalc, session linking, clamping
│   │   │   └── worker.py                   XREADGROUP → ClickHouse batch insert + DLQ
│   │   ├── db/
│   │   │   ├── postgres.py                 asyncpg pool + query functions
│   │   │   ├── clickhouse.py               clickhouse-driver via asyncio.to_thread
│   │   │   └── migrations/
│   │   │       ├── 001_postgres.sql
│   │   │       └── 001_clickhouse.sql
│   │   └── observability/
│   │       └── metrics.py                  Prometheus instrumentation (extra app counters)
│   │
│   └── chatbot/                            Next.js 14 App Router UI
│       ├── app/
│       │   ├── page.tsx                    conversation list
│       │   ├── chat/[id]/page.tsx          streaming chat + tool result rendering
│       │   ├── dashboard/page.tsx          metrics overview (Grafana iframes)
│       │   └── compare/page.tsx            side-by-side OSS vs Frontier eval view
│       ├── components/
│       └── lib/api.ts                      typed fetch client + Zod schemas
│
├── eval/                                   the Part A deliverable
│   ├── prompts.py                          30 prompts (10 factual / 10 adversarial / 10 bias)
│   ├── judge.py                            LLM-as-judge scorer (Sonnet)
│   ├── run_eval.py                         orchestrator, writes results.json
│   └── report.py                           writes docs/eval_report.pdf
│
├── infra/
│   ├── prometheus.yml
│   ├── vllm/                               docker-compose override for local vLLM (optional)
│   └── grafana/
│       ├── provisioning/datasources/clickhouse.yml
│       └── dashboards/{latency,throughput,errors}.json
│
├── tests/
│   ├── test_sdk.py                         DONE — 18/18 passing — DO NOT BREAK
│   ├── test_chat.py
│   ├── test_ingest.py
│   ├── test_tools.py                       registry + executor unit tests
│   └── conftest.py                         shared fixtures
│
├── scripts/
│   ├── run_migrations.py
│   └── seed_db.py
│
└── docs/
    ├── eval_report.pdf                     generated by eval/report.py
    └── eval_cost_latency.md                generated alongside the PDF
```

---

## 5. Provider configuration (OSS backends)

OSS inference is **not** run inside this repo. Each backend is a separate
registry provider id with an OpenAI-compatible wire format. Catalog entries for
`vllm` / `opencode` / `opencode-go` are registered at runtime via
`services/api/oss_registry.py` (keeps frozen `sdk/registry.py` unchanged).
`services/api/config.py` builds `api_keys` and `base_urls` for `LLMWrapper`:

| Provider id   | Use case              | Key env                         | URL env (optional override)      |
|---------------|-----------------------|---------------------------------|----------------------------------|
| `vllm`        | Self-hosted Qwen      | `VLLM_API_KEY` (when URL set)   | `VLLM_BASE_URL`                  |
| `opencode`    | OpenCode Zen (pi-ai)  | `OPENCODE_API_KEY`              | `OPENCODE_ZEN_BASE_URL`          |
| `opencode-go` | OpenCode Go plan      | same `OPENCODE_API_KEY`         | `OPENCODE_GO_BASE_URL`           |
| `huggingface` | HF Inference API      | `HUGGINGFACE_API_KEY`           | (built into registry)            |

`base_urls()` only adds vLLM when `VLLM_BASE_URL` is set, and OpenCode URLs only
when `OPENCODE_API_KEY` is set (defaults for Zen/Go URLs exist in settings but
are not injected without a key).

```python
wrapper = LLMWrapper(
    api_keys=settings.api_keys(),
    base_urls=settings.base_urls(),
    ingestion_url=settings.ingestion_url,
)

# Chat / compare: UI sends provider + model per request
model = get_model("vllm", "qwen2.5-0.5b-instruct")
model = get_model("opencode", "kimi-k2.5")
model = get_model("opencode-go", "glm-5")
model = get_model("huggingface", "qwen2.5-0.5b-instruct")
```

**Env vars (see `.env.example`):**

```
VLLM_BASE_URL=                    # empty by default — no vLLM in compose
VLLM_API_KEY=EMPTY
VLLM_MODEL=qwen2.5-0.5b-instruct
OPENCODE_API_KEY=
OPENCODE_ZEN_BASE_URL=https://opencode.ai/zen/v1
OPENCODE_ZEN_MODEL=kimi-k2.5
OPENCODE_GO_BASE_URL=https://opencode.ai/zen/go/v1
OPENCODE_GO_MODEL=deepseek-v4-flash
OSS_PROVIDER=                     # CLI eval only: vllm | opencode | opencode-go | huggingface
OSS_MODEL=
```

**CLI eval (`eval/run_eval.py`):** `settings.resolve_oss()` cascade is
`OSS_PROVIDER` (explicit) → `opencode-go` if `OPENCODE_API_KEY` is set
(preferred — required for the agent / tool-use eval since
`supports_tools=True`) → `huggingface` if only `HUGGINGFACE_API_KEY` is set
(static-eval only, no tool support) → `ValueError`. Unknown provider ids or
missing backend config (e.g. `OSS_PROVIDER=vllm` without `VLLM_BASE_URL`)
raise before the eval loop starts.

For tests, never call real providers — use the fake provider in `tests/conftest.py`.

---

## 6. Tool calling (no MCP)

Native, in-process function calling. Tools are plain Python callables wrapped
in a `Tool` dataclass and registered at startup. The SDK already normalises
provider-agnostic tool schemas → Anthropic `tools`, OpenAI `tools`, Google
`function_declarations`. We just need a registry and an executor on top.

### 6.1 The `Tool` and `ToolRegistry`

```python
# services/api/tools/registry.py
from dataclasses import dataclass
from typing import Callable, Any
import inspect, asyncio

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict   # JSON Schema for arguments
    handler: Callable[..., Any]

    def schema(self) -> dict:
        # Matches sdk/normalize.py expectations: {name, description, parameters}
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    async def call(self, name: str, args: dict) -> Any:
        tool = self._tools[name]
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(**args)
        return await asyncio.to_thread(tool.handler, **args)
```

A single `ToolRegistry` instance lives on `app.state.tools`, populated in the
lifespan with all `builtins.py` tools. `deps.tools_registry` injects it.

### 6.2 Mock daily-assistant tools (`builtins.py`)

All five return **deterministic mock data** — no external integration. They
exist so the model has something realistic to call, so the eval can probe
tool use, and so the UI can render tool cards. Schemas are JSON Schema draft-07
because that's what all three providers accept.

| Tool | Purpose | Returns |
|---|---|---|
| `schedule_call` | book a call with someone | `{event_id, participant, when, duration_min, link}` |
| `update_calendar` | move/edit an existing event | `{event_id, when, notes, status: "updated"}` |
| `list_calendar` | list events in a date range | `[{event_id, title, when, participant}]` (3 mocks) |
| `set_reminder` | one-shot reminder | `{reminder_id, text, when, status: "scheduled"}` |
| `get_weather` | weather at a location | `{location, temp_c, condition, source: "mock"}` |

Example registration:

```python
def schedule_call(participant: str, when: str, duration_min: int = 30) -> dict:
    return {
        "event_id": f"evt_{uuid.uuid4().hex[:8]}",
        "participant": participant,
        "when": when,
        "duration_min": duration_min,
        "link": "https://meet.example/mock",
    }

registry.register(Tool(
    name="schedule_call",
    description="Schedule a call with a participant at a specific ISO-8601 time.",
    parameters={
        "type": "object",
        "properties": {
            "participant": {"type": "string", "description": "Name or email."},
            "when": {"type": "string", "description": "ISO-8601 datetime."},
            "duration_min": {"type": "integer", "minimum": 5, "maximum": 240, "default": 30},
        },
        "required": ["participant", "when"],
    },
    handler=schedule_call,
))
```

### 6.3 The chat loop with tools

`routers/chat.py` calls `LLMWrapper.stream()` with `Context.tools=registry.schemas()`.
After the stream yields `done`, inspect `event.message.tool_calls`:

```python
ctx = Context(system_prompt=SYSTEM, messages=history, tools=tools_registry.schemas())

for _ in range(MAX_TOOL_HOPS):           # cap at 5 to prevent loops
    final_msg: AssistantMessage | None = None
    async for event in wrapper.stream(model, ctx, session_id=sid, conversation_id=cid):
        if event.type == "text_delta":
            yield sse({"type": "text_delta", "delta": event.delta})
        elif event.type == "tool_call_end":
            yield sse({"type": "tool_call", "id": event.tool_call.id,
                       "name": event.tool_call.name, "args": event.tool_call.arguments})
        elif event.type == "done":
            final_msg = event.message

    if not final_msg or not final_msg.tool_calls:
        break

    # Execute every tool call in parallel, then feed results back as a new turn.
    results = await asyncio.gather(*[
        tools_registry.call(tc.name, tc.arguments) for tc in final_msg.tool_calls
    ])
    for tc, result in zip(final_msg.tool_calls, results):
        yield sse({"type": "tool_result", "id": tc.id, "result": result})

    ctx.messages.append(final_msg)
    ctx.messages.append(ToolResultMessage(results=[
        ToolResult(tool_call_id=tc.id, content=json.dumps(result))
        for tc, result in zip(final_msg.tool_calls, results)
    ]))

yield "data: [DONE]\n\n"
```

Persist the assistant message and each tool result into Postgres `messages`
with `role IN ('assistant', 'tool_result')` so the conversation can be
replayed.

### 6.4 SSE wire format (extended for tools)

```
data: {"type": "text_delta", "delta": "Booking your call now…"}\n\n
data: {"type": "tool_call", "id": "tc_1", "name": "schedule_call",
       "args": {"participant": "Ankur", "when": "2026-05-22T15:00:00Z"}}\n\n
data: {"type": "tool_result", "id": "tc_1",
       "result": {"event_id": "evt_a1b2c3d4", "link": "https://meet.example/mock"}}\n\n
data: {"type": "text_delta", "delta": "Done. The call is on the calendar."}\n\n
data: {"type": "done", "usage": {"input": 142, "output": 38, "cost_usd": 0.00021}}\n\n
data: [DONE]\n\n
```

The frontend renders `tool_call` and `tool_result` as inline cards between
assistant text bubbles.

---

## 7. Database schemas

### Postgres (relational)

```sql
CREATE TABLE sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_agent  TEXT,
    ip_hash     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE conversations (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   UUID REFERENCES sessions(id),
    title        TEXT DEFAULT '',
    provider     TEXT NOT NULL,
    model        TEXT NOT NULL,
    status       TEXT DEFAULT 'active',
    created_at   TIMESTAMPTZ DEFAULT now(),
    updated_at   TIMESTAMPTZ DEFAULT now(),
    cancelled_at TIMESTAMPTZ
);

CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,        -- user | assistant | tool_result
    content         TEXT NOT NULL,        -- JSON for tool_result; plain text otherwise
    tool_call_id    TEXT,                 -- non-null only for role='tool_result'
    token_count     INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

### ClickHouse (analytical)

```sql
CREATE TABLE inference_logs (
    trace_id            String,
    session_id          String,
    conversation_id     String,
    provider            LowCardinality(String),
    model               LowCardinality(String),
    api_protocol        LowCardinality(String),
    started_at          DateTime64(3),
    ended_at            DateTime64(3),
    latency_ms          Float64,
    input_tokens        UInt32,
    output_tokens       UInt32,
    cache_read_tokens   UInt32,
    cache_write_tokens  UInt32,
    total_tokens        UInt32,
    cost_usd            Float64,
    input_preview       String,
    output_preview      String,
    status              LowCardinality(String),
    error_message       Nullable(String),
    stop_reason         LowCardinality(String),
    stream              Bool,
    temperature         Nullable(Float32),
    max_tokens          Nullable(UInt32),
    date                Date DEFAULT toDate(started_at)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (provider, started_at)
TTL date + INTERVAL 90 DAY;
```

---

## 8. Docker Compose services

| service | image | port | purpose |
|---|---|---|---|
| postgres | `postgres:16-alpine` | 5432 | conversations, messages, sessions |
| clickhouse | `clickhouse/clickhouse-server:24-alpine` | 8123, 9000 | inference_logs |
| redis | `redis:7-alpine` | 6379 | ingestion queue |
| api | `./services/api` | 8000 | FastAPI backend |
| chatbot | `./services/chatbot` | 3000 | Next.js UI |
| prometheus | `prom/prometheus:v2.55.0` | 9090 | metrics scraping |
| grafana | `grafana/grafana:11.3.0` | 3001 | dashboards (`grafana-clickhouse-datasource` plugin) |

Startup order: `postgres + clickhouse + redis` must be healthy before
`api`. `api` runs `python scripts/run_migrations.py` before `uvicorn`.
`chatbot` depends on `api`. vLLM is **not** in the local compose — for the
public deployment it's added via `infra/deploy/docker-compose.prod.yml`.

---

## 9. Task order

Work tasks in order. Don't start the next task until the current one's tests pass.

| # | Task | Done when |
|---|---|---|
| 1 | FastAPI core (`main.py`, `config.py`, `deps.py`, `health.py`) | `uvicorn services.api.main:app` starts; `GET /health` → ok; `/metrics` returns Prom text |
| 2 | Chat router + Postgres + SSE | `pytest tests/test_chat.py` passes; manual curl streams SSE; rows land in Postgres |
| 3 | Ingestion pipeline (`/ingest/log` + Redis + worker + ClickHouse) | `pytest tests/test_ingest.py`; a chat call surfaces in ClickHouse within 2s |
| 4 | **Tool calling** (registry + executor + mock builtins + chat-loop integration) | `pytest tests/test_tools.py`; model can call `schedule_call` and the UI renders the result card |
| 5 | Next.js UI (list, chat with tool cards, dashboard, **side-by-side compare**) | `npm run dev` starts; can create/cancel/resume; `/compare` shows both models side-by-side |
| 6 | Grafana dashboards + full Compose | `docker compose up` runs all 7 services green; dashboards populate after chatting |
| 7 | Eval runner + PDF report (**Part A deliverable**) | `eval/run_eval.py` writes `results.json`; `eval/report.py` writes `docs/eval_report.pdf` |

---

## 10. Eval methodology (Task 7)

Compare **GLM-5 (OpenCode Go)** vs **Claude Sonnet 4.6** across three eval
layers — each measures a different thing, none of them double-count.

### 10.1 Three layers

| Layer | What it measures | Scorer | Files |
|---|---|---|---|
| **L1 Heuristic** | Did the response obey the shared structured system prompt? Persona signature, required H1/H2/H3 headers, banned phrases, length bounds. | Deterministic Python checks. | `eval/heuristics.py` |
| **L2 LLM-judge static** | Hallucination / bias / safety on plain-text Q&A across 30 prompts (10 factual / 10 adversarial / 10 bias). | Claude Sonnet 4.6 (0–5 per axis). | `eval/prompts.py`, `eval/judge.py` |
| **L3 LLM-judge agent** | 5-axis trajectory scoring across 20 tool-use prompts (single-tool / multi-tool / no-tool / ambiguous / adversarial-tool). Axes: tool_selection, argument_correctness, task_completion, output_grounding, safety_with_tools. | Claude Sonnet 4.6 (0–5 per axis). | `eval/agent_prompts.py`, `eval/agent_runner.py`, `eval/agent_judge.py` |

Both models receive the **same** structured system prompt (`AGENT_SYSTEM_PROMPT`
in `eval/heuristics.py`); the judges are told to ignore formatting so L1 and
L2/L3 don't overlap.

### 10.2 Outputs

- `eval/results.json` — `{metadata, static_results, agent_results}`. Each
  row carries both models' responses (or full trajectories), heuristic
  pass/fail, judge scores, latency, tokens, cost.
- `docs/eval_report.pdf` — 5 pages:
  1. Summary cards + static radar + heuristic compliance + verdict
  2. Static per-category bars + latency/cost/jailbreak table
  3. Notable static failures (worst 5 per model)
  4. Agent 5-axis cards + 5-axis radar + hop/error summary
  5. Agent per-category bars + worst-3 trajectories per model

### 10.3 UI surface

`http://localhost:3000/compare?view=benchmark` opens a master-detail
browser: list of all 50 prompts on the left, click any prompt to drill into
both models' responses (or trajectories), heuristic checklist, judge
rationale, latency and cost. A **Run eval** button kicks off the eval as a
background task via `POST /eval/run` and polls `GET /eval/runs/latest`
every 2 s.

### 10.4 Running

```bash
python eval/run_eval.py             # both sections (default)
python eval/run_eval.py --static    # just the 30
python eval/run_eval.py --agent     # just the 20 tool-use trajectories
python eval/run_eval.py --all --limit 3   # smoke run
python eval/report.py               # regenerate the PDF from results.json
```

The full rubrics live in `eval/judge.py` and `eval/agent_judge.py`. Don't
rewrite them from memory — read those files.

---

## 11. SSE, sessions, frontend conventions

- Session: frontend stores a UUID in `localStorage["ollive_session_id"]`,
  sends it on every request as `X-Session-ID`. If missing, API generates one
  and returns it in the response header.
- Streaming: backend yields each `StreamEvent` from `LLMWrapper.stream()` as
  `data: <json>\n\n`. Frontend uses `fetch` + `ReadableStream` (not
  `EventSource`, because we need POST).
- `InferenceLog` is sent to ingestion by the wrapper automatically — the route
  handler does not touch logging.
- TypeScript strict, no `any`, Zod for runtime validation, Tailwind only.

---

## 12. Running locally

```bash
cp .env.example .env
# fill: ANTHROPIC_API_KEY, HUGGINGFACE_API_KEY (eval OSS default), optional VLLM_BASE_URL

docker compose up

# http://localhost:3000   chatbot UI (chat, dashboard, /compare)
# http://localhost:8000/docs   FastAPI swagger
# http://localhost:3001   Grafana (admin/admin)
# http://localhost:9090   Prometheus
```

Tests run on the host, not in Docker:

```bash
pip install -r requirements.txt
pytest -v
```

Eval (the deliverable):

```bash
python eval/run_eval.py     # → eval/results.json
python eval/report.py       # → docs/eval_report.pdf + docs/eval_cost_latency.md
```

### Public deployment (Azure VM)

A single `Standard_NC4as_T4_v3` VM hosts the whole stack — FastAPI, Next.js,
Postgres, ClickHouse, Redis, Prometheus, Grafana, *and* vLLM serving
Qwen2.5-1.5B-Instruct (registered with `supports_tools=True` so the L3 agent
eval can drive it). Caddy fronts everything: TLS via Let's Encrypt, bearer-
token auth on `/v1/*`, reverse proxies `/api/*`, `/grafana/*`, `/` → chatbot.

```bash
az login
./infra/deploy/deploy.sh --anthropic-key "$ANTHROPIC_API_KEY"
# prints https://ollive-XXX.eastus.cloudapp.azure.com + bearer token
```

Full runbook in `infra/deploy/README.md`. The prod overlay
(`infra/deploy/docker-compose.prod.yml`) is composed on top of the root
`docker-compose.yml` — it adds the `vllm` and `caddy` services, drops
`--reload` from the API, and rebinds Postgres/ClickHouse/Redis/Prometheus
to `127.0.0.1` for defense in depth.

Cost: ~$0.526/hr on-demand (~$380/mo). vLLM uses ~4 GB of the 16 GB T4,
leaving headroom for batched eval traffic.

### Guardrails

`services/api/guardrails/` is the deterministic safety layer that sits in
front of `LLMWrapper`. Three pure functions:

- `check_input(text)` — length cap, control chars, prompt-injection regex
- `check_output(text)` — banned chat-template tokens, destructive commands
- `validate_tool_args(name, args)` — ISO-8601 `when`, bounded `duration_min`

`routers/chat.py` calls them inline and emits `{"type": "guardrail_block"}`
SSE events on block. The L2 safety axis in `eval/judge.py` is the
backstop that measures whether they actually work.

---

## 13. What not to do

- Don't modify `sdk/` or `tests/test_sdk.py`.
- Don't call provider SDKs directly outside `sdk/providers/`.
- Don't write synchronous DB calls in async FastAPI routes.
- Don't store raw IPs or unredacted PII anywhere.
- Don't block the chat path on logging — ingestion is fire-and-forget.
- Don't introduce MCP, LangChain, or a tool-calling framework. The registry
  in §6 is the whole thing.
- Don't run real LLM calls in tests — use the fake provider in `conftest.py`.
- Don't commit `.env`.
