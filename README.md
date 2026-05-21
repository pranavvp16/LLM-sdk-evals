# Ollive Inference Platform

Multi-provider LLM chatbot + inference logging pipeline. Submission for the
Ollive AI assignment (`work@ollive.ai`).

> **Source of truth:** read [`CLAUDE.md`](./CLAUDE.md) first. It is the single
> authoritative spec — architecture, database schemas, tool-calling design,
> every task, and the evaluation rubric.

## Quick start

```bash
cp .env.example .env
# fill in at least ANTHROPIC_API_KEY

docker compose up
```

| Service       | URL                              |
|---------------|----------------------------------|
| Chatbot UI    | http://localhost:3000            |
| FastAPI docs  | http://localhost:8000/docs       |
| Grafana       | http://localhost:3001 (admin/admin) |
| Prometheus    | http://localhost:9090            |

## Layout

```
sdk/                  Multi-provider LLM wrapper (DONE — do not modify)
services/api/         FastAPI backend + ingestion worker
services/chatbot/     Next.js 14 chatbot UI
eval/                 LLM-as-judge evaluation runner + PDF report
infra/                Prometheus + Grafana provisioning
scripts/              run_migrations.py, seed_db.py
tests/                pytest-asyncio
```

## Tests

```bash
pip install -r requirements.txt
pytest -v
```

`tests/test_sdk.py` is already passing (18/18). Chat and ingestion tests are
filled in as the corresponding tasks land.

## Eval (Part A deliverable)

```bash
python eval/run_eval.py     # writes eval/results.json
python eval/report.py       # writes docs/eval_report.pdf
```
