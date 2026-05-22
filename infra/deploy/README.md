# Single-VM public deployment

Deploys the whole stack — FastAPI, Next.js chatbot, Postgres, ClickHouse,
Redis, Prometheus, Grafana, **and a self-hosted Ollama OSS endpoint** — on
one Azure VM, behind one Caddy HTTPS endpoint.

OSS inference runs locally on the VM via **Ollama** serving
`qwen2.5:1.5b` (Q4_K_M, ~1 GB), exposed publicly at `https://<host>/v1/*`
behind a bearer-token gate. OpenCode-Go stays available as an optional
secondary provider when `OPENCODE_API_KEY` is supplied. The CPU VM
(`Standard_E8s_v5`, 8 vCPU / 64 GB / no GPU) is the right shape for a
quantized 1.5B model — expect 8–15 tok/s.

## Topology

```
                 ┌───────────────────────────────────────────────┐
 Internet ──443──► caddy  ── TLS termination ──────────────────┤
                 │   ├─ /api/*      → api:8000                  │
                 │   ├─ /grafana/*  → grafana:3000              │
                 │   ├─ /v1/*       → ollama:11434 (bearer)     │
                 │   ├─ /health     → api:8000/health           │
                 │   └─ /           → chatbot:3000              │
                 │                                              │
                 │  ollama (qwen2.5:1.5b) on docker network     │
                 │  postgres / clickhouse / redis / prometheus  │
                 │  bound to 127.0.0.1 only (defense in depth)  │
                 └──────────────────────────────────────────────┘
```

## SKU and cost

`Standard_E8s_v5` — 8 vCPU / 64 GB RAM (default). Comfortable headroom; the
stack itself uses ~12 GB.

| Mode         | Hourly | Monthly |
|--------------|--------|---------|
| On-demand    | ~$0.50 | ~$360   |

Plus ~$0.17/day for an ACR Basic registry. Override `--vm-size` for a different
shape.

## Prerequisites

```bash
az login
gh auth login                                # for CI secrets
export ANTHROPIC_API_KEY=sk-ant-...
# optional — OpenCode-Go as a secondary OSS provider
export OPENCODE_API_KEY=oc-...
# optional — HF inference as a static-eval fallback
export HUGGINGFACE_API_KEY=hf-...
```

## Deploy

From the repo root:

```bash
./infra/deploy/deploy.sh \
    --anthropic-key "$ANTHROPIC_API_KEY" \
    [--opencode-key  "$OPENCODE_API_KEY"]   # optional
    [--ollama-api-key "$(openssl rand -hex 32)"]   # auto-generated if omitted
```

`--ollama-api-key` is the bearer Caddy enforces on the public `/v1/*` path.
Skip it and the script generates one; it ends up in
`/opt/ollive/credentials.txt` on the VM.

The script:

1. Creates resource group `ollive-rg` in `eastus`.
2. Creates Azure Container Registry (auto-named `ollivacrXXXX`).
3. Generates a CI deploy SSH keypair at `infra/deploy/.ci_deploy_key{,.pub}`.
4. Creates VM `ollive-vm` (`Standard_E8s_v5`) with cloud-init that:
   - Installs Docker + Compose
   - Adds the CI deploy pubkey to `azureuser`'s `authorized_keys`
   - Clones the repo, writes `.env`, brings the stack up
5. Opens NSG ports 80, 443.

First boot takes ~5–10 min, plus ~60s on top while Ollama pulls
`qwen2.5:1.5b` (~1 GB) into its model cache. Watch:

```bash
ssh azureuser@<public-ip> 'sudo tail -f /var/log/cloud-init-output.log'
```

When ready, retrieve secrets:

```bash
ssh azureuser@<public-ip> 'sudo cat /opt/ollive/credentials.txt'
```

That prints the public URL, Grafana admin password, and the ACR registry name.

## CI/CD

`.github/workflows/deploy.yml` runs on every push to `main` (and via
`workflow_dispatch`).

Stages:

1. **build** — builds API + chatbot Docker images, pushes both to ACR
   tagged with the commit SHA and `:latest`.
2. **deploy** — SSHes into the VM, pulls the new images via the
   `infra/deploy/docker-compose.ci.yml` overlay, and runs
   `docker compose up -d`. Old images are pruned afterward.

### Required GitHub secrets

The `deploy.sh` script prints the exact `gh secret set` commands at the end.
Copy-paste them once.

| Secret              | Source                                       |
|---------------------|----------------------------------------------|
| `ACR_LOGIN_SERVER`  | e.g. `ollivacr1234.azurecr.io`               |
| `ACR_USERNAME`      | `az acr credential show --name <acr>`        |
| `ACR_PASSWORD`      | same command                                 |
| `DEPLOY_HOST`       | the FQDN printed by `deploy.sh`              |
| `PUBLIC_HOSTNAME`   | same FQDN — baked into chatbot at build time |
| `DEPLOY_SSH_KEY`    | private half of `infra/deploy/.ci_deploy_key`|

Once those are set, push to `main` triggers a full build + rollout in
~5–6 minutes.

## Smoke-test the public endpoint

```bash
HOST=<fqdn-from-deploy.sh>
BEARER=$(ssh azureuser@<public-ip> 'sudo grep "Ollama /v1 bearer" /opt/ollive/credentials.txt | awk "{print \$NF}"')

curl https://$HOST/health                                   # → {"status":"ok"}
curl -H "Authorization: Bearer $BEARER" https://$HOST/v1/models   # lists qwen2.5:1.5b
curl https://$HOST/v1/models                                # → 401 (bearer enforced)
open https://$HOST/              # Next.js chatbot
open https://$HOST/grafana/      # dashboards
```

## Push code updates

After CI/CD is set up, any commit on `main` deploys automatically. For an
emergency hotfix without going through CI:

```bash
ssh azureuser@<public-ip>
sudo -i
cd /opt/ollive/repo
git pull
docker compose -f docker-compose.yml -f infra/deploy/docker-compose.prod.yml \
    up -d --build
```

## Run the eval against the public endpoint

OSS inference runs on the VM. Point the eval at the public `/v1` endpoint:

```bash
export ANTHROPIC_API_KEY=...
export OSS_PROVIDER=ollama
export OSS_MODEL=qwen2.5:1.5b
export OLLAMA_BASE_URL=https://<fqdn>/v1
export OLLAMA_API_KEY=<bearer from credentials.txt>

python eval/run_eval.py --all --limit 3   # smoke
python eval/run_eval.py                   # full
python eval/report.py                     # writes docs/eval_report.pdf + docs/eval_cost_latency.md
```

The same eval can also run inside the VM against `http://ollama:11434/v1`
without the bearer if you'd rather skip the public round-trip.

## Tear down

```bash
az group delete --name ollive-rg --yes --no-wait
```

The resource group is the unit of cleanup — deleting it removes the VM, NSG,
public IP, OS disk, DNS reservation, and ACR.
