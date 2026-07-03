# M7 Design: Productionization — Serve CLI, Gateway Wiring, Batch, Observability

Status: **Implemented — CPU half** (2026-07-02). All D1–D8 landed with tests;
gates C1–C4/C5–C6 proven against mocks (C2/C3 additionally by the CI compose
smoke drill); GPU bring-up is `docs/gpu-runbook.md` §9. Human sign-off pending.
**Amended 2026-07-03** (roadmap, goal G5): at thousands-of-GPU fleet scale D2's own
revisit triggers fire — k8s is adopted as the machine layer (G5 F1); D6 gains
prefix-aware placement then KV tiering as its recorded revisit path (G5 F2/F4);
D8's no-OTel stance flips now that a tracing consumer exists (G5 F1d). Everything
shipped here remains the per-node baseline; see `docs/roadmap.md` §5.
Milestone: M7
Date: 2026-07-02
Depends on: Goal G3 (`docs/goals/g3-production-deployment.md`, gates C1–C7);
M1 server/orchestration; m5 D4 (`ReplicaPool`); m6 D2 (`openai` remote-replica
backend). Independent of GPU hardware — every deliverable is CPU-verifiable.

## 1. Goal

Package the existing in-process components into a deployable product for an
on-prem-DC topology: one `kairyu serve` entrypoint that builds either a
**gateway** (server + orchestrator + `ReplicaPool` of remote replicas, with
auth, health, metrics, batch) or a **replica** (server + local engine) from a
YAML `DeploymentSpec`, shipped as one container image, composed with
systemd + docker compose, fronted by managed cloud WAF/LB over a private
interconnect (doc-only).

## 2. Key design decisions and rationale

### D1 — Topology: thin managed cloud front, stateless DC gateway tier, N replica nodes

```
Internet → [managed WAF + L7 LB + TLS]  (cloud, doc-only)
         → [private interconnect]       (Direct Connect class, doc-only)
         → [gateway ×2, CPU, stateless] kairyu serve gateway.yaml
         → [replica ×N, GPU]            kairyu serve replica.yaml
```

The gateway is `create_app` + `Orchestrator` + `ReplicaPool` whose members are
`openai`-backend clients pointing at replica nodes (m6 D2's remote-replica
path, already pooled/SSE/keyless-capable). Replica nodes run the same server
with a local engine (`mock` on CPU, `kairyu`/`vllm` on GPU). Gateways hold no
request state (the batch store is the one exception — D7), so HA is "run two
behind the edge LB".

### D2 — No Kubernetes in M7; systemd + docker compose; containerize everything

The fleet is small and static by design lineage (static `ClusterSpec`, no
elasticity, no Ray — g2 §6, m6 D1). k8s's core value (bin-packing,
rescheduling, elasticity) is inert on pet GPU nodes, while its cost
(control plane, upgrades, CNI, NVIDIA GPU operator, etcd) is exactly the ops
burden flagged in the requirements. Nomad adds a niche dependency without
removing the burden. Therefore: one image, compose files per node, systemd
units wrapping `docker compose up`. Everything is containerized now so a
later k3s/RKE2 adoption is manifest-writing, not re-architecture. Revisit
triggers (documented in `docs/deployment.md`): fleet > ~5–8 nodes, multiple
teams sharing the cluster, rolling deploys becoming weekly toil.

### D3 — `DeploymentSpec` is new; `ClusterSpec` is not extended

`ClusterSpec` (m6 D1) encodes the 2-node TP/PP/P-D coherence-domain
validation; a serving deployment needs a different vocabulary: served models →
backend factory kwargs (via `kairyu.engine.registry.create_backend`), pool
sections with N remote members, server settings, optional orchestrator (reuse
`kairyu.dsl.loader`). Merging the two would couple serving fleet size to the
G2 2-node cap. They compose instead: a replica node's intra-node GPU layout
may reference a ClusterSpec file; the gateway's DeploymentSpec only knows the
replica's endpoint. Schema (pydantic, `kairyu/deploy/spec.py`):

```yaml
server: { host, port, api_keys_env, max_concurrency, metrics: bool }
engines:            # name -> single backend
  small: { backend: mock, options: {...} }
pools:              # name -> ReplicaPool of backends
  llama-70b:
    replicas:
      - { backend: openai, options: { base_url: "http://gpu-0:8000/v1", model: "...", api_key_env: null } }
      - { backend: openai, options: { base_url: "http://gpu-1:8000/v1", model: "...", api_key_env: null } }
    unhealthy_after: 3
    queue_depth_threshold: 8
    probe_interval_s: 5.0
orchestrator: { spec: agent_pool.yaml }   # optional, reuses DSL loader
batch: { data_dir: /var/lib/kairyu/batch, max_concurrency: 4 }  # optional
```

### D4 — Health/readiness/metrics live in the serve layer; the pool stays passive

`/health` = process liveness. `/readyz` = every engine constructed and every
pool has ≥1 healthy replica. `/metrics` = Prometheus text format
(`prometheus-client`, pure-Python — D8). The background **prober** is a
FastAPI-lifespan task in `kairyu/deploy/prober.py`: for each ejected replica
it GETs the replica's `/health` (replicas run the same server, so the
endpoint exists by construction) and on 200 calls the existing
`ReplicaPool.probe(index)`. m5 D4's "no background tasks" is preserved in
letter and spirit: the pool remains pure hashing; the task lives outside it
and uses only public accessors. `ReplicaPool` gains read-only accessors
(`healthy`, `replica_count`, decision counters) — additive, no behavior
change.

### D5 — Auth: managed WAF at the edge; static API keys at the gateway; keyless node-to-node

The edge (WAF/LB) owns DDoS, per-client rate limits, TLS. The gateway ships
defense-in-depth: optional static API keys sourced from an env var
(comma-separated, constant-time compare), exempting `/health` and `/readyz`;
plus a global concurrency guard (asyncio semaphore → 429 + `Retry-After`).
Replica nodes inside the DC accept keyless traffic (m6 D2's
`api_key_env=None`) or a shared key — deployment guide shows both. Kairyu
builds no WAF and no per-key rate accounting.

### D6 — The cache layer is per-replica radix KV + pool session affinity; no Redis

Session affinity (rendezvous hashing on `cache_hint.session_id`) already
keeps a session's turns on the replica holding its warm KV prefix — that IS
the cache architecture. A shared response cache is rejected: sampled outputs
(temperature > 0) make exact-reuse rare, and a cache tier adds the ops
dependency this milestone minimizes. **Amendment-grade gap fixed here**: the
HTTP path never set `cache_hint`, so external traffic got no affinity at all.
M7 maps the OpenAI `user` field and/or `X-Session-ID` header to
`CacheHint(session_id=...)` in `app.py`. Revisit trigger for a response
cache: telemetry showing a material rate of byte-identical requests at
temperature 0.

### D7 — Batch: minimal OpenAI-compatible `/v1/files` + `/v1/batches`, filesystem-backed

An in-gateway asyncio worker drains queued batch jobs through the same served
engines/pools under its own concurrency cap (strictly below the server's
global cap, so interactive latency is protected — gate C4). Storage is a
data-dir on disk (JSONL input/output/error files + JSON job state); no
Redis/Celery/queue at this node count. Single-gateway scope: with two
gateways, pin batch traffic to one or share the data dir (documented).
Restart recovery marks `in_progress` jobs `failed` — honest and simple.

### D8 — Observability: `prometheus-client` + stdlib JSON logs; no OTel

Metrics: `kairyu_requests_total{model,code}`,
`kairyu_request_duration_seconds` (histogram),
`kairyu_replica_outstanding{pool,replica}`,
`kairyu_replica_healthy{pool,replica}`,
`kairyu_pool_decisions_total{pool,reason}` (the affinity-hit-rate signal),
`kairyu_batch_jobs_total{state}`. Logging: stdlib `logging` with a JSON
formatter and request-ID field — consistent with the existing JSONL
router-log style; no structlog/OTel dependency until a tracing consumer
exists.

## 3. What M7 does not include

WAF / rate limiting beyond the concurrency guard / TLS (edge-owned, D5);
autoscaling, elasticity, dynamic replica registration (G2 §6 lineage); hot
model swap (rolling restart per `docs/deployment.md`); Redis / response cache
(D6); Kubernetes manifests as tested artifacts (doc appendix only, D2);
multi-region. GPU bring-up of the topology is `docs/gpu-runbook.md` §9.

## 4. Amendments to earlier decisions

1. **g2 §6 / m6 §3 "exactly 2 nodes"** — clarified: binds the TP/PP coherence
   domain, not the count of independent DP replica endpoints behind
   `ReplicaPool` (G3 §5). Logged in PROGRESS.md.
2. **m5 D4 "no background tasks"** — unchanged for the pool itself; the
   prober is a serve-layer lifespan task calling `probe()` (D4).

## 5. Verification

- Phases 1/2/4: pytest via ASGI transport (existing pattern), ruff, 80%
  coverage gate. Prober/worker tested with injected transports and intervals
  (no sleeps).
- Phase 3: CI `compose-smoke` job — cold `docker compose up`, `/readyz`
  poll, non-stream + SSE completion, affinity assertion via metrics, replica
  kill/recover drill (gates C1–C3).
- G3 gate table checked off in the goals doc as phases land.
