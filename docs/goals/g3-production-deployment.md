# Goal G3: Production Deployment — Gateway, Replica Fleet, Batch, Observability

Status: **Accepted** (2026-07-02). Drives M7.
CPU-verifiable gates run against mock replicas; GPU bring-up slots in via
`docs/gpu-runbook.md` §9 once M2/M5/M6 gates are green.

## 1. Goal

Turn the Kairyu library into a deployable product for an on-prem-DC-centric
topology: a stateless CPU **gateway tier** (L3 server + L2 orchestration +
`ReplicaPool` over remote replicas) in front of N **GPU replica nodes** (each
an independent Kairyu OpenAI-compatible server), with a thin managed cloud
front (WAF / L7 LB / TLS) reaching the DC over a private interconnect.
Everything Kairyu-side must be provable on CPU with mock replicas; nothing in
this goal requires GPU hardware.

## 2. Deployment contract

- Gateway and replica nodes run the **same artifact**: one container image,
  one `kairyu serve <config.yaml>` entrypoint; the config decides the role.
- Replicas are static, declared in the gateway's `DeploymentSpec`; no service
  discovery, no elasticity (G2 §6 lineage). Fleet orchestration is
  systemd + docker compose; Kubernetes is a documented migration path, not a
  dependency.
- The WAF, edge LB, TLS termination, and DC–cloud interconnect are managed /
  network concerns documented in `docs/deployment.md`; Kairyu ships
  defense-in-depth behind them (API-key auth, concurrency guard).

## 3. Acceptance gates

| Gate | Criterion | Where proven |
|---|---|---|
| C1 | Gateway serves `/v1/chat/completions` (non-stream + SSE) through a `ReplicaPool` of remote `openai`-backend replicas; requests carrying the same session (`user` field or `X-Session-ID`) land on the same replica (observed via `kairyu_pool_decisions_total{reason="session_affinity"}`) | `tests/server/test_serve_builder.py` + compose smoke |
| C2 | Killing one replica yields zero 5xx on subsequent requests (health ejection) and the background prober restores it after recovery without operator action | compose smoke kill/recover drill |
| C3 | The compose topology (1 gateway + 3 mock replicas) passes the smoke script in CI from a cold `docker compose up` | `.github/workflows/ci.yml` `compose-smoke` job |
| C4 | A `/v1/batches` job over the pool completes while interactive requests stay admitted: the batch worker's concurrency cap is enforced below the server's global cap | `tests/server/test_batches.py` |
| C5 | With auth enabled, requests without a valid API key get 401 on `/v1/*`; `/health` and `/readyz` stay open; node-to-node replica calls work keyless | `tests/server/test_auth.py` |
| C6 | `/metrics` exposes request counts/latency histograms, per-replica outstanding/health, and pool decision counts in Prometheus text format; `/readyz` reflects pool health (≥1 healthy replica) | `tests/server/test_health_metrics.py` |
| C7 | Rolling model update procedure (drain → restart with new weights → `/readyz` → next) is documented and rehearsed against mock replicas with zero failed requests | `docs/deployment.md` + compose drill |

## 4. Non-goals

- Building a WAF, rate limiter beyond a global concurrency guard, or TLS
  termination — the managed edge owns these (documented split of duties).
- Autoscaling, elasticity, live migration, dynamic replica registration
  (unchanged from G2 §6).
- A shared response/Redis cache tier — the cache layer is per-replica radix
  KV plus `ReplicaPool` session affinity (m7 D6 records the revisit trigger).
- Hot model swap inside a running replica (rolling restart is the procedure).
- Multi-region / multi-cluster federation.

## 5. Amendment to G2 scope

G2 §6's "exactly 2 nodes" bounds the TP/PP **coherence domain** (collectives,
KV transfer plane), not the number of independent DP replica endpoints behind
`ReplicaPool`. M7 serves N replica endpoints; each endpoint internally uses
whatever M5/M6 layout its node is validated for. Recorded in PROGRESS.md as an
amendment entry.
