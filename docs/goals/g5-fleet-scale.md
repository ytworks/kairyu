# Goal G5: Fleet-Scale Control Plane — Elasticity, KV-Aware Routing, P/D Pools, Tenancy (Roadmap Track F)

Status: Goal defined (2026-07-03). Supersedes-by-amendment: m7 D2 (no-k8s),
m5 D4/m7 D6 (session-hash affinity, per-replica cache only), m6 D1 (static-only
topology; the no-Ray decision stands), ClusterSpec 2-node cap, and the
G2 §6 / G3 §4 autoscaling/elasticity/dynamic-registration non-goals. All
amendments recorded in PROGRESS.md; original entries untouched.
Depends on: G3 (gates C1–C7 are the per-node baseline this goal scales out);
phases F3+ additionally on Track E hardware phases per `docs/roadmap.md` §4.
Date: 2026-07-03

## 1. Goal

Operate thousands of GPUs — heterogeneous across the hardware profiles of
`docs/roadmap.md` §2 (NVLink-HBM and PCIe-GDDR nodes, A100 and later) — as one
serving fleet: replicas join/leave/drain
without gateway restarts, requests are placed where their KV prefix already lives,
prefill/decode pools are managed across racks, KV capacity extends over DRAM/NVMe,
and tenants get quotas, SLO-based admission, and metered usage. Kubernetes owns the
machine layer (pods, restarts, rollouts); **Kairyu keeps the model-aware brain**
(routing, P:D ratios, admission, autoscaling decisions) — adopting llm-d/Dynamo
patterns without replacing the differentiating L2.

SLO vocabulary used by every gate below:

- **goodput@SLO** — completed requests/s meeting a stated TTFT/TPOT SLO.
- **prefix-hit-rate** — fraction of prompt tokens served from radix KV (engine truth,
  aggregated fleet-wide).
- **placement p99** — gateway time from request receipt to replica selection.
- **scale-up latency** — replica-count change decision → first token served by the
  new replica.

## 2. Deployment contract (evolves G3 §2)

- One artifact, one `kairyu serve` entrypoint, config decides the role — unchanged.
- `DeploymentSpec` pools gain `discovery: static | k8s-endpoints | register`;
  **static remains the default** so every existing test, example, and compose file
  keeps working unchanged.
- CPU-first discipline: F1/F2 gates run against CPU-mock fleets in CI (kind cluster);
  GPU validation slots in per `docs/gpu-runbook.md`.

## 3. Acceptance gates

### Stage F1 — Elastic control plane (CPU-mock)

| Gate | Target | Where proven |
|---|---|---|
| F1a | ReplicaPool dynamic membership: kind cluster, 1 gateway + 200 mock replicas, 10%/min churn for 10 min → zero 5xx, placement p99 <10 ms | kind CI job |
| F1b | Rolling restart of 100 mock replicas via `kubectl rollout` + `/drain` → zero failed requests, no operator action (C7 lineage, automated) | kind CI job |
| F1c | 3 gateways behind an LB with consistent-hash session partitioning pass the C1 affinity assertion; batch jobs complete with the shared `BatchStore` | kind CI job |
| F1d | One request produces one end-to-end OTel trace (gateway route → pool place → replica call); Conductor runs show per-stage spans | trace fixture test |

### Stage F2 — KV-aware routing (CPU-first, then 4–8 GPU testbed)

| Gate | Target | Where proven |
|---|---|---|
| F2a | Prefix-trie scorer: on a shared-prefix trace against mock replicas with simulated cache state, prefix-hit-rate ≥2× the session-hashing baseline, no goodput loss on a uniform trace, placement p99 <10 ms at 500 replicas | CPU bench |
| F2b | RadixKV KV-event index: staleness <500 ms under churn; router degrades gracefully to the approximate trie when the event stream dies | CPU test + chaos fixture |
| F2c | Real-engine validation: TTFT p95 reduction ≥30% on a multi-turn+RAG trace vs session-hashing, 4–8 GPUs | GPU testbed |
| F2d | Placement decisions land in the JSONL decision log (`prefix_match` reason) and feed `learning/dataset.py`; bandit-tuned α/β beats hand-tuned on a replayed trace | CPU bench |

### Stage F3 — NIC KV transfer + P/D pools (needs RDMA hardware)

| Gate | Target | Where proven |
|---|---|---|
| F3a | Transport bake-off (NCCL-p2p-staging vs UCX/RDMA vs **NIXL**) on the real sharded fragment layout: winner sustains ≥70% of measured NIC line rate at ≥64-page batches (B2's ≤8 µs/token budget restated against measured, not nominal, rate) | `bench/kv_transfer_bench.py` |
| F3b | Cross-node P-D through pool pairing: TTFT p50 inflation ≤20% vs colocated (B3 carried); rack-locality respected in pairing decisions (logged) | GPU bench |
| F3c | Mixed long-prefill/decode workload on the 70B tier: goodput@SLO ≥ +25% vs best colocated config; P:D planner v0 re-splits pools from SLO telemetry without restarts | GPU bench |
| F3d | ClusterSpec cap raised to 8; `kairyu.launch` brings up a multi-node coherence domain via k8s pod-group rendezvous; the group registers as ONE ReplicaPool endpoint and passes the C2 kill/recover drill | kind + GPU drill |

### Stage F4 — KV tiering

| Gate | Target | Where proven |
|---|---|---|
| F4a | DRAM offload: restore-from-DRAM beats recompute above a measured prefix-length crossover; the crossover is published, not assumed | GPU bench |
| F4b | Agentic multi-turn trace with tiering on: fleet prefix-hit-rate gain reported; TPOT p99 unregressed (offload work stays off the decode critical path) | GPU bench |
| F4c | Global-pool decision doc: F2's telemetry quantifies cross-replica duplicate-prefix mass; buy (Mooncake/LMCache) vs build (KVTransport extension) decided with data — m7 D6's revisit trigger honored | decision doc |

### Stage F5 — Tenancy, SLO admission, autoscaling

| Gate | Target | Where proven |
|---|---|---|
| F5a | 2× overload: interactive TTFT p99 SLO holds while the batch tier absorbs residual capacity (priority classes flow gateway → replica scheduler admission) | CPU-mock bench |
| F5b | Tenant isolation: a tenant at 10× its quota cannot degrade another tenant's p99 (token-bucket + admission, not just 429s) | CPU-mock test |
| F5c | SLO-based early rejection: at saturation, predicted-violation requests are shed/deferred-to-batch; goodput@SLO ≥ queue-and-hope baseline | CPU-mock bench |
| F5d | Autoscaler: 0→50 replicas of the 14B model in ≤5 min (weight pre-staging measured separately); scale decisions logged with their goodput/queue/KV-utilization inputs | GPU drill |
| F5e | Usage metering reconciles with request logs to <0.1% on a replayed trace; per-tenant cached-token counts exported (feeds G6 pricing signals) | CPU test |

## 4. Non-goals

- Multi-region / multi-cluster federation (single-DC fleet).
- Building a WAF/TLS edge (G3 D5 split of duties stands).
- Live migration of in-flight requests between replicas.
- A bespoke cluster scheduler — k8s is adopted, not rebuilt; Ray remains rejected.
- Adopting llm-d/Dynamo wholesale (patterns yes, replacement of L2 no).

## 5. Seams (informative, non-binding)

- `ReplicaPool` stays an `EngineBackend`; dynamic membership and prefix scoring land
  inside `orchestration/replica.py` without changing the pool-as-backend contract.
- `DeploymentSpec` (`deploy/spec.py`) carries discovery modes, topology labels,
  tenancy sections; `ClusterSpec` stays the coherence-domain config (G3 §5 principle).
- KV events are emitted by `radix_kv.py` (it already tracks block identity for the
  radix tree — events are an additive observer, not a KV redesign).
- The prober (`deploy/prober.py`) becomes the registry/reconciler seam next to k8s
  probes; pool-side ejection stays (faster than kubelet, works in compose).
- Metering/quotas extend `entrypoints/server/middleware.py` + `settings.py`; the
  ledger reuses the `batch/store.py` atomic-file pattern.

## 6. Evidence and reporting rules

G2 §8 rules carry forward. Fleet gates additionally record: replica count, churn
rate, k8s/kind versions, and the discovery mode in the results file. Chaos results
(F1a/F2b) publish the kill schedule alongside the numbers.

## 7. Human sign-off checklist (blocking)

- [ ] Fleet design doc(s) written and design-reviewed (amendments applied)
- [ ] F1–F2 gates green in CI (CPU-mock)
- [ ] F3–F5 gates green with results files pushed
