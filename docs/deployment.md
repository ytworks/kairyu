# Deployment Guide — On-Prem DC with a Managed Cloud Front

How to run Kairyu as a product: one container image, one `kairyu serve`
entrypoint, an on-prem GPU fleet behind a stateless gateway tier, and a thin
managed cloud edge. Design rationale: `docs/design/m7-productionization.md`;
acceptance gates: `docs/goals/g3-production-deployment.md`.

## 1. Topology

```
Internet
   │
Cloud edge (managed; no Kairyu code here)
   ├─ WAF            — DDoS, per-client rate limits, bot filtering
   ├─ L7 LB          — TLS termination, health-checked routing to gateways
   │
Private interconnect (Direct Connect / ExpressRoute class; IPsec VPN fallback)
   │
DC gateway tier — 2× CPU nodes, stateless          kairyu serve gateway.yaml
   ├─ OpenAI-compatible API + orchestration (Router/Conductor/MoA)
   ├─ ReplicaPool per served model → remote GPU replicas
   ├─ /health /readyz /metrics, API keys, concurrency guard, batch worker
   │
DC GPU replica tier — N nodes                      kairyu serve replica.yaml
   └─ each node: same image, local engine (kairyu / vllm backend),
      internal TP / P-D / PP layout per docs/gpu-runbook.md §6–7
```

Traffic crossing the interconnect is completions requests/responses only.
KV pages, TP collectives, and P-D transfers never leave the DC fabric
(IB/RoCE stays node-to-node inside the DC).

## 2. Division of security duties

| Concern | Owner | Kairyu's part |
|---|---|---|
| DDoS, bot filtering, per-client rate limits | Cloud WAF | — |
| TLS termination, certificates | Cloud LB (or DC reverse proxy) | serves plain HTTP behind it |
| Client authentication | Gateway | `server.api_keys_env` (static keys, constant-time compare) |
| Process overload | Gateway | `server.max_concurrency` → 429 + Retry-After |
| Node-to-node auth inside the DC | Deployment choice | keyless (`api_key_env: null`) or a shared key env var |
| Audit trail | Gateway | JSON access log with `X-Request-ID`, JSONL router decision log |

## 3. Node setup (systemd + docker compose — design m7 D2)

Kubernetes is deliberately not required: the fleet is small and static, GPU
nodes are pinned to hardware, and the design lineage excludes elasticity
(g2 §6). Everything is containerized, so adopting k3s/RKE2 later is
manifest-writing, not re-architecture. Revisit k8s when any of these hold:
the fleet grows past ~5–8 nodes, multiple teams deploy onto the cluster, or
rolling deploys become weekly toil.

Per node: install Docker, drop the compose file + config, and wrap it in a
systemd unit so the stack survives reboots:

```ini
# /etc/systemd/system/kairyu.service
[Unit]
Description=Kairyu node (gateway or replica; config decides)
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/kairyu
ExecStart=/usr/bin/docker compose up -d --wait
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

GPU replica nodes additionally need the NVIDIA container toolkit and a
`deploy.resources.reservations.devices` (or `gpus: all`) stanza in their
compose service.

## 4. Configuration walkthrough

Both roles run the same image (`Dockerfile` at the repo root); the mounted
DeploymentSpec decides what the process is. Working CPU examples live in
`deploy/compose/` and are exercised by `scripts/compose_smoke.sh` in CI.

Replica node (GPU):

```yaml
server: { host: 0.0.0.0, port: 8000 }
engines:
  llama-70b:
    backend: kairyu            # or vllm; mock for CPU smoke
    options: { model: "meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size: 4 }
```

Gateway:

```yaml
server:
  host: 0.0.0.0
  port: 8000
  api_keys_env: KAIRYU_API_KEYS     # comma-separated client keys
  max_concurrency: 256
pools:
  llama-70b:
    replicas:
      - backend: openai
        options: { base_url: "http://gpu-0:8000/v1", model: "llama-70b", api_key_env: null }
      - backend: openai
        options: { base_url: "http://gpu-1:8000/v1", model: "llama-70b", api_key_env: null }
    unhealthy_after: 3
    queue_depth_threshold: 8
    probe_interval_s: 5.0
orchestrator: { spec: agent_pool.yaml }          # optional: kairyu-auto routing
embeddings:
  embed-test:
    backend: mock                                # deterministic built-in CPU backend
    dimensions: 384
batch: { data_dir: /var/lib/kairyu/batch, max_concurrency: 8 }
```

Operational notes:

- **Session affinity is the cache layer.** Clients that send the OpenAI
  `user` field (or an `X-Session-ID` header) keep a conversation on the
  replica holding its warm radix-KV prefix. Watch
  `kairyu_pool_decisions_total{reason="session_affinity"}` — a low share on
  multi-turn traffic means clients aren't sending session identity.
- **Gateway HA**: gateways are stateless (the batch data dir is the one
  exception) — run two behind the edge LB. Point batch clients at one
  gateway, or share `batch.data_dir` over NFS.
- **Two GPU nodes acting as one model** (TP/PP/P-D across nodes) is an
  engine-layer concern configured by `ClusterSpec` per `docs/gpu-runbook.md`
  §7; the gateway still sees one OpenAI endpoint per coherence domain.
- **Embedding model IDs are explicit.** Each `embeddings:` key is listed by
  `/v1/models`, routes only to its configured backend, and must not collide
  with an engine, pool, or orchestrator name. Unknown IDs return
  `model_not_found` without execution or usage accounting.

## 5. Rolling model update (gate C7)

Weights update = rolling replica restart; there is no hot swap (m7 §3).
For each replica node, one at a time:

1. Stop the replica container. In-flight requests fail once; the gateway
   ejects the replica after `unhealthy_after` consecutive failures and
   traffic redistributes (verified by the smoke drill's kill step).
2. Update the image/weights reference, start the container, and wait until
   the node's own `/readyz` returns 200.
3. The gateway's prober restores the replica automatically (watch
   `kairyu_replica_healthy` return to 1). Proceed to the next node.

Rehearse the drill on the CPU compose topology: `scripts/compose_smoke.sh`
runs exactly this sequence against mock replicas.

## 6. Observability

- `/metrics` (Prometheus): request counts by model/status, latency
  histograms, per-replica outstanding/health, pool decision counts, batch
  job states. Scrape every gateway and replica.
- JSON logs on stdout (one access line per request, prober/batch events);
  ship with the log collector of your choice.
- `/readyz` is the LB health check for gateways; `/health` is the container
  healthcheck on every node.

## 7. DC–cloud interconnect

Only the gateway tier needs to be reachable from the cloud edge; size the
link for request/response payloads (tokens, not tensors — KV never crosses
it). A Direct Connect / ExpressRoute class link gives predictable latency
for TTFT-sensitive SLOs; keep an IPsec VPN as the fallback path. Restrict
the edge→DC path to the gateway port; replica ports stay DC-internal.

## 8. Appendix: Kubernetes (untested reference)

If a revisit trigger from §3 fires, the migration is: one Deployment per
role, the DeploymentSpec as a ConfigMap, `/health`→livenessProbe,
`/readyz`→readinessProbe, a Service in front of gateway pods, and the NVIDIA
GPU operator on replica nodes (pin one replica pod per node with
`nodeSelector` + `resources.limits.nvidia.com/gpu`). These manifests are
deliberately not shipped as tested artifacts — the compose topology is the
supported path until the triggers fire (m7 D2).
