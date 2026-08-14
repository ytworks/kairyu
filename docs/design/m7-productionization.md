# M7 Design: Productionization — Serve CLI, Gateway Wiring, Batch, Observability

Status: **Implemented — CPU half** (2026-07-02). All D1–D8 landed with tests;
gates C1–C4/C5–C6 proven against mocks (C2/C3 additionally by the CI compose
smoke drill); GPU bring-up is `docs/gpu-runbook.md` §9. Human sign-off pending.
**Amended 2026-07-03** (roadmap, goal G5): at thousands-of-GPU fleet scale D2's own
revisit triggers fire — k8s is adopted as the machine layer (G5 F1); D6 gains
prefix-aware placement then KV tiering as its recorded revisit path (G5 F2/F4);
D8's no-OTel stance flips now that a tracing consumer exists (G5 F1d). Everything
shipped here remains the per-node baseline; see `docs/roadmap.md` §5.
**Amended 2026-07-29** (D3): the deployment-owned `ServerSection` explicitly
translates to runtime `ServerSettings` instead of inheriting its schema.
**Amended 2026-07-29** (D3, issue #230): `kairyu validate` performs a
deterministic offline preflight of the deployment schema and its declared local
artifact graph without starting serving or model execution.
**Amended 2026-08-01** (D6, issue #187): native per-replica RadixKV gains an
optional bounded, NUMA-attested pinned-DRAM tier whose restore policy is loaded
only from retained, runtime-identity-matched crossover evidence.
**Amended 2026-08-07** (D3, issue #343): deployment pools may opt into the
existing bounded approximate `PrefixIndex`; omission remains byte-identical and
does not imply the separate exact KV-event transport lifecycle.
**Amended 2026-08-07** (D5, issue #341): an explicitly configured admission
timeout lets a single local built-in backend use its advertised sequence budget
as the active-request cap and hold only the remaining configured allowance in a
bounded, timed FIFO queue. Omission, multiple models, and unknown/pool backends
retain the configured cap and historical immediate saturation 429.
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
server: { host, port, api_keys_env, max_concurrency, admission_wait_timeout_s, metrics: bool }
engines:            # name -> single backend
  small: { backend: mock, options: {...} }
pools:              # name -> ReplicaPool of backends
  llama-70b:
    replicas:
      - { backend: openai, options: { base_url: "http://gpu-0:8000/v1", model: "...", api_key_env: null, upstream: kairyu } }
      - { backend: openai, options: { base_url: "http://gpu-1:8000/v1", model: "...", api_key_env: null, upstream: kairyu } }
    unhealthy_after: 3
    queue_depth_threshold: 8
    prefix_index: true       # opt-in bounded approximate KV-aware routing
    probe_interval_s: 5.0
orchestrator: { spec: agent_pool.yaml }   # optional, reuses DSL loader
batch: { data_dir: /var/lib/kairyu/batch, max_concurrency: 4 }  # optional
```

`ServerSection` independently owns this durable deployment vocabulary and
explicitly translates it to the runtime `ServerSettings` value. It does not
inherit the runtime model: adding a runtime-only setting therefore cannot
silently add a key or change the generated DeploymentSpec schema. Existing
deployment keys and defaults remain backward-compatible; a new public YAML
setting must be added to its owning spec model and mapped deliberately.
Pool-level `prefix_index: true` constructs the existing process-local bounded
`PrefixIndex` and passes it to `ReplicaPool`; omission preserves session HRW and
load fallback byte-for-byte. Exact KV-event routing remains owned by its event
transport/provider lifecycle and is not implied by this approximate opt-in.
`ServerSection` validates the durable YAML artifact; `ServerSettings` remains
the internal value validated for direct serve-layer callers and owns
environment-backed API/admin key resolution.

`kairyu validate <deployment.yaml>` is the validation-only boundary. It checks
the DeploymentSpec schema plus declared local orchestrator specs, `.jinja`
templates, and `kairyu`/`kairyu-proc` model and tokenizer references. The
command is always offline: it reads no credential environment values, reveals
no secrets, constructs no backend, materializes no model tensors, starts no
server, performs no network access, and probes no hardware. Metadata-only
model shapes and checkpoint headers are sufficient for the local compatibility
checks. Schema and local-filesystem failures are binding; network checks are
skipped and hardware checks are indeterminate.
Exit status is `0` for a valid graph, `1` for validation failure, and argparse's
`2` for CLI misuse. The current DeploymentSpec declares no standalone adapter,
grammar, or benchmark artifact links, so those surfaces are explicitly outside
this command rather than represented by speculative checks.

### D4 — Health/readiness/metrics live in the serve layer; the pool stays passive

`/health` = process liveness. `/readyz` = every engine constructed and every
pool has ≥1 validated, non-ejected replica. `/metrics` = Prometheus text format
(`prometheus-client`, pure-Python — D8). A replica with a declared readiness
URL starts unknown and is excluded from placement and readiness until a
successful probe; backend traffic is never implicit validation. Replicas with
no readiness URL remain locally trusted for direct/programmatic compatibility.

The background **prober** is a FastAPI-lifespan task in
`kairyu/deploy/prober.py`. Its first tick runs immediately at startup, then each
tick snapshots every unknown/ejected replica by stable ID, opaque entry
generation, and resolved `/readyz` URL. Requests run with bounded concurrency;
one failure is isolated, and a 200 response calls `ReplicaPool.probe(id)` only
if that exact generation still exists. This prevents a late response from
validating a replacement that reused the same ID. m5 D4's "no background
tasks" is preserved: the pool remains pure hashing and exposes only read-only
health/validation/generation accessors plus explicit probe state changes.

**Serving hot-path amendment (2026-08-07, issue #349).** Metrics path
templating uses a 1,024-entry LRU keyed by the raw path, retaining the existing
bounded-cardinality label while avoiding repeated regex work on ordinary
routes. SSE JSON line-separator escaping replaces three membership scans with
one ASCII classification before translating non-ASCII content. The chat body
limit wraps `receive` and forwards each
validated chunk immediately; it retains only the running byte count, so the
middleware no longer duplicates the complete request body before Starlette
materializes it.

### D5 — Auth: managed WAF at the edge; static API keys at the gateway; keyless node-to-node

The edge (WAF/LB) owns DDoS, per-client rate limits, TLS. The gateway ships
defense-in-depth: optional static API keys sourced from an env var
(comma-separated, constant-time compare), exempting `/health` and `/readyz`;
plus a global concurrency guard. `max_concurrency` bounds active plus queued
`/v1/*` requests. When `admission_wait_timeout_s` is explicitly set for exactly
one local built-in backend with an immutable sequence budget, that budget caps
active requests and the remaining allowance waits in a version-independent
FIFO for at most the configured timeout; overflow and timeout return 429 +
`Retry-After`. The queue is class-blind because the middleware runs before body
parsing: interactive and batch priority begins at native scheduler admission,
so operators must keep this bounded wait inside their TTFT policy. Omission,
multiple models, and unknown or pooled backends preserve the configured active
cap and therefore have no implicit queue.
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

#### D6 amendment (2026-08-01) — native per-replica pinned-DRAM KV tier

**Decision:** extend each native replica's existing RadixKV cache with one
optional, bounded pinned-DRAM tier. This is a local eviction/restore tier, not
a global pool: each TP rank owns one startup-allocated slab fingerprinted to
its model, TP rank, KV layout, dtype, and page geometry. CUDA ranks temporarily
bind the calling thread to a disjoint share of GPU-local CPUs while creating
the transfer stream, allocating and first-touching the complete slab, and
attesting every mapped page through Linux sysfs/procfs. The caller's affinity
is restored on exit; host checksums use the same scoped rank-local affinity.
CPU-only construction keeps its existing non-NUMA path.

The CUDA backend retains that one attested allocation but interprets its
physical views as `[fragment, slot, bytes]`, where each K/V layer fragment has
the same page-indexed order as `PagedKVPool`. It caches the host and device
plane owners once and submits one non-blocking Torch copy per fragment and
jointly contiguous host/device extent. This removes per-page Python view
construction from the transfer boundary. The logical per-page byte order,
SHA-256 checksum, fingerprint, and ownership protocol do not change; CPU and
injected compatibility backends retain the flat `[slot, page-bytes]` seam.

Radix eviction snapshots only computed, page-aligned prefixes. The DRAM object
key is the full SHA-256 prefix-chain digest rather than the compact 64-bit
routing/display hash. Every host page also carries a full SHA-256 checksum and
the rank-local physical-layout fingerprint. A restore must validate all three,
complete into newly reserved HBM pages, and publish the restored Radix node
only after physical H2D completion. A dedicated CUDA stream and timing events
own asynchronous D2H/H2D work independently of the model stream. Host slots,
source pages, and destination pages remain reserved until the completion
handle finalizes exactly once. Cancellation, callback failure, checksum or
identity mismatch, and especially unknown CUDA completion fail closed:
ambiguous host slots and HBM page IDs are quarantined and never laundered back
into either allocator through a retry or another logical key.

TP model collectives remain NCCL. A separate Gloo control group serializes
`available_prefix`, offload, restore, and discard on every rank, validates one
ACK per rank, and publishes a logical restore only after unanimous all-rank
success. A safely settled miss falls back to recompute; partial or unknown
ownership is discarded where safe or raised without publishing/reusing the
ambiguous pages. The existing Radix LRU stays the logical eviction owner, and
an evicting node is temporarily invisible to matching, pinning, and reentrant
cache callbacks.

The feature is enabled only by supplying both a positive
`dram_kv_tier_capacity_pages` and a retained `dram_kv_tier_profile`. The
profile independently re-derives the first stable measured restore-winning
suffix from nine paired repeats per prefix length, then must match the exact
runtime identity: model config, complete installed Kairyu Python-source
rollup, actual attention execution composition, batching limit, CPU/NUMA/PCIe
cohort, GPU/Torch/CUDA identity, KV layout, TP degree, and exact versioned
transfer-backend/host-layout identity. Startup rejects a stale profile, a
TP-rank identity disagreement, or capacity smaller than
`ceil(min_restore_tokens / page_size)`. P-D separation is deliberately not
supported by this first local tier and rejects the option at configuration
time; it does not change the P-D `KVTransport` ownership contract.

Both benchmark arms start with empty destination pages and end with one
next-token result. Restore validates and transfers the cached KV, then replays
the final prompt-token query because restored KV alone contains no logits.
Cold recompute processes the complete prompt, including its final token in the
natural final production chunk, and samples directly from that hidden state;
it does not execute an additional one-token model forward.

A completed schema-v1 FlashInfer TP4/TP8 collection passed its correctness and
provenance checks but produced no stable restore-winning suffix. Audit then
found that v1 cold recompute split the final prompt token into an additional
model invocation, biasing the comparison toward restore. That collection is
diagnostic only: it cannot seed policy and cannot be relabelled as schema v2.

The exact-source schema-v2 Qwen3-32B run closes F4a with separately executed,
non-overlapping TP4 and TP8 shards from clean commit
`edd535f7018695fc03c479a86fbd690174cca5ef` and one immutable image. The
retained TP4 profile starts its stable restore-winning suffix at 1,024 tokens;
TP8 passes from the 16-token measured lower bound, so the manifest records its
crossover as at or below 16 tokens and the deployable profile conservatively
sets `min_restore_tokens` to 16. Assembly, retained-copy verification, and
independent raw replay all pass. The complete raw evidence, identity-bound
profiles, manifest, image inspect, full container IDs, and created/exited
container records are retained under
`bench/results/g5-f4a-dram-kv-tier-qwen3-32b-rtxpro6000-2026-08-01/`.

#### D6 amendment (2026-07-31) — F4c global KV pool remains deferred

**Decision:** keep per-replica RadixKV plus Kairyu's F2 prefix-aware placement.
Do not deploy a global KV pool now. Complete the native DRAM tier and its
agentic-trace gate (F4a/#187 and F4b/#188) first. If the predeclared revisit
trigger below fires, adopt **Mooncake Store only as a bounded storage/transfer
data-plane provider behind a separate Kairyu-owned global-KV object-store
adapter**. This is not the existing three-method `KVTransport` seam. Kairyu
continues to own the token/model namespace, radix index, request/replica
routing, logical object naming, cache truth, restore-versus-recompute policy,
shard commit, tenant isolation, and fallback. Mooncake owns only physical
object allocation, immutable storage, and transfer. LMCache is not adopted as
Kairyu's cache-policy layer, and `KVTransport` is not expanded into a
home-grown global store.

This is the m7 D6 revisit promised by G5 F4c. It does not change D6's separate
response-cache decision or its byte-identical-request trigger.

##### Measured duplicate-prefix and recomputation mass

`verification/fleet/diagnostic/global_kv_pool_decision.py` independently hashes and replays the
retained F2a and F2c evidence. It does not rerun either measurement and does
not copy numbers out of their manifests:

| Evidence | Session-HRW control | Prefix-aware placement | What the replay proves |
|---|---:|---:|---|
| F2a, 500 logical replicas, 64 seeded shared-prefix families | 27/1,024 hits; final 1,061 family copies; 997 redundant family copies = 3,988 logical 256-character prefix chunk-copies | 1,024/1,024 hits; final 64 family copies; zero redundant copies | Seed and placement order replay reconstructs logical residency exactly. HRW accumulated 513,809 duplicate family-copy/request-steps; routing recovered all 4,096 reusable chunks and avoided 3,988 recomputed chunks (97.3633% of the reusable opportunity) |
| F2c, four real Qwen3-32B TP2 replicas on eight GPUs | 329,280 / 659,266 cached prompt tokens | 648,976 / 659,266 cached prompt tokens | On 256 identical prompt pairs, routing avoided 319,696 matched recomputed tokens (48.4927% of all prompt tokens and 96.8817% of control misses), with no pairwise cache regression |
| F2c performance | TTFT p95 527.958 ms | TTFT p95 134.358 ms | Candidate/control TTFT ratio 0.254486 while goodput ratio remained 0.999998 |

The units are deliberately separate:

- **Logical duplicate-copy mass** is the extra family/chunk copies reconstructed
  from F2a's seed, replica, sequence, and cache-hit rows. The trace has no
  eviction, so every successful cold placement adds one resident family copy.
- **Matched avoidable recomputation mass** is candidate cached work minus
  session-HRW cached work for the identical paired prompt.
- **Incremental global-pool mass** would be a post-routing local miss for which
  a compatible copy actually exists on another replica *and* transfer beats the
  measured F4a recompute crossover. F2a's treatment has none in its seeded
  reusable set. F2c's remaining 10,290 uncached tokens (1.5608% of prompt
  tokens) include novel suffixes, so 1.5608% is only a gross upper bound, not a
  measured remote-reuse rate.

No artifact records physical KV bytes, byte-seconds, or an exact-event
real-engine residency index. This decision therefore makes no such claim.

The generated artifact is
`bench/results/f4c-global-kv-pool-decision-2026-07-31.json`. Reproduce it
offline with:

```bash
uv run --frozen python verification/fleet/diagnostic/global_kv_pool_decision.py \
  --verify-artifact \
  bench/results/f4c-global-kv-pool-decision-2026-07-31.json \
  --assert-gate
```

The input F2a raw SHA-256 is
`82bb2a2ff420dbd4e244685ce2a83f38379028604be3c1077e85daf5b31cd0f3`;
the input F2c raw and trace SHA-256 values are
`4cfcdeba2b7473aa6c2b28409dbf21de23d775d9b08e971beed6bdab875abe64`
and
`51d188671432bf791c02d66d91e6a7d785eb2bd01f64e29a41a62e74f9957dad`.
The generated F4c decision artifact SHA-256 is
`1f75eca37df253ae27b651b4702f988ce364165378d80ce451615a3b7a5b06d3`.

##### Buy/build comparison

The external snapshot is dated 2026-07-31 and pins
[Mooncake v0.3.12.post1](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.12.post1)
plus
[main `f5f6a94`](https://github.com/kvcache-ai/Mooncake/commit/f5f6a94edcf6ee226435909d0483c321075ed951),
and
[LMCache v0.5.2](https://github.com/LMCache/LMCache/releases/tag/v0.5.2)
plus
[`dev` `145ec2c`](https://github.com/LMCache/LMCache/commit/145ec2c2a3f032a30d80593e8f67bdf614700f5e).
Both projects are Apache-2.0 and pre-1.0. Mooncake and LMCache are not strictly
exclusive: LMCache can use Mooncake as a remote backend.

| Option | Operational surface | Correctness/failure model | Performance evidence and fit | Ownership verdict |
|---|---|---|---|---|
| **Mooncake Store** | Master plus Store clients and Transfer Engine; DRAM/VRAM/NVMe, TCP for development and RDMA/multi-NIC for production; optional HA log/snapshot operation | Immutable object `Put/Get/Remove`; after a successful Put, Get is strongly consistent and returns the complete most-recent object rather than a partial/old value. Capacity allocation and requested replication remain best effort independently of HA; HA preserves Master metadata/service continuity, not a requested replica count | Its vLLM agentic result reports 3.8x throughput and 46x p50 TTFT improvement, and its RDMA result reports 142.25 GB/s (71.1% of theoretical). These are vendor results on other hardware/workloads, not Kairyu evidence | **Chosen only after a trigger**, as physical object allocation/storage/transfer below Kairyu's logical cache semantics. Best match for preserving L1/L2 ownership |
| **LMCache** | Connector plus token index, async multi-tier policy, MP server/controller, and selectable CPU/disk/NIXL/Redis/Mooncake backends | Cache failure is intended to degrade to recompute; durability/consistency follows the selected backend. Its engine connector and 256-token default chunk semantics must be integrated and version-matched | Its documented Qwen3-8B long-document example reports mean TTFT 757 → 185 ms. It has no Kairyu connector, and this is not a matched Kairyu comparison | **Not selected.** Its index, tiering policy, connector, and controller overlap Kairyu RadixKV/F2 responsibilities |
| **Extend Kairyu KVTransport into a global store** | Kairyu would add lookup, leases, storage membership, replication, eviction, recovery, capacity control, and operations to today's `register/send/recv` transfer seam | Native PageFrame/layout semantics fit, but Kairyu would own every distributed-store failure mode and upgrade contract | Reuses current serde/NIXL work, but no global lookup/store performance or reliability evidence exists | **Not selected.** Keep KVTransport for native P/D and tier movement; do not turn it into a global metadata/storage service |

Primary external references:
[Mooncake architecture](https://kvcache-ai.github.io/Mooncake/design/architecture.html),
[Mooncake Store semantics](https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html),
[Mooncake deployment and HA](https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html),
[Mooncake vLLM shared-pool result](https://kvcache-ai.github.io/Mooncake/performance/vllm/vllm-v1-mooncake-store.html),
[Mooncake RDMA result](https://kvcache-ai.github.io/Mooncake/performance/vllm/vllm-v1-pd-performance.html),
[LMCache architecture](https://docs.lmcache.ai/developer_guide/architecture.html),
[LMCache MP architecture](https://docs.lmcache.ai/mp/architecture.html), and
[LMCache benchmark recipe](https://docs.lmcache.ai/getting_started/benchmarking.html).

##### Revisit trigger and prototype contract

Revisit only after F4a and F4b publish their retained evidence. Before reading
results, declare one deployment, checkpoint/revision, TP degree, hardware
profile, and UTC start. Then retain three consecutive, non-overlapping windows
of exactly 10,000 eligible requests from that cohort. An eligible request is a
successfully admitted, cache-enabled text-generation request with a non-empty
prompt; eligibility is independent of whether telemetry is present. Every
consecutive eligible request is included and must have exact compatible-block
telemetry. A missing identity, residency, token-count, or timing row invalidates
the whole window rather than allowing that request or window to be excluded or
replaced after results are known.

For each window calculate:

1. `remote_token_fraction = remote_reusable_tokens / eligible_prompt_tokens`.
   The numerator is the model-token count in exact full blocks that miss on the
   selected replica at routing time while a committed, namespace-compatible
   copy exists on another replica. The denominator is every model prompt token
   in the eligible requests, including locally cached and novel tokens.
2. `remote_recompute_time_fraction = remote_block_recompute_gpu_time /
   total_prefill_gpu_active_time`. The numerator applies the predeclared,
   same-cohort F4a token-count/recompute curve to those same exact remote
   blocks; the denominator is measured prefill GPU-active time over all
   eligible requests in the window.

The trigger fires only if the **same branch** holds across all three windows:
`min(remote_token_fraction) >= 0.05` or
`min(remote_recompute_time_fraction) >= 0.10`. Future telemetry must carry full
block identity, namespace compatibility, local and remote committed residency
at routing time, prompt-token counts, and prefill GPU time. F2c's gross
uncached count has no remote-residency identity and therefore cannot satisfy
the trigger.

The first triggered prototype is Mooncake Store, not a production rollout. It
must prove all of the following against the unchanged per-replica/F2-routing
control:

- a namespace containing the full checkpoint, tokenizer, RoPE/config, adapter,
  KV dtype/layout/page size, engine ABI, TP degree/rank, and tenant identity;
- a full-digest object key (the current 64-bit placement fingerprint is not a
  global object identity), per-shard checksums, and an all-shards commit
  manifest before RadixKV publication;
- timeout, missing/partial object, digest/layout mismatch, controller loss, or
  source loss degrades to a cache miss and recomputation without a failed
  request or partial KV publication;
- tenant authorization and namespace isolation;
- lookup plus transfer beats measured recompute above the F4a crossover;
- exact greedy output parity, no TPOT p99 regression under a predeclared paired
  crossover method, and a positive goodput/TTFT result after all store costs;
- a rollback that removes the adapter and returns to the current architecture
  without changing request or model semantics.

### D7 — Batch: minimal OpenAI-compatible `/v1/files` + `/v1/batches`, filesystem-backed

An in-gateway asyncio worker drains queued batch jobs through the same served
engines/pools under its own concurrency cap (strictly below the server's
global cap, so interactive latency is protected — gate C4). Storage is a
data-dir on disk (JSONL input/output/error files + JSON job state); no
Redis/Celery/queue at this node count. Single-gateway scope: with two
gateways, pin batch traffic to one or share the data dir (documented).
Restart recovery marks `in_progress` jobs `failed` — honest and simple.

**Amendment (2026-07-14):** every input row is a typed request envelope with
a non-blank, per-job-unique `custom_id`, `method: POST`, and a URL equal to the
job endpoint. Interactive and batch chat share one transport-neutral validation
and buffered-dispatch service, including tool/format/image/model/sampling/backend
preflight checks and post-generation tool-choice enforcement. Controlled request
failures do not dispatch; arbitrary backend failures expose only their exception
class while retaining the full traceback in server logs.

**I/O and interactive-pressure amendment (2026-08-07, issue #342):** filesystem
and PostgreSQL output/error transactions now admit encoded rows to the existing
bounded background JSONL writer; only the worker's off-loop finalization waits
for the accepted rows before atomic publication or rollback. All synchronous
store calls in the HTTP routes and filesystem-worker path run outside the shared
event loop, and file content is returned as fixed-size off-loop chunks rather
than one whole-file allocation. A filesystem-store lock makes job start, cancel,
and terminal publication atomic now that routes and workers can use different
threads. When `server.ttft_slo_s` is enabled, a batch consumer waits before
starting its next line while interactive work is active and the controller
predicts that one more interactive request would exceed the SLO, but continues
to observe cancellation and claim loss. This is admission-only: already-dispatched
batch work is not cancelled or preempted, and disabling predictive admission
preserves the fixed consumer cap.

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
3. **m7 D3 server-schema ownership** — `ServerSection` is a frozen,
   extra-forbidden DeploymentSpec model with an explicit conversion to
   `ServerSettings`. Runtime settings remain internal unless the external YAML
   vocabulary deliberately adopts and maps them.

## 5. Verification

- Phases 1/2/4: pytest via ASGI transport (existing pattern), ruff, 80%
  coverage gate. Prober/worker tested with injected transports and intervals
  (no sleeps).
- Phase 3: CI `compose-smoke` job — cold `docker compose up`, `/readyz`
  poll, non-stream + SSE completion, affinity assertion via metrics, replica
  kill/recover drill (gates C1–C3).
- G3 gate table checked off in the goals doc as phases land.
