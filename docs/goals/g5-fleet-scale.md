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
- CPU-first discipline: F1 deployment/lifecycle gates run against CPU-mock
  fleets in kind. F2a isolates selector quality and cost in a 500-logical-entry
  CPU bench while reusing F1a's deployment precedent; later F2 gates use the
  fixture named in their table row. GPU validation slots in per
  `docs/gpu-runbook.md`.

## 3. Acceptance gates

### Stage F1 — Elastic control plane (CPU-mock)

| Gate | Target | Where proven |
|---|---|---|
| F1a | ReplicaPool dynamic membership: kind cluster, 1 gateway + 200 mock replicas, 10%/min churn for 10 min → zero 5xx, placement p99 <10 ms | kind CI job |
| F1b | Drain-first partitioned StatefulSet restart of exactly 100 mock replicas via one `kubectl rollout restart`: retry=0, zero failed requests, no human/operator repair, exact old/new UID and revision joins, and independently replayed raw rollout/request/readiness evidence (C7 lineage) | dedicated formal kind CI job; PR smoke is non-acceptance |
| F1c | 3 gateways behind an LB with consistent-hash session partitioning pass the C1 affinity assertion; batch jobs complete with the shared `BatchStore` | kind CI job |
| F1d | One request produces one end-to-end OTel trace (gateway route → pool place → replica call); Conductor runs show per-stage spans | **Closed** by deterministic fixture + separate-container Compose smoke (m10 A34; 2026-07-31) |

F1a and F1b are closed by retained exact-head Actions runs `30374404150` and
`30387260062`. F1c reuses their kind-runner capacity, image/source provenance,
NodePort traffic, discovery/placement/membership, and zero-failure rollout
evidence instead of repeating either measurement. Its one additional binding
drill is limited to the distinct three-gateway affinity, shared PostgreSQL
BatchStore, and fenced owner-Pod failover contract in m10 A29.

F1c is closed by exact-head source run `30399229234` at commit `be40b97`.
All 26 replay checks passed over six sticky sessions, all three gateways, 12
stable replica UIDs, one common PostgreSQL identity, a fence-1 owner-Pod kill
followed by a different gateway's post-expiry fence-2 reclaim, 200/200
successful batch lines, and byte-identical output through every gateway. The
complete raw and replay artifact is retained in
`bench/results/f1c-three-gateway/`. The evidence-only closure commit does not
repeat the binding drill.

F1d is closed under m10 A34. A deterministic in-memory fixture proves the
gateway, route, pool-placement, replica-call, remote-server, and per-stage
Conductor parentage as well as success, error, cancellation, stream lifetime,
and privacy. The mandatory Compose CI smoke proves the same W3C trace crosses
real gateway and replica containers, joins records by request/trace/span/parent
IDs rather than log order or timing, observes the final response body, and
rejects prompt/output canaries from every span.

### Stage F2 — KV-aware routing (CPU-first, then 4–8 GPU testbed)

| Gate | Target | Where proven |
|---|---|---|
| F2a | Prefix-trie scorer: on an identical cross-session shared-prefix trace against 500 mock replicas with independently simulated cache state, backend-truth cached prompt-work rate is ≥2× the non-zero session-hashing baseline; on 21 alternating paired uniform session-only (blank-root) rounds, the exact distribution-free one-sided ≥95% median lower bound and the full-sample geometric mean of the <10 ms SLO-goodput ratio are each ≥0.99, equivalently with ≥15/21 individual ratios ≥0.99 for the median bound; shared and uniform placement p99 are each <10 ms | replayable CPU bench (m10 A30) |
| F2b | RadixKV KV-event index: every exact route uses truth <250 ms old and therefore strictly <500 ms under formal churn; killing one binding physical feed makes the full 200-entry route use the approximate oracle; restoring it converges by complete replay without process restart in <500 ms | replayable CPU chaos fixture (m10 A31) |
| F2c | Real-engine validation: candidate/control nearest-rank TTFT p95 ratio ≤0.70 pooled, at the seventh ordered ratio of eight crossover rounds, and by geometric mean on a multi-turn+RAG trace; SLO-goodput ratio ≥0.99 pooled, at the second ordered ratio, and by geometric mean; pooled engine-token cache rate strictly improves without a per-round regression | **Closed** by retained 8-GPU artifact (m10 A32; 2026-07-29) |
| F2d | Production `kind=replica` decisions join exactly once to `placement_outcome` TTFT rows and feed `learning/dataset.py`; after disjoint-family training selects and freezes one normalized `λ=β/α` policy (`α=1`) by minimum mean TTFT, complete held-out deterministic virtual-time replay from the same frozen initial state must show strictly lower mean TTFT than the declared `λ=0.25` baseline, with complete, zero-failure, balanced, independently replayable evidence. p95 and action differences are diagnostic only | **Closed** by retained CPU replay artifact (m10 A33; 2026-07-29) |

F2a reuses F1a run `30374404150` only for its kind deployment, ingress clock,
discovery, placement-log, provenance, and hosted-runner precedents. It does not
reuse F1a's measurements and does not repeat the kind drill. The one distinct
binding run uses 500 logical eligible `ReplicaPool` entries through the public
generation path, retains raw cache/placement/performance JSONL, and independently
replays both policies under the same per-request session IDs and initial cache
state. One fully retained and replayed non-binding uniform calibration pair
warms CPython's policy-specific allocator path before the 21 binding paired
rounds; it does not enter any metric. Alternating execution order, the exact
median lower bound, full-sample geometric-mean guard, and sign-count guard make
the declared 1% equivalence tolerance robust to host scheduling jitter. The
location bound is the seventh ordered ratio for 21 pairs, with exact 96.0823%
binomial coverage; no round is removed or clipped, and the former Student-t
log-mean bound remains diagnostic only. Immediately before each calibration and
binding arm's clock, that same pool and policy execute a declared 512-request
run-in over disjoint prompts. Its
deterministic trace digest, completed count, and positive interval remain in
the arm summary and are independently bound before measurement, but its
samples enter no metric; this prevents fresh-pool code and CPU-frequency ramp
from favoring the second arm. SLO-goodput divides qualifying completions by the
summed public-generation dispatch time of every offered request, so a slow
non-qualifying request cannot disappear from the cost; round wall time is
audit-only. The implementation
carries one lazy cumulative-hash chain from placement through successful
publication. The process-local approximate index uses versioned XXH3-64 keys;
Conductor carries a root only when its shared prefix contains a complete
256-character chunk, and that hint is the exact local XXH3 root. A blank
`CacheHint` declares session-only affinity and bypasses native prefix work;
binding uniform traffic exercises this production opt-out/HRW path rather than
general cold prefix-tracking overhead. Sessionless requests
retain local discovery, while malformed non-empty hints and custom chunk sizes
retain local hashing. Cold success admits only the root needed for later
discovery through a dedicated fast path, while a successful warm route promotes
the same chain to full depth. It does not claim a 500-Pod deployment.

F2a is closed by exact-source run `30411111758` at `c067cb8`. Shared cached
prompt-work improved 37.9259x; the uniform paired median, exact median lower
bound, and full-sample geometric mean were 1.002142, 0.999512, and 1.008610,
with 21/21 ratios at or above 0.99. Worst-trace placement p99 was 0.145979 ms.
The independently replayed raw artifact is retained under
`bench/results/f2a-prefix-routing-500-2026-07-28/`.

F2b reuses F1a's exact seed-175, 200-replica, ten-by-twenty identity schedule
without repeating its Kind/NodePort measurement, and reuses F2a only for the
production `ReplicaPool`/prefix-routing precedent. Its distinct CPU fixture
compresses wall pacing while preserving every churn ordinal. It drives 199
sequenced in-process feeds plus one physical ZMQ feed through the same
200-entry pool; exact scores are one atomic fleet observation, so loss of the
representative feed makes the entire request use the approximate trie.
Sequence gaps, cache-lifetime epochs, bounded replay, authoritative snapshots,
inactive tombstones, and same-object epoch rotation prevent delayed frames
from reviving incomplete cache truth.

The 250 ms route lease leaves half of the strict 500 ms acceptance envelope as
host-scheduling headroom. The verifier uses actual monotonic event and route
times and fails any pause/resume action lateness, offered-route lateness, or
selected-route blind spot at 500 ms; a later catch-up iteration cannot conceal
OS or event-loop jitter, and no sample is excluded or relabeled. It
reconstructs the churn schedule, membership generations, event
`(epoch, sequence)` joins, high-water replay, exact/approximate routing oracles,
the state captured at replay completion, the final 200-replica state digest,
and unchanged pool/router/index/publisher/subscriber identities from raw
JSONL. One clean exact-source formal run is binding; a descendant PR replays
retained bytes instead of measuring again when every gate-input source hash is
unchanged, the recorded completed-success run is verified through the GitHub
Actions API, and both retained files are byte-identical to that run's original
artifact.

Exact-source Actions run `30417507859` at `f383806` closes F2b. Across all 500
offered routes, maximum exact truth age was 232.314498 ms, the first stale
approximate route followed the physical feed pause by 251.339950 ms, and
same-process complete-replay recovery followed resume by 50.740933 ms.
Maximum route lateness and selected-route gap were 3.608193 and 21.536138 ms,
so no catch-up execution hid an OS scheduling stall. The independently
replayed 2,196-row artifact is retained byte-identically at
`bench/results/f2b-kv-event-retained/`; F1a and F2a were not rerun, and this
evidence-only retention does not repeat F2b.

F2c uses the production `ReplicaPool` plus streaming
`OpenAICompatBackend` path against four independent Qwen3-32B TP2 endpoints on
all eight GPUs. Two cache-disjoint, two-replica cohorts run candidate
`PrefixIndex` and session-HRW control simultaneously and exchange policies
across eight rounds. A recorded, single-use namespace keeps smoke and retry
roots disjoint. Each round has 16 unique 2,048-word RAG families: a cold seed
targets one logical replica, while the measured session hashes to the
opposite replica. Paired policies receive identical session/prefix hints,
prompts, and fixed eight-token generation. Each family predeclares a canonical
assistant continuation whose digest and resulting turn-2 prompt digest are
trace-bound. Turn 1 must succeed on both arms before turn 2, but neither
post-treatment output is reused; both arms receive the same frozen transcript.
Production decisions must be `prefix_match` versus `session_affinity`, and
paired prompt tokens and completion work must agree while each arm's exact
engine usage remains retained. Nearest-rank p95, the seventh ordered TTFT
ratio, pooled ratio, and geometric mean bind the 30% gain.
Goodput binds a 0.99 pooled, second-order, and geometric-mean floor; exact
`cached_tokens / prompt_tokens` must improve pooled without a round
regression. Raw scheduling skew and lateness remain replayed diagnostics
without an arbitrary fail cutoff. Per-request output digests remain raw
evidence, while their cross-arm match count, total, and rate are diagnostic
only. The first formal attempt stopped on the former exact-output assertion;
a fixed-endpoint reproduction found individually repeatable but cross-endpoint
different continuations despite fully warm caches. This is consistent with a
BF16/TP near-tie under different cache-population execution shapes, including
possible chunk/prefill-history differences; it does not establish semantic
cache corruption or a physical GPU-pair effect. Per G2's free-running
precedent, it is not a routing correctness gate. Router JSONL, topology,
configuration, model/source hashes, and all raw request evidence are
independently replayed.
This intentionally does not broaden F2c into DeploymentSpec exact-KV-event
wiring, whose hash-provider and subscriber lifecycle are a separate D7 product
responsibility.

F2c is closed by the exact-source formal artifact retained at
`bench/results/f2c-kv-aware-ttft-qwen3-32b-2026-07-29/`. Its offline verifier
accepted every check over 512 binding requests with zero failures. Pooled
control-to-candidate TTFT p95 was 527.957623 ms → 134.357747 ms, with
candidate/control ratios of 0.2544858548 pooled, 0.2550841404 at the seventh
ordered round, and 0.2530080045 by geometric mean. Engine-token cache rate was
0.4994645560 → 0.9843917326 with all rounds noninferior. SLO-goodput ratios
were 0.9999979014 pooled, 0.9998437390 at the second ordered round, and
0.9999978783 by geometric mean. Output agreement remained diagnostic at
239/256 (0.93359375); paired-receipt skew and schedule lateness maxima were
5.182959 ms and 7.470463 ms.

The retained evidence binds source
`80b039b5d429c656871a480c2740740951b29b97`, image
`kairyu-f2c@sha256:d2c01580964f461a3d3d2a02ced5303e69c681696d4a38179162084e1624121f`,
raw SHA-256
`4cfcdeba2b7473aa6c2b28409dbf21de23d775d9b08e971beed6bdab875abe64`,
and trace SHA-256
`51d188671432bf791c02d66d91e6a7d785eb2bd01f64e29a41a62e74f9957dad`.

F2d is closed under m10 A33 by the exact-source deterministic full-policy
replay at `86dde278d0f2a093bde64f5d1d9cba9aca9e1221`. Seven normalized
policies replayed all 768 requests in each arm over 48 training families from
fresh identical state; `learning/dataset.py` selected `λ=1.0` before held-out
execution. On 16 family-disjoint held-out episodes, both the frozen winner and
declared `λ=0.25` baseline replayed the same 256 requests from fresh identical
state. Mean TTFT was 4.43359375 versus 8.5 virtual ticks, all 5,888 production
placement rows joined one-to-one to successful outcomes, and the offline
verifier independently reconstructed every decision, queue state, TTFT, split,
selection, metric, hash, and verdict. p95 and 176/256 action differences remain
diagnostic only. The retained evidence is
`bench/results/f2d-prefix-weight-replay-2026-07-29/`, with manifest SHA-256
`3205721922fd8c013ae6336aaa4ffcb0a1938a40059e70acb500b5acba86ac3c`,
raw SHA-256
`1ccc5ab012e5ee6677f96709ec60cc15ea5db32cefb72360941238ca505c75eb`,
and production-router SHA-256
`3296fdd000aede574ea5c3a152ff1ef0f54e204545bfb1f9aa61f7b47c83546f`.

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
| F5e | Usage metering reconciles with request logs to <0.1% on a replayed trace; per-tenant cached-token counts exported (feeds G6 pricing signals) | CPU test + `bench/fleet_usage_replay.py` |

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
