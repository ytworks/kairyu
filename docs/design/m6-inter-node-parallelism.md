# M6 Design: Inter-Node Multi-GPU (2-Node DP, KV Transfer Plane, P-D, PP)

Status: **Reviewed — APPROVE-WITH-AMENDMENTS** (agent design-review panel, 2026-07-02;
see §7). Two-node GPU hardware required for the GPU phase; human sign-off pending.
**Amended 2026-07-03** (roadmap): D1's no-Ray decision stands, but the static-only
topology is relaxed — goal G5 adds a replica registry and `kairyu.launch`, and the
ClusterSpec 2-node coherence-domain cap rises to 8 for frontier-MoE EP/PP domains
(G5 F3). D3's transport bake-off gains NIXL as a third contender, and B2's fabric
budget is restated against measured NIC line rate (G2 §7 2026-07-03). See
`docs/roadmap.md` §5.
Milestone: M6
Date: 2026-07-02
Depends on: Goal G2 (`docs/goals/g2-multi-gpu.md`, gates B1–B5); all M5 gates green;
M5 design (`docs/design/m5-intra-node-parallelism.md` — ReplicaPool, Communicator,
KVHandoff seams extend across the network here); async `ModelRunner` result handle
(m2 §5 item 3 reserved it for CUDA graphs; PP promotes it to a requirement, see D5)

## 1. Goal

Extend M5 to two 8×H100 nodes over ≥400 Gb/s IB/RoCE at G2's M6 gates, in the goal's
stage order: 2-node DP first (B1), the KV transfer plane as a standalone gated
primitive (B2), inter-node P-D (B3), PP=2 last (B4–B5).

## 2. Key design decisions and rationale

### D1 — Static topology config; no Ray

A `ClusterSpec` (frozen dataclass, YAML-loadable via the existing DSL loader seam,
`kairyu/dsl/`) declares nodes, per-node GPU count, roles (replica / prefill / decode /
pp-stage), and endpoints. Rendezvous is torchrun-style (env-var master addr + rank),
launched by a small `kairyu.launch` helper. Ray is rejected despite m1 §3's "Ray arrives
with multi-node" note: a fixed 2-node topology needs no dynamic placement, and G2
excludes elasticity — a scheduler-framework dependency for two static nodes fails YAGNI.
This supersedes the m1 note (recorded as a PROGRESS.md amendment entry) and is flagged
for review as a goal-adjacent deviation. ClusterSpec validation enforces that a node is
either a PP stage or a P-D role, never both (§5).

### D2 — 2-node DP reuses ReplicaPool; the remote-worker backend needs real work first

(Amended: the draft claimed "no engine change at all" — review against
`kairyu/engine/openai_backend.py` showed that is false.) Each node runs its best M5
config as an independent kairyu OpenAI-compatible server; the front node's `ReplicaPool`
(M5 D4) gains remote members via the `openai` external-worker backend — a replica is
just an `EngineBackend`, local or remote, and affinity hashing, health ejection, and
`record_replica` logging carry over verbatim. **No engine-core change** is required, but
the backend itself needs four fixes before B1 is measurable:

1. **Real SSE streaming** — today `stream()` yields once after `generate()` completes,
   so front-node TTFT for remote replicas equals full completion time, destroying any
   TTFT-based SLO (and with it B1's goodput-at-SLO number).
2. **A persistent pooled `httpx.AsyncClient`** — today a new client (TCP/TLS handshake)
   is created per request, directly against the <10 ms p99 router budget.
3. **Optional auth** — today a missing API-key env var raises; keyless node-to-node
   replicas must be constructible.
4. **Token-count passthrough** — today `token_ids=()` always, leaving TPOT/goodput
   accounting empty through remote replicas.

Stage 6.1's "lowest-risk first" rationale stands, but as "no new distributed machinery",
not "no code change". Front-node ceiling risk and mitigation in §5.

### D3 — KV transfer plane: `KVTransport` protocol, gated standalone on the real layout

One new protocol (`kairyu/engine/core/kv_transport.py`):

```
register(pool) -> None                      # pin/register page memory once, at startup
send(dst, page_ids, seq_meta) -> handle     # async, page-batched
recv(src) -> (page_ids, seq_meta)           # into pre-registered pages
```

- Unit of transfer is the contiguous KV page (m2 §2.3 chose the layout for this),
  batched ≥64 pages per send (B2's measurement condition).
- (Amended — the physical unit is smaller than the logical one.) A logical page spans
  80 layers, and under TP the KV is head-sharded, so per rank a "page" is dozens of
  fragments of ~4–32 KB. Per-fragment `ncclSend` cannot sustain 20 GB/s; therefore
  **fragment aggregation is part of the protocol, not the transport's problem**: either
  RDMA scatter-gather work requests over the registered pool, or packing into a
  registered staging ring (an extra local copy, accepted and disclosed). `send()`
  batches at the aggregation granularity internally.
- Implementations: `LocalTransport` (in-process copy — CPU tests; also the M5 intra-node
  handoff degenerate case) and one GPU-phase network transport, chosen by a bake-off
  (NCCL p2p with staging vs UCX/RDMA SGL) run by `bench/kv_transfer_bench.py`. B2 gates
  whichever wins. **The bench must run against the real sharded per-layer page layout,
  not an idealized contiguous buffer** — otherwise B2 green-lights a transport that
  fails inside B3.
- Sequence metadata (token ids for radix fold-in, per-page validity) rides with the
  pages so the receiver can insert them as computed radix nodes (same fold-in path as
  M5 D5). Validity is sourced from **runner-side chunk-completion events**, never from
  scheduler state (see D4).

### D4 — Inter-node P-D: streamed handoff, overlapped with prefill at the right granularity

Same `PDCoordinator` shape as M5 D5 with the handoff swapped to `KVTransport`. The B3
target (≤20% TTFT inflation) is achievable only if transfer overlaps prefill, so overlap
is structural — with three amendments from review:

- **Send hooks the execute path, not the scheduler.** Under the overlap loop,
  `computed_prompt` and node-granular `computed` flags run a step ahead of the physical
  KV write (`mark_computed` fires at scheduling time). Chunk-complete sends are
  triggered by the runner's execute-completion of each chunk — the only point where the
  KV is known to be written.
- **Page alignment**: chunk boundaries are token-budget-sized, not page-aligned. A chunk
  ending mid-page leaves that page unsendable until a later chunk fills it; the send
  granularity is *completed pages*. The prompt's final tail page is partial by design
  (private, outside the tree) and ships as a private page at final handoff, exactly as
  in M5 D5.
- **The final/only chunk gets layer-group streaming.** For a 2k-token prompt with a
  2048-token chunk budget, "send per chunk" degenerates to send-after-prefill: 320 MB /
  20 GB/s = 16 ms fully on the critical path — straddling B3 with no p99 margin. KV for
  a token's layer ℓ is final once layer ℓ's forward completes, so the last chunk streams
  its pages in layer groups (send layers 0..k while k+1..79 compute). Additionally, P-D
  prefill mode defaults to a smaller chunk budget (≤1024) so multi-chunk overlap exists
  even for short prompts. For ≥2-chunk prompts the plain mechanism already hides
  transfer (chunk compute ≈ 70 ms at TP=8 vs 16 ms transfer).
- The decode node pre-allocates the page range at admission (prompt length known), and
  (amended) **reports its cached-prefix page count back**, so the sender skips pages the
  decode node already holds warm — affinity routing makes this case common, and writing
  into live shared pages, while data-identical, wastes B3 bandwidth.
- **Preemption exemption (amended)**: `_preempt_for_decode` targets running requests
  with no outputs — i.e., every P-D prefill mid-stream would be a preferred victim,
  invalidating pages the decode node already received. Streaming P-D requests are
  exempted from recompute-preemption on the prefill core (they are the node's whole
  purpose; admission budgets, not preemption, regulate them).
- Failure: a transfer error before decode-start requeues the request on the prefill
  node's scheduler; no partial-state recovery beyond that (G2 non-goal).

### D5 — PP=2: inter-step pipelining behind an async runner contract

(Amended: the draft's intra-step micro-batching is **withdrawn** — review showed it
cannot satisfy B4's two clauses simultaneously. Decode is weight-read-bound, so a stage
pass costs ~weight-read time regardless of micro-batch size: splitting a decode batch
into m micro-batches multiplies per-stage passes without shrinking them, giving ~50%
TPOT inflation at m=2 vs the ≤10% gate; while m=1 idles each stage 50%, capping
throughput at ~1.0× vs the ≥1.6× gate. And under the synchronous `execute()` contract
with the overlap loop's serial device executor, the pipeline fully drains every step —
utilization is bounded by m/(m+s−1) ≈ 1.33× at m=s=2. No knob escapes this.)

The redesign pipelines **across scheduler steps**, not within one:

- The `ModelRunner` contract gains the async result handle already reserved by m2 §5
  item 3 for CUDA graphs: `submit(step_input) -> handle` + completion callback. This is
  a flagged G2 §7 seam amendment (the step-loop contract changes shape); EngineCore/
  OverlapEngineCore adopt it for all runners (the sync path is `submit`+immediate
  resolve).
- `PipelinedModelRunner` holds two in-flight scheduler steps with stage affinity: step
  N occupies stage 1 while step N+1 occupies stage 0 — each stage always processes a
  **full decode batch** (never split), so TPOT ≈ single-node equal-TP + one hidden-state
  hop (~0.5 MB per step over IB ≈ tens of µs, ≪ 20–50 ms TPOT: the ≤10% clause), and
  steady-state throughput ≈ 2× per-stage rate (the ≥1.6× clause with margin for stage
  imbalance). The existing `overlap.py` schedule-ahead deque (depth 2, late commit) is
  exactly the seam that feeds two steps into the pipe; depth-2 overlap is therefore a
  PP requirement, not an option. Micro-batching survives only where it belongs: chunked
  **prefill** within a step.
- The scheduler's `in_flight`/`position` accounting already tolerates two uncommitted
  steps (built for overlap depth 2); first bring-up nonetheless runs a serial-commit
  correctness pass (m3 §2.1 precedent) before enabling the pipeline.
- Inter-stage traffic uses the `Communicator` send/recv (M5 D2) over the fabric.
- (Amended) **One driver-side RadixKV, stage-invariant page IDs**: every stage stores
  its 40 layers' KV for *every* request's tokens — what differs per stage is tensor
  content, not page ownership. Same insight as M5 D1 (accounting is rank-invariant),
  now layer-slice-invariant: one accounting authority on the driver, per-stage pools
  hold their layer slice in the same page slots. The draft's "per-stage sovereign
  RadixKV with per-stage page IDs" is withdrawn — it would need mirrored split/LRU
  decisions with no mechanism.
- The runner exports a **bubble-fraction metric per step** (B4 makes reporting it
  mandatory).

### D6 — Fabric truth-in-reporting

`bench/kv_transfer_bench.py` doubles as the fabric microbench: it measures raw
link bandwidth first, then paged transfer efficiency against it — B2's "measured, not
nominal" rule (G2 §8) falls out of one script. Results embed NCCL/UCX versions and
`ibstat`-equivalent link info.

## 3. What M6 explicitly does not include

>2 nodes; heterogeneous GPUs; elastic membership (ClusterSpec is static); PP>2;
cross-node TP (TP stays inside NVLink domains — IB all-reduce per layer would violate
A4-class latency floors by an order of magnitude); MoE/EP; KV offload tiers; fault
tolerance beyond B1's replica ejection and D4's requeue (all G2 non-goals).

**B5 comparison set (amended, pinned here):** because cross-node TP is excluded by
design, Kairyu has no config equivalent to vLLM multi-node TP; B5's vLLM comparisons are
**PP=2 and 2-node DP only**. A results reviewer cannot demand a Kairyu-vs-vLLM-TP16
number this design deliberately does not build.

## 4. Development strategy without local GPU

1. **CPU-testable now**: `ClusterSpec` + validation + YAML loading; `KVTransport`
   protocol + `LocalTransport` + a TCP-loopback transport (real serialization, real
   async, no RDMA) so `bench/kv_transfer_bench.py` runs end-to-end on CPU;
   streamed P-D handoff logic (execute-hooked sends, completed-page granularity,
   cached-page skip, preemption exemption, requeue-on-failure) against fake transports;
   the async `ModelRunner` submit/handle contract + `PipelinedModelRunner` two-step
   stage-affinity pipelining with bubble accounting over fake stage workers
   (deterministic step counts); `openai_backend` fixes (SSE streaming, pooled client,
   optional auth, token passthrough) testable against an in-process HTTP server;
   ReplicaPool remote members over the same.
2. **GPU/2-node session**: NCCL/UCX transport bake-off on the real sharded layout (B2),
   sharded-stage model load, layer-group streaming tuning, then B1→B5 in goal order.

## 5. Risks

- **Transport bake-off is the schedule risk**: RDMA registration of the paged pool
  interacts with the CUDA allocator (pages must be pinned and contiguous — they are, by
  design, but allocator churn can fragment registration). Mitigation: register the whole
  pool once at startup (D3's `register`), never per-transfer. The staging-ring fallback
  trades one local copy for registration simplicity; the bake-off decides with B2 data.
- **Front-node proxy ceiling (B1)**: all request + streamed-token traffic for two nodes
  transits one asyncio process; at saturation, SSE serialization alone can cap goodput
  below 1.85× for reasons unrelated to the engines. Mitigations, in escalation order:
  multiple front workers behind SO_REUSEPORT; direct-to-replica streaming with the
  front node only placing sessions. The B1 run reports front-node CPU so the ceiling is
  visible in results.
- **Shared pages on the send path**: a chunk's pages can be radix-shared on the prefill
  node — transfer is a read (copy semantics), refcounts stay node-local; no distributed
  refcounting. Cross-node prefix reuse is NOT attempted (a session's cache lives where
  affinity lands it — measured, not assumed: §6 B1 row).
- **PP + P-D composition**: never required by the goal; ClusterSpec validation keeps
  them orthogonal (D1) to avoid a topology matrix explosion.

## 6. Bench plan

| Gate | Script | Notes |
|---|---|---|
| B1 | `bench/serving_bench.py --dp-replicas node-a,node-b` + `bench/multiturn_prefix.py --replicas node-a,node-b` | goodput vs single node; router p99 incl. hop; affinity hit-rate retention across nodes reported (not gated — B1's gate is goodput; the hit rate substantiates §5's affinity reliance) |
| B2 | `bench/kv_transfer_bench.py` (new; CPU-runnable with loopback) | raw fabric first, then ≥64-page batches **on the real sharded layout**; ≥20 GB/s, ≤8 µs/token |
| B3 | `bench/pd_mixed.py --prefill-node A --decode-node B` | TTFT inflation vs M5 colocated; TPOT p99 retention; layer-group streaming on/off ablation |
| B4 | `bench/serving_bench.py --pp 2` | TPOT inflation, throughput efficiency, bubble fraction |
| B5 | same scripts vs vLLM multi-node (Ray) for PP=2 and 2-node DP only (§3) | parity axes per G2 |

## 7. Review record

Agent design-review panel, 2026-07-02 — same three-reviewer panel as m5 (engine
correctness vs code; performance realism vs gates; goal compliance/scope). Verdict:
**APPROVE-WITH-AMENDMENTS**. Disposition:

- **Blockers fixed in doc**: intra-step micro-batching withdrawn — PP redesigned around
  inter-step pipelining with the async runner handle promoted from m2 §5 item 3 reserve
  to a requirement, full decode batches per stage, depth-2 overlap mandatory (D5);
  "no engine change at all" corrected — the four `openai_backend` fixes B1 actually
  needs are enumerated (D2).
- **Amendments applied**: per-stage sovereign RadixKV withdrawn → one driver-side
  accounting authority with layer-slice pools (D5); sends hooked to execute-completion
  events with completed-page granularity and tail-page handling (D4); layer-group
  streaming + ≤1024 P-D prefill chunk budget so B3's own 2k example actually overlaps
  (D4); fragment aggregation (RDMA SGL / staging ring) made part of the KVTransport
  protocol and B2 re-scoped to the real sharded layout (D3); streaming P-D prefills
  exempted from recompute-preemption (D4); receiver cached-page skip (D4); B5
  comparison set pinned to PP=2 + DP (§3); B1 cross-node affinity hit rate added as a
  reported metric (§6); front-node ceiling mitigation stated (§5); m1 Ray-note
  supersession recorded in PROGRESS.md (D1).
- Human sign-off: pending.
