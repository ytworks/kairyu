# M5 Design: Intra-Node Multi-GPU (TP Runner, DP Replicas, P-D Disaggregation)

Status: **Reviewed — APPROVE-WITH-AMENDMENTS** (agent design-review panel, 2026-07-02;
see §7). GPU hardware (8×H100) required for the GPU phase; human sign-off pending.
Milestone: M5
Date: 2026-07-02
Depends on: Goal G2 (`docs/goals/g2-multi-gpu.md`, gates A1–A10); M2 GPU phase Gates 1–3
(single-GPU 8B runner correct and benchmarked, `docs/gpu-runbook.md` §1–2); typed
immutable `StepInput` (m2 §5 item 3 — promoted from deferred to **blocking prerequisite**
of Stage 5.1, see D2)

## 1. Goal

Serve Llama-3.3-70B FP8 on one 8×H100 NVLink node at G2's M5 gates: TP with correctness
anchors and scaling-efficiency floors (A1–A7), DP replicas behind the L2 orchestration
layer with session affinity (A8–A9), and intra-node P-D disaggregation (A10). All
acceptance numbers from `bench/`; TP=2 is the scaling base (70B FP8 does not fit one
80 GB GPU).

## 2. Key design decisions and rationale

### D1 — TP lives inside one ModelRunner; KV accounting stays rank-invariant

`TPModelRunner` implements the existing `ModelRunner` protocol
(`kairyu/engine/core/engine_core.py`): a driver owns the (unchanged) Scheduler and
RadixKV; worker ranks execute the same step. The load-bearing observation: **KV
*accounting* is device-agnostic.** Pages, the radix tree, refcounts, and page IDs are
identical on every rank — only the KV *tensor* is head-sharded (70B GQA has 8 KV heads;
at TP=8 each rank stores 1 head in the same page slots). So the CPU-side RadixKV remains
a single structure on the driver, page tables broadcast with the step input, and G2's A7
(radix reuse preserved under TP) holds by construction rather than by synchronization.
Verified against the code in review: RadixKV/Scheduler hold no device state. Caveat
(amended): the pool's `num_pages` must be sized as the **min over ranks** of post-
weight-shard free memory at the given TP degree — accounting is rank-invariant only
after pool size is fixed identically on the driver. Scheduler, radix KV, and step-loop
public contracts unchanged (G2 §7 seam).

### D2 — Dedicated non-rank driver process; Communicator protocol; NCCL only at the edges

(Amended: the driver is **not** rank 0.) The driver is a dedicated scheduler process
that is not a TP rank: workers are the 8 NCCL ranks; the driver broadcasts a typed,
**immutable** `StepInput` snapshot (page tables, positions, new token ids, sampling
params) over shm/zmq and receives sampled ids. Rationale: (a) a rank-0 driver puts
scheduling, broadcast, and its own rank's kernel launches on one GIL — a direct A4
hazard; (b) `execute()` today receives live mutable `_RequestState` objects, and under
`OverlapEngineCore` the schedule thread mutates them while the runner thread would
serialize them — torn page tables across processes. Hence typed `StepInput` (m2 §5
item 3) is a blocking prerequisite of Stage 5.1, with a stated per-step driver budget of
≤1 ms.

A minimal `Communicator` protocol (broadcast / all_reduce / all_gather / barrier /
send / recv) is the only place collective communication appears. Implementations:
`NcclCommunicator` (GPU phase) and `FakeCommunicator` (in-process, deterministic, for
CPU tests of the driver/worker step protocol). No custom collectives (G2 non-goal), but
NCCL **NVLS / symmetric-memory collectives are a required mechanism, not an
optimization** — A3/A4's efficiency floors have no margin without them (§5). Worker
processes are spawned with `torch.multiprocessing` (spawn), one per GPU; no Ray
intra-node.

### D3 — `tensor_parallel_size` stops being a no-op; API shape unchanged

`LLM(...)` and `AsyncEngineArgs` forward `tensor_parallel_size` into the backend factory
(`_default_backend` → `create_backend`, `kairyu/entrypoints/llm.py`,
`async_engine.py`, `kairyu/engine/registry.py`). The `kairyu` backend consumes it
(constructs `TPModelRunner` with N ranks); the `vllm` backend already passes it through;
`mock` records it (tests assert plumbing). Validation at the boundary: `>= 1`, and the
kairyu backend rejects values that don't divide the model's KV-head count (8 for both
contract models; TP=2/4/8 all valid, Q heads 64/32 also divide).

### D4 — DP replicas are an orchestration-layer concern: `ReplicaPool` + affinity

DP never touches the engine. A `ReplicaPool` (new, `kairyu/orchestration/replica.py`)
wraps N `EngineBackend` instances behind the same `EngineBackend` protocol, so L3 and the
OpenAI server are unchanged. **Seam amendment (flagged per G2 §7):** the G2 seam says DP
routing "goes through the existing L2 Router"; the `Router` protocol
(`kairyu/orchestration/router.py`) returns `tier1|tier2|multi_agent` and cannot express
replica choice, so `ReplicaPool` is a sibling component in the same L2 layer rather than
a `Router` implementation. It inherits the router's <10 ms p99 budget and its JSONL
log — via a new `JsonlRouterLog.record_replica()` method (a `kind: "replica"` entry; the
M4 dataset builder filters by kind, so the training corpus is unaffected). G2 §7's seam
wording is amended accordingly (see G2 "Amendments").

Placement policy, in order:

1. **Session affinity**: requests with `cache_hint.session_id` map to a replica by
   rendezvous hashing over healthy replicas — this is what preserves multi-turn radix
   hits (G2 A8's ≥90% clause); naive round-robin is rejected for cause.
2. **Least outstanding requests** for session-less traffic, and as the fallback when the
   affine replica's queue depth exceeds a threshold (load-skew valve, §5).

Health: a replica that fails K consecutive requests is removed from the hash ring until
a probe succeeds; nothing more (G2 non-goal: no migration/autoscaling).

### D5 — P-D intra-node: per-role processes on disjoint GPU sets, copy-on-handoff

(Amended: the draft's zero-copy page-donation default is **withdrawn**. Review showed it
(a) requires both roles to share physical GPUs, recreating exactly the interference A10
eliminates, and (b) is unsound over one `PagePool` with two RadixKV trees — eviction
frees refcount-0 leaves directly, so either tree can pool-free pages the other still
maps.) The design is now:

- **Topology**: prefill role and decode role are separate OS processes on disjoint GPU
  sets (e.g., prefill TP=4 on GPUs 0–3, decode TP=4 on GPUs 4–7; role→GPU-set assignment
  is part of the config surface). Separate processes also remove the shared-GIL jitter
  a two-cores-one-process layout would inject into decode TPOT p99 (the A10 metric).
- **Each role owns a sovereign Scheduler + RadixKV + pool** (admission halves already
  exist: prefill-only budget on one side, `pd_separation` + `decode_token_budget` on the
  other, `kairyu/engine/core/scheduler.py`).
- **Handoff is a copy**, page-granular, via the `KVHandoff` protocol
  (`transfer(pages, seq_meta) -> handle`): intra-node GPU impl = pairwise
  device-to-device copies on a side stream (rank i → rank i under matched TP degree;
  4k tokens × 160 KB = 640 MB total, ~160 MB per pair at NVLink rates ≈ 0.4 ms raw —
  the A10 ≤5 ms budget holds on copy alone, no zero-copy needed). CPU impl is a dict
  copy; it is what unit tests exercise, and M6 swaps in a network transport behind the
  same interface.
- **Handoff protocol (amended — the ordering is load-bearing):** the prefill-core
  request runs with `max_new_tokens=1`, so after the prompt-completing chunk samples
  token 0 no further decode chunks are ever scheduled (no in-flight hazard under
  overlap). The `PDCoordinator` (new, `kairyu/engine/core/pd.py`) intercepts at
  **execute-completion** of that chunk — after the KV is physically written, **before**
  `update()` commits — copies all pages *including the private tail page* plus token 0,
  and only when the transfer handle resolves does it apply `update()`, letting the
  request finish normally (`commit_and_release` then folds the prompt into the prefill
  tree for cross-request prefix reuse — the prefill node keeps serving warm prefixes).
  Copy-before-commit is what prevents `commit_and_release` from pool-freeing the tail
  page under the copy. Transfer failure → the token is not committed; abort + requeue.
- **Decode-side resume (amended — a new Scheduler entry point, flagged per G2 §7 as an
  additive public-contract extension):** `resume_with_kv(request, pages, tail_page,
  first_token)` constructs state directly as `computed_prompt = prompt_len`,
  `in_flight = 0`, `outputs = [token0]`, full pages folded into the decode tree as
  computed nodes, tail page attached private. Constructing `outputs` non-empty is
  load-bearing twice: it bypasses `_admit_waiting`'s recompute-last-token rule (no
  re-sampling of token 0), and it shields the request from `_preempt_for_decode`
  (victims must have empty outputs) — without it, decode-core preemption would set
  `computed_prompt = 0` and recompute the whole prompt on the decode GPUs, defeating
  the topology. Radix fold-in keeps A7 holding in P-D mode (bench row added, §6).

### D6 — Scaling measurement is part of the deliverable, not an afterthought

`bench/serving_bench.py` grows `--tensor-parallel`, `--dp-replicas`, `--pd` topology
arguments and a sweep mode that emits the TP=2 base and TP=4/8 points into one results
file (G2 §8 same-file rule). `bench/multiturn_prefix.py` gains a `--replicas` mode to
measure A8's affinity hit-rate clause.

## 3. Prerequisites promoted by review; what M5 does not include

**Promoted to explicit A4 prerequisites** (amended — the arithmetic leaves <1–2 ms of
per-step CPU budget at TP=8): decode CUDA-graph capture per TP topology (without graphs,
per-step launch overhead alone exceeds the whole budget), the non-rank driver process
(D2), and NVLS-class collectives (D2). These land at GPU-phase start, before the A3–A5
sweeps.

Not included: PP (M6), inter-node anything (M6), MoE/EP, sequence parallelism, custom
collectives, KV offload tiers, elastic replica scaling (all G2 non-goals).

## 4. Development strategy without local GPU

Same split as M2 §3. CPU now, GPU session later:

1. **CPU-testable now**: `Communicator` protocol + `FakeCommunicator`; typed immutable
   `StepInput` + the driver/worker step protocol (broadcast, gather sampled ids) with
   fake ranks; `tensor_parallel_size` plumbing (+ divisibility validation);
   `ReplicaPool` with affinity/least-loaded/health policies over `MockBackend`s +
   `record_replica` logging; `PDCoordinator` + `KVHandoff` with CPU pages including the
   copy-before-commit ordering, tail-page transfer, and radix fold-in;
   `Scheduler.resume_with_kv`; bench topology args (runnable against mock).
   Deterministic unit tests, 80%+ coverage, same rigor as M1/M2.
2. **GPU session**: `NcclCommunicator` (+NVLS), sharded FP8 70B load (per-rank
   safetensors shards), FlashInfer paged attention under head-sharded KV, CUDA-graph
   capture per TP topology, stream-based handoff copy, then gates A1–A10 in `bench/`
   order.

## 5. Risks

- **A3 margin is ~zero at 4k prompts**: pricing in per-GPU GEMM efficiency loss at TP=8
  (smaller N dims → MFU ~40% vs ~50% at TP=2) plus the constant TTFT terms in G2 §3's
  definition, the ≤⅓ floor is winnable only with all-reduce/GEMM overlap and NVLS —
  hence D2 makes them required mechanisms. If the GPU-phase measurement still misses,
  the fallback is a G2 amendment restating A3 at 8k-token prompts (compute-dominated),
  flagged through the goal's amendment process — not silent gate relaxation.
- **All-reduce in the decode hot path**: TP decode adds 2 all-reduces/layer × 80 layers;
  A4's 50% floor was set for exactly this. The A4 budget arithmetic (TP=8 ≈ 2.6 ms
  weight-read + 2.4–4.0 ms comm) is why CUDA graphs and the ≤1 ms driver budget are
  prerequisites, not optimizations (§3).
- **Affinity vs load skew**: session affinity can hot-spot a replica. D4's queue-depth
  fallback is the valve; the A8 sweep reports both hit rate and per-replica load spread
  so the tradeoff is visible in results, not hidden.
- **Handoff correctness**: the copy-before-commit ordering (D5) is enforced by the
  coordinator owning the `update()` call for the prompt-completing token; a test pins
  that no `commit_and_release`/`abort` path can run between KV write and copy
  completion.

## 6. Bench plan

| Gate | Script | Notes |
|---|---|---|
| A1, A2 | `bench/parity_tp.py` (new) | 64 fixed prompts, overlap ON/OFF; 8B TP=1 vs TP=2, 70B TP=2 vs 4/8 |
| A3–A5 | `bench/serving_bench.py --sweep-tp 2,4,8` | TP=2 base in same results file; plus a report-only TPOT point at concurrency 64 (stresses all-reduce growth; no threshold, A9 spirit) |
| A6 | `bench/serving_bench.py` vs pinned vLLM TP=4/8 | ShareGPT@128 + shared-prefix trace |
| A7 | `bench/multiturn_prefix.py --tensor-parallel N` and `--pd` | hit rate at TP=4/8 AND under the P-D split (D5 claims it; this row proves it) |
| A8, A9 | `bench/serving_bench.py --dp-replicas 2` + `multiturn_prefix.py --replicas 2` | goodput, router p99, affinity hit retention; DP-vs-TP sweep |
| A10 | `bench/pd_mixed.py` (new) | long-prefill + decode-SLO mix; handoff p99 |

## 7. Review record

Agent design-review panel, 2026-07-02 — three parallel reviewers (engine correctness
vs code; performance realism vs gates; goal compliance/scope). Verdict:
**APPROVE-WITH-AMENDMENTS**. Disposition:

- **Blockers fixed in doc**: zero-copy donation withdrawn (unsound dual-tree pool
  accounting + physically incompatible with disjoint-GPU roles) → copy-on-handoff
  default with per-role processes (D5); handoff ordering rewritten to
  copy-at-execute-completion-before-commit with `max_new_tokens=1` on the prefill core
  (D5); driver demoted from rank 0 to a dedicated non-rank process with typed immutable
  StepInput as a Stage 5.1 blocking prerequisite (D2).
- **Amendments applied**: CUDA graphs + NVLS + driver budget promoted to A4/A3
  prerequisites (§3, §5); `resume_with_kv` semantics fully specified (tail page,
  no-resample, preemption shield) and flagged as an additive Scheduler contract
  extension (D5); ReplicaPool flagged as an L2 seam amendment with
  `record_replica()` named honestly (D4); pool sizing = min over ranks (D1); A7-under-
  P-D and conc-64 report-only bench rows added (§6).
- Verified by review: D1 rank-invariance and KV-head divisibility check out against
  `radix_kv.py`/`scheduler.py`.
- Human sign-off: pending.
