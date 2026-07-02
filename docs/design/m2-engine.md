# M2 Design: Kairyu Core Engine (Overlap Scheduler + Radix-Paged KV)

Status: **Reviewed — APPROVE-WITH-AMENDMENTS** (agent design-review panel, 2026-07-02; see §6).
GPU hardware (H100/A100) still required for the GPU phase; human sign-off pending.
Milestone: M2
Date: 2026-07-02
Depends on: M1 (`EngineBackend` protocol is the integration seam; no L2/L3 changes needed)

## 1. Goal

A custom Python engine (`kairyu.engine.core`, registered as backend name `kairyu`) that runs
Llama-3.1-8B in FP8 (W8A8) and, on the acceptance workloads, beats or matches vLLM V1:

- ShareGPT + synthetic multi-turn, ≥128 concurrent requests: TTFT p50/p99, TPOT, goodput
  measured against vLLM and SGLang on identical hardware. Target: ≥20% TTFT reduction vs
  vLLM, or better p99 at equal throughput.
- Shared-prefix-50% workload: KV cache hit rate >80% (this is where the Radix-Paged design
  and the M1 `cache_hint` plumbing pay off — orchestration steps hit warm prefixes).

All numbers must come from `bench/` reproduction scripts; no estimated or extrapolated
results are ever reported (goal acceptance criteria).

## 2. Architecture

### 2.1 Process model

```
AsyncKairyuEngine (asyncio, API-facing)
  └─ EngineCore (separate process, busy loop)      ← ZeroMQ pair socket, msgpack
       ├─ Scheduler (CPU)                          ← overlap-pipelined with GPU step
       ├─ KVManager (Radix-Paged hybrid)
       └─ ModelRunner (GPU: FlashInfer kernels, CUDA graphs in M3)
```

Mirrors vLLM V1's split (API process vs EngineCore process) because Python GIL contention
between HTTP serving and the scheduling loop is a measured bottleneck in vLLM V0. Tokenizer
runs API-side (detokenization is incremental, per V1).

### 2.2 Zero-overhead overlap scheduling

The known vLLM V1 bubble: schedule(step N+1) waits for output of step N. SGLang's overlap
scheduler hides CPU work under the GPU step; we adopt the same structure:

- The scheduler prepares batch N+1 (page tables, sampling metadata, block allocations)
  while the GPU executes batch N.
- Sampled token IDs of step N are needed to build inputs of N+1 only for the sequences that
  advanced; we launch step N+1 with a *placeholder last-token slot* and patch it from the
  GPU-side sampled tensor (device-to-device copy, no host sync) — the SGLang "future token"
  technique. Host synchronization happens only for finish-condition checks, which run one
  step late (a request may generate one surplus token; it is trimmed at output).
- Invariant to test: engine step loop never blocks on `.item()`/`.cpu()` in the hot path
  (asserted by a torch profiler check in the perf test suite).

### 2.3 KV management: Radix tree over paged blocks

- Fixed-size pages (16 tokens default, FlashInfer paged-KV layout).
- A radix tree keyed on token-ID chunks maps prefixes → page lists. Nodes hold refcounts;
  eviction is leaf-LRU over refcount==0 nodes (SGLang's RadixAttention policy) — this beats
  vLLM's hash-per-block prefix caching on multi-turn reuse because partial-prefix matches
  don't require exact block-aligned hash hits.
- `GenerationRequest.cache_hint.session_id` (plumbed in M1) pins a session's radix path
  against eviction between orchestration steps (bounded TTL so abandoned sessions drain).
- No copy-on-write needed (amended per review): sharing is page-aligned and partial pages
  (prompt tails, in-progress decode pages) are always private, so shared pages are
  immutable by construction. Children are keyed by their first page's token tuple.
- Computed gating (amended per review): nodes carry a ``computed`` flag; matching never
  descends into uncomputed nodes, so chunked prefill in progress is never shared as if it
  were valid KV. Identical in-flight prompts keep private pages instead of colliding.
- Generated tokens are folded into the tree at request finish
  (``commit_and_release``) — turn N+1 prompts with turn N's completion appended, so this
  is what makes multi-turn prefixes actually hit. Partially-filled pages return to the
  pool. A fully-cached prompt still recomputes its last token to produce logits.
- Metrics: `kv_hit_tokens / prefill_tokens` exported per request and aggregated — this is
  the acceptance metric for the >80% hit-rate criterion.

### 2.4 Prefill/decode policy

- Default: chunked prefill (token-budget scheduler, decode-priority) — one policy knob
  `max_num_batched_tokens`.
- Optional single-node P-D separation (M3): two CUDA streams / two model replicas on one
  node with page-granular KV handoff. M2 only reserves the config surface
  (`pd_separation: bool`) and keeps the KV layout transfer-friendly (contiguous pages).

### 2.5 Model execution

- Kernels: FlashInfer (paged attention prefill+decode, fused RoPE, sampling); no custom
  CUDA in M2. Fallback to FlashAttention-2 where FlashInfer lacks a shape.
- FP8 W8A8 via compressed-tensors checkpoints (llm-compressor output) with per-tensor
  scales; BF16 fallback path for correctness A/B tests.
- Model zoo M2: Llama-3.1-8B only (goal-specified). Architecture-agnostic loader deferred.
- Correctness gate: greedy-decode token-level parity with HF transformers BF16 on 64 fixed
  prompts (FP8 compared by logprob tolerance, not exact match).

### 2.6 What M2 explicitly does not include

CUDA graphs, speculative decoding, xgrammar, AWQ/GPTQ, multi-node — all M3+ (goal
milestones). The scheduler and KV manager are written so none of these require interface
changes (CUDA-graph capture wraps ModelRunner; spec decode adds a draft stage inside the
step loop).

## 3. Development strategy without local GPU

This machine (macOS/arm64) cannot run CUDA. Split:

1. **Pure-Python components developed and tested locally now:** scheduler policy, radix
   tree + page allocator (refcount/LRU/COW logic), request lifecycle, msgpack protocol —
   all as deterministic unit tests (same rigor as M1, coverage gate stays at 80%).
2. **GPU-dependent components** (ModelRunner, FlashInfer calls, FP8 load) behind a
   `torch.cuda.is_available()` guard, tested on a rented H100/A100 box; bench scripts and
   CI GPU job specs land with them.

This ordering front-loads the algorithmically risky parts (KV manager, overlap loop) where
unit tests are possible, and keeps the GPU session focused on integration + measurement.

## 4. Bench plan (reproduction scripts in `bench/`)

| Script | Workload | Metrics | Baselines |
|---|---|---|---|
| `bench/serving_sharegpt.py` | ShareGPT, 128/256 concurrent | TTFT p50/p99, TPOT, goodput | vLLM V1, SGLang |
| `bench/multiturn_prefix.py` | synthetic multi-turn, 50% shared prefix | KV hit rate, TTFT | vLLM (prefix caching on) |
| `bench/orchestration_e2e.py` | M1 Conductor DAG on real engine | per-step TTFT vs cold | vLLM backend via M1 |

Identical: model, dtype, max_num_batched_tokens, request trace, hardware, driver. Configs
committed alongside results (`bench/results/<date>-<gpu>.json`).

## 5. Risks

- Overlap patching of the last-token slot is subtle (off-by-one on stop conditions);
  mitigated by the correctness gate running with overlap on AND off. The implemented
  CPU-side design carries an explicit ``position`` on each decode chunk plus per-request
  ``in_flight`` accounting (scheduler.py/overlap.py); §2.2's "placeholder patching" prose
  and this mechanism are the same technique — the GPU runner owns a future-token device
  buffer indexed by (request, position). Restated invariant: the *scheduling* thread never
  blocks on device sync; the worker thread syncs once per step to produce sampled ids.
- FP8 accuracy drift on Llama-3.1-8B: measured via logprob tolerance gate before any perf
  claim. FP8 **KV-cache** dtype is a separate decision from W8A8 weights (changes
  FlashInfer kernel selection) — decide at GPU-phase start; default bf16 KV.
- FlashInfer API churn: pin version in `pyproject.toml` GPU extra (`kairyu[gpu]`).

Mandatory pre-GPU work items from the design review:

1. ~~EOS/stop-token semantics under overlap~~ **DONE 2026-07-02**: `EngineRequest.eos_token_id`,
   surplus in-flight tokens trimmed (not errors) when a request finishes early under
   schedule-ahead; tested (`test_scheduler_robustness.py`).
2. ~~Preemption under KV pressure~~ **DONE 2026-07-02**: decode-priority recompute-preemption
   of the youngest output-free running request via `release_preempted` (does NOT mark
   pages computed), plus `decode_watermark_pages` admission reserve; tested. Output-KV
   recompute for victims that already generated tokens is a GPU-phase extension.
3. Typed `StepInput` for the ModelRunner (page tables, positions, new token ids, sampling
   params) instead of handing over mutable scheduler state; async result handle for M3
   CUDA graphs; per-step streaming output path out of the engine core; tokenizer /
   incremental detokenizer component; ZMQ/msgpack frame spec for the process split.
   `Scheduler.abort()` (client disconnect) **DONE 2026-07-02**; the rest lands with the
   GPU runner since its shape depends on FlashInfer metadata layout.
4. ~~Session-pin TTL~~ **DONE 2026-07-02**: `pin(..., ttl_allocations=N)` expires lazily on
   allocation ticks; tested.
5. Eviction victim scan is O(nodes) per eviction — switch to a heap before measuring
   scheduler overhead (profiling first; only matters at large tree sizes).

Bench controls (amended per review, applies to §4): pin exact versions of vLLM/SGLang/
FlashInfer/driver; disclose that vLLM V1 runs CUDA graphs by default while kairyu M2 has
none (M3 closes this); define goodput's SLO threshold explicitly; measure TTFT at first
streamed token including tokenize+queueing; ≥3 runs with p50/p99 across runs, fixed seeds,
warmup excluded; add an open-loop arrival-rate sweep, not just fixed concurrency; include
SGLang as a baseline on the shared-prefix benchmark (its radix cache is the direct
competitor there).

## 6. Review record

Agent design-review panel, 2026-07-02 (senior inference-engine reviewer persona over the
doc + implemented CPU-side code). Verdict: **APPROVE-WITH-AMENDMENTS**. Disposition:

- Fixed in code same day: cached-prefix compute skip (`KVAllocation.num_cached_tokens` +
  scheduler init from cached tokens, last token always recomputed); computed-gating on
  radix nodes (no garbage-KV sharing, in-flight collision keeps pages private);
  generated-token fold-in at finish (`commit_and_release`). All covered by tests.
- Fixed in doc same day: §2.3 rewritten to page-aligned no-COW design; §2.2/§5 reconciled
  with the implemented position/in-flight mechanism; bench controls added to §5.
- Deferred to GPU-phase start (tracked in §5 "mandatory pre-GPU work items"): EOS-under-
  overlap surplus handling, preemption + decode watermark, typed StepInput/abort/streaming
  /tokenizer/ZMQ spec, pin TTL, eviction heap.
- Human sign-off: pending.
