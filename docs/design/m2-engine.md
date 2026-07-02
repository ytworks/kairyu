# M2 Design: Kairyu Core Engine (Overlap Scheduler + Radix-Paged KV)

Status: Draft — requires review approval AND GPU hardware (H100 or A100) before implementation
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
- Copy-on-write on divergence: shared pages stay shared until a sequence writes into a
  partially-filled page, then the tail page is copied (refcount-aware).
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
  mitigated by the correctness gate running with overlap on AND off.
- FP8 accuracy drift on Llama-3.1-8B: measured via logprob tolerance gate before any perf
  claim.
- FlashInfer API churn: pin version in `pyproject.toml` GPU extra (`kairyu[gpu]`).
