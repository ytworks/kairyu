# Frontier model native runtime and example boundary

Status: **Implemented for single-rank incremental execution and CPU/tiny-fixture
contracts; production GPU and DeepSeek distributed gates are open** (2026-08-09)

This document amends FZ-D1 in `frontier-model-zoo.md`. It records what the
frontier example rebuild may claim before full-checkpoint GPU evidence exists.

## FN-D1 — Production selection is explicit and fail-closed

`execution_mode: native` is mandatory in the new Qwen3.6 and DeepSeek V4
Kairyu example configs. `execution_mode: reference` is retained only for CPU,
small fixtures, and diagnostics. A frontier architecture cannot silently enter
the generic paged-KV runner, use vLLM inside Kairyu, enable a draft decoder, or
reduce `max_model_len` when a capability or memory check fails.

The native single-rank runner advances only new tokens through the official
architecture implementation's `forward_cached` contract. Kairyu, rather than
that model wrapper, owns admission, scheduler lifetime, prefix identity,
sampling, cancellation, and cache rollback.

## FN-D2 — CacheDescriptor is the scheduler-facing ABI

`CacheDescriptor` and `CacheHandle` expose a model-specific composite cache
without pretending all state is KV:

- Qwen: FP32 gated-DeltaNet recurrent/conv state plus BF16 paged KV for the
  full-attention layers.
- DeepSeek: block-256 HCA and CSA state, 4/128 compression metadata, sparse
  top-k/indexer state, FP4-indexer-cache provenance, and mHC state.
- Prefix reuse stores only complete state snapshots in a byte-bounded LRU.
  A generic token-prefix hit alone never skips recurrent/compressed work.
- Transactions clone opaque state before commit and restore it on rollback.
  Nested transactions are rejected.

The current runner keeps opaque addresses model-owned and therefore remains
eager. CUDA Graph pointer stability and model-specific speculative
commit/rollback require their separate GPU gates before they can be enabled.

## FN-D3 — Checkpoint and parser trust boundary

Qwen3.6 and DeepSeek V4 are loaded through pinned Transformers architecture
classes with remote code disabled in the Kairyu process. The DeepSeek public
block-FP8 loader delegates tensor materialization to the official class without
requantization. The mixed FP4-expert/FP8-nonexpert full-checkpoint tensor ABI
and SM120 packed kernels still require full-checkpoint GPU validation.

The L3 API carries `reasoning_effort: low|high|max`, preserves
`reasoning_content` in complete and streamed responses, and parses the pinned
DeepSeek DSML tool-call envelope. OpenAI-compatible replica gateways render the
checkpoint chat template before sending an identity-wrapped request; Kairyu
does not use legacy role concatenation.

## FN-D4 — DeepSeek distributed target and open gate

The committed descriptor and validation surface recognize EP 1/2/4/8 and the
example topology is locked to two EP4 + Attention-DP replicas. The required
native DeepSeek worker must shard experts at load time, retain replicated
attention state per DP rank, and perform fixed-capacity NCCL all-to-all
dispatch/combine without host-derived split vectors.

That worker is **not GPU-validated or production-complete in this change**.
The existing `DistEPLauncher` remains Qwen3-MoE-specific; it must not be
reported as DeepSeek V4 support. Consequently DeepSeek Kairyu readiness,
EP4/EP8 selection, 1M context, CUDA Graph, DSpark, soak, and failure-recovery
remain formal open gates. No EP8 topology lock is generated until all gates
pass and SLO-goodput exceeds EP4 by at least 2%.

## FN-D5 — Orchestration policy

The L2 DSL can load a SHA-256-pinned calibrated router artifact. Artifacts
below a 0.99 quality-ratio confidence lower bound are rejected. `auto-max`
maps to three Tier1 proposals plus Tier2 synthesis. Tier1 direct failures retry
Tier2 once; a stream retries only before any output has been emitted, avoiding
mixed answers. Inputs are never truncated by the router or gateway.

The checked-in router is an all-Tier2 structural baseline. It is safe by
construction but does not claim Tier1 goodput. A measured train/holdout
artifact may replace it only after the benchmark calibration gate passes.

## FN-D6 — Rebuilt examples and evidence

`examples/` contains only the shared controllers and the Qwen 1-GPU,
DeepSeek 8-GPU, and combined 8-GPU environments. Model revisions, external
images, CUDA bases, contexts, GPU counts, VRAM, disk, SM120 capability, and
NUMA-local CPU sets are fail-closed. The first download hashes every model file
and subsequent starts mount the same volume read-only with offline mode.

Each environment exposes the same `run.sh` and `bench.sh` interface. Benchmark
IDs can run alone or through `all`; reports and evidence are finalized after
each attempted benchmark, and `compare` starts the two backends sequentially.
CLI enumeration, shell syntax, Compose expansion, and report mechanics are
CPU/static gates. Full benchmark completion is intentionally a later GPU gate.

## FN-D7 — Enablement gates

- Qwen MTP stays off until greedy equality, sampling non-inferiority, the full
  quality suite, and at least 5% SLO-goodput improvement pass.
- DeepSeek DSpark stays off under the equivalent 7-token gate.
- Kairyu performance is not publishable when paired quality is inferior.
- `PROGRESS.md` must not claim production frontier support until the real
  Qwen 262K and DeepSeek EP4/EP8 1M GPU runs, 30-minute soak, OOM/worker-failure
  recovery, and vLLM comparison all close.
