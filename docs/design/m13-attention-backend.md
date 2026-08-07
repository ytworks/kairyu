# M13 Design: AttentionBackend Seam — Torch, MLA, FlashInfer, FlashAttention 3/4

Status: **Implemented** (2026-07-03). Reviewed — APPROVE-WITH-AMENDMENTS (1-reviewer panel with web
verification of the FlashInfer 0.6.x API and the DeepSeek-V2 MLA paper,
2026-07-03; amendments below are binding). The issue #277 FlashAttention-3/4
extension is implemented (2026-07-30). FA4 has retained SM120 correctness and
performance evidence. FA3 is a strict, explicit SM90 path whose import/API/
shape contract is fake-tested when representative hardware is unavailable;
that coverage makes no FA3 performance or automatic-selection claim.
Milestone: M13 (roadmap Track E1 GPU-path-local; kernel-adapter pattern setter
for M14/M18)
Date: 2026-07-03
Depends on: M12 (`paged_attention` extraction point, `PagedKVPool`,
`DenseDecoder`). Consumed by: M15 (MLA), deploy day (`pytest -m gpu`).

## 1. Goal

Extract M12's inline attention into a swappable `AttentionBackend` seam and
write the GPU adapters NOW (local-complete mandate): a device-agnostic torch
backend (today's CPU path, CUDA-ready as-is), the MLA reference math M15
needs (two algebraically equivalent forms, cross-checked), and the FlashInfer
adapter — deferred-import, its metadata/indptr construction CPU-tested against
a fake module, its kernel launch `@gpu`-marked. This milestone sets the
GPU-adapter pattern (naming, fakes, coverage) every later kernel follows.

## 2. Key design decisions

### D1 — Protocol: per-request and native ragged-prefill entries

`kairyu/engine/core/attention/__init__.py`:

```python
class AttentionBackend(Protocol):
    def attend(self, query, kv_pool, layer, page_table, seq_len, chunk_start) -> Tensor:
        """query [T, heads, head_dim] -> context [T, heads*head_dim]."""
```

Exactly M12's `paged_attention` signature (designed for this extraction).
Issue #224 adds the completed GPU-day extension:

- `supports_batched_prefill` is an explicit capability declaration. A backend
  whose `attend_batched` is a Python row loop does not opt in.
- `attend_prefill(flat_query, ..., qo_indptr)` consumes one flat ragged query
  and request-local paged-KV metadata. FlashInfer performs one plan for the
  step and one run per layer, with the shared backend instance reusing that
  plan across layers.
- `PrefillBatch` preserves ScheduledChunk order, query bounds, page tables,
  sequence lengths, cached `write_from`, and token-to-request row IDs.
  `PagedKVPool.write_ragged` owns the vectorized slot mapping; attention
  kernels never own K/V placement.

Writable physical pages are exclusive across batch rows. Fully cached pages
may be shared and are filtered out before the write. Page aliases within one
row or a writable page referenced by another row fail before model execution.

Issue #322 amends the host preparation without weakening that invariant:
ownership is checked once per physical page, and all typed prefill metadata is
laid out in one fresh per-batch buffer. CUDA uses pinned host storage and one
non-blocking H2D copy. Application code has no reusable staging pool, and the
pinned allocator defers recycling until the copy completes, so no overwrite
race or application-owned staging event is introduced.

### D2 — `TorchAttentionBackend` (`torch_backend.py`)

M12's `paged_attention` moved verbatim (rectangular mask, `enable_gqa`) —
device-agnostic: the same code runs CUDA tensors on deploy.
`models/attention.py`'s `Attention` module gains a constructor-injected
`backend` (default `TorchAttentionBackend()`); KV writes stay in the module —
the backend computes attention only. `DenseDecoder(config, attention_backend=)`
threads it; `PagedModelRunner` passes through.

### D3 — MLA reference math (`mla_torch.py`), consumed by M15

DeepSeek MLA: per-token compressed latent `c_kv` (`kv_lora_rank`) + decoupled
rope key `k_pe` (`qk_rope_head_dim`); the pool stores `[c_kv ‖ k_pe]` (the
real deploy layout — M15's pool variant). Two forms implemented over given
projection weights, both CPU-tested equal:

- **decompress-then-attend**: `K_nope = c_kv @ W_UK`, `V = c_kv @ W_UV`, then
  standard MHA with `K = [K_nope ‖ k_pe]` per head.
- **absorbed (matrix-absorption)**: fold `W_UK` into the query
  (`q_nope' = q_nope @ W_UK^T` per head) and attend in latent space
  (score = `q_nope' · c_kv + q_pe · k_pe`), fold `W_UV` into the output —
  the memory-bound decode form real serving uses.

Cross-check gate: both forms equal within 1e-5 on random weights, and equal a
naive full-materialization oracle. (M15 wires them into a DeepSeek arch; M13
pins the math so the riskiest kernel work has a trusted reference early —
roadmap flagged MLA-on-SM120 as the highest kernel risk.)

### D4 — FlashInfer adapter (`flashinfer_gpu.py`), written blind, contract-pinned

Deferred `import flashinfer` inside the constructor; coverage-omitted by the
existing `*_gpu.py` glob. Uses the paged wrappers:

- prefill/chunk: `BatchPrefillWithPagedKVCacheWrapper` — `plan(qo_indptr=[0,T],
  paged_kv_indptr=[0,P], paged_kv_indices=page_table[:P],
  paged_kv_last_page_len=[llp], num_qo_heads, num_kv_heads, head_dim,
  page_size, causal=True)`; FlashInfer aligns causality bottom-right for
  rectangular qo/kv, matching our chunk-over-cached-prefix semantics.
- decode (T == 1): `BatchDecodeWithPagedKVCacheWrapper` with the same paged-kv
  arrays.
- The pool tensors are NHD layout per page (`[page, page_size, heads, dim]`)
  — FlashInfer's `kv_layout="NHD"` with our `k`/`v` slices
  (`pool.k[layer]`, `pool.v[layer]`).

**Contract tests (CPU)**: a fake `flashinfer` module injected via
`sys.modules` records plan/run calls; tests pin the indptr/indices/
last-page-len arithmetic (incl. partial last pages and multi-page tables),
the plan-before-run ordering, and dtype/shape passing. `tests/gpu/`
mirrors the contract suite 1:1 against real kernels with
`TorchAttentionBackend` as the oracle (deploy day). API drift risk is
accepted and bounded: the adapter is one file, the fake pins OUR call
sequence, and the version is pinned in the `[gpu]` extra.

### D5 — Selector (`selector.py`)

`select_backend(profile: HardwareProfile | None) -> AttentionBackend`:
`KAIRYU_ATTENTION_BACKEND` accepts `auto`, `torch`, `flashinfer`,
`flashattention3`, and `flashattention4`. An explicit backend is strict:
dependency, architecture, and tensor-shape validation happens before the
replica serves, and a failure names the unmet requirement rather than silently
falling back. `auto` uses `profile.kernel_tier` — `torch` for CPU/unknown and
the stable FlashInfer path for `fa2`/`full` tiers. `build_engine_loop(
model_path=...)` calls it with `probe()` so deploy day remains config-free.

Selection produces a reportable decision, not only an implementation name:
requested and resolved names, source (`env` or hardware profile), rationale,
and phase components. `/backends` exposes this decision so a hybrid path is
never mislabeled as a monolithic backend.

`auto` does not promote FA3 or FA4 merely because the package imports or a
single timing sample is faster. Promotion requires retained evidence for the
exact model/dtype/shape/GPU profile: correctness parity plus interleaved warm
samples whose execution order and OS-jitter observations are recorded. With
no qualifying evidence, the stable fallback above is intentional. If that
profile-selected optional implementation cannot be constructed, `auto` alone
falls back to torch and retains the sanitized failure type in its actual
decision. Explicit selections never fall back. The constructed decision is
carried by the backend/runner/engine and is what `/backends` reports; health
inspection does not re-infer a different implementation from environment
state. TP startup exchanges a canonical decision identity containing the
requested/resolved names, source, components, and architecture, so a rank-local
fallback or heterogeneous kernel path cannot silently create a mixed group.

### D6 — FlashAttention phase adapters (`flashattention_gpu.py`)

FA3 and FA4 own prefill only. FlashInfer continues to own paged decode and its
CUDA-graph-safe plan/replay contract. This separation is selected for measured
serving behavior and is made explicit in the decision's `prefill`, `decode`,
and `kv_mode` components.

- **FA3:** supported through the official upstream `hopper/` package, pinned
  to tag `fa4-v4.0.0.beta24`, commit
  `849f660f73b176e5ad5670e7f822c7fa9f3eaf8b`. Kairyu exposes it on SM90;
  its fake-module contract pins imports, APIs, shapes, GQA, causality, and
  fail-closed architecture handling without treating fake coverage as measured
  performance.
- **FA4:** the optional `flashattention4` install extra pins
  `flash-attn-4[cu13]==4.0.0b24`, upstream's recommended CUDA 13 variant.
  SM90/SM100/SM110 use FA4's direct paged-KV interface. SM120 keeps the
  original page table as the source of truth and performs explicit
  device-to-device page materialization for FA4 prefill; no host gather is
  allowed. Beta24 caches its selected architecture process-wide. Construction
  therefore compares the selected device's real capability, the Kairyu
  profile, `FLASH_ATTENTION_ARCH`, and the upstream global cache before model
  loading, and every call enters that device's CUDA context. Two FA4 roles on
  different SM families must use separate processes; the second backend fails
  startup rather than compiling or launching a kernel for the first role's SM.
- **Shared semantics:** both adapters preserve GQA head grouping,
  bottom-right rectangular causality, page identity, dtype, and the existing
  CUDA-graph boundary. Unsupported architectures and shapes are rejected
  before serving. Prefill failures are not allowed to change decode ownership
  or trigger an implicit torch/FlashInfer fallback.

The deployment builder owns process lifecycle. It calls the public
`kairyu-proc` startup hook before the application lifespan yields, so an
explicit dependency, architecture, or model-shape failure prevents the socket
from serving. The parent receives the child's actual decision over a versioned
optional startup frame. Requests are tied to a monotonic child generation:
death/reset delivers one terminal error to that generation only, and shutdown
cancels and drains an in-progress model load before releasing process
ownership.

## 3. Non-goals and fallbacks

- MLA wired into an architecture (M15); FlashMLA / Triton MLA kernels
  (deploy-day, SM90/100; SM120 fallback is G4 M-B1).
- Attention dtype policies beyond the pool's dtype.
- MLA and custom/Torch backends without native ragged-prefill capability retain
  the established sequential prefill path. A single prefill request also stays
  sequential to avoid plan/packing overhead. Mixed prefill/decode steps run one
  prefill chain first and then the existing eager/graph decode chain.
- Ragged prefill CUDA-graph capture is not claimed: FlashInfer planning is a
  host phase and ragged shapes vary. Decode graph capture remains unchanged.
- Automatic FA3/FA4 promotion without retained profile-specific evidence.

## 4. Phasing

1. Protocol + torch backend extraction (all M12 parity suites must stay green
   — the extraction is behavior-free).
2. MLA reference math + equivalence gate.
3. FlashInfer adapter + fake-module contract tests + `tests/gpu/` mirror.
4. Selector + wiring (`build_engine_loop` uses `probe()`).
5. Issue #277: FA3/FA4 prefill adapters, hybrid phase reporting, strict
   validation, packaging, Helm selection, and retained-evidence policy.

## 5. Verification

- Full suite green (501 baseline); M12 hf-parity suites unchanged.
- MLA: two forms ≡ naive oracle (1e-5), shapes for GQA-less MHA latents.
- Fake-flashinfer: indptr math pinned for 1-page, partial-last-page, and
  many-page tables; plan/run ordering; decode-vs-prefill wrapper choice.
- Selector: env override, CPU→torch, fa2/full→flashinfer (constructed lazily —
  no import unless selected).
- Issue #224: CPU mixed-length/cache/chunk/preemption/KV-ownership parity;
  real SM120 BF16 FlashInfer parity; one prefill plan and one run per layer
  instead of one chain per request; Qwen3-32B TP8 all-rank structural evidence.
  Fixed wall-clock thresholds are diagnostic only because OS jitter must not
  decide this gate.
- Selector/Helm: all five public values render and resolve; invalid or
  unavailable explicit selections fail without fallback; an unavailable
  profile-selected optional backend falls from `auto` to torch and the actual
  decision is reported; empty Helm default omits the environment variable and
  therefore uses `auto`.
- FA3/FA4 fakes: import/version/architecture/shape errors; GQA and bottom-right
  causal arguments; direct-paged versus D2D-materialized page identity; hybrid
  phase report.
- GPU on the available host: FA4 on SM120 compares against the FlashInfer
  oracle and serves Qwen3-32B at TP8. Auto-promotion evidence uses interleaved
  warm samples and retains order plus jitter metadata with the result. FA3 has
  fake-module/API-contract coverage only in this environment; no SM90 device
  or performance result is claimed, and no automatic FA3 policy is inferred
  from that fail-closed contract coverage.

## 6. Review record (binding amendments)

FlashInfer adapter (all verified against docs.flashinfer.ai 0.6.x + TGI/sglang
issue reports):
- prefill `plan()` takes **`head_dim_qk`** (+`head_dim_vo`), NOT `head_dim`
  (that kwarg is decode-wrapper-only) — the fake pins both spellings.
- Both wrappers take the same **workspace buffer** first ctor arg (394 MiB
  uint8, matching vLLM's serving default; FlashInfer's documented 128 MiB
  recommendation overflowed at 190,840,832 bytes on the Qwen3-32B 1,024-token
  chunked-prefill gate). The adapter is stateful — ONE shared instance across
  all layers (DenseDecoder threading provides this; load-bearing).
- `plan()` must pass `q_data_type`/`kv_data_type` explicitly (defaults are
  fp16 — silent mismatch with bf16 pools); fp32 pools are NOT a FlashInfer
  kernel path — the tests/gpu mirror constructs fp16/bf16 pools.
- indptr/indices/last_page_len are **int32 torch tensors**; indptr +
  last_page_len on HOST, indices on device; last_page_len =
  (seq_len-1) % page_size + 1. Tuple (pool.k[layer], pool.v[layer]) is a
  valid NHD paged_kv_cache; run() returns [T, H, D] → reshape to [T, H*D].
- `causal=True` is bottom-right aligned: correct iff `chunk_start + T ==
  seq_len` — assert it (all call sites satisfy it today).
- Plan cached per (page_table, seq_len, chunk_start, T) — layer 0 plans,
  layers 1..N-1 run (per-layer replanning is pathological).
- Decode wrapper kept with `use_tensor_cores=True` (GQA fast path).

MLA reference (verified vs arXiv:2405.04434):
- **sm_scale = (qk_nope_head_dim + qk_rope_head_dim)^-0.5 in BOTH forms and
  the oracle** (the absorbed form's 576-dim layout makes SDPA defaults wrong
  — and default-vs-default would pass the gate wrongly); scale is a
  parameter (M15's YaRN mscale absorbs into it).
- **k_pe is a single shared head** broadcast across query heads (q_pe is
  per-head) — the pool variant is a 1-kv-head pool of width
  kv_lora_rank + qk_rope_head_dim, stored POST-RoPE (decompress form must
  not re-rope).
- Input contract: already-projected, already-roped (q_nope [T,H,d_nope],
  q_pe [T,H,d_rope]); q-side LoRA is M15 wiring. Equivalence suite covers
  d_nope != v_head_dim.

Seam safety: backends are plain objects, NEVER nn.Module (a workspace buffer
submodule would corrupt state_dict names); no shared-instance default arg
(`backend or TorchAttentionBackend()`); positions-contiguity assumption
documented on the protocol; invalid env override fails loudly.

Issue #277 extension (implemented 2026-07-30; retained evidence 2026-07-29):
- Public selection values are exactly `auto`, `torch`, `flashinfer`,
  `flashattention3`, and `flashattention4`; explicit means strict, while only
  `auto` may use a fallback.
- FA3/FA4 are composite serving decisions: FlashAttention prefill and
  FlashInfer decode. Health output must expose both components, dependency
  versions, GPU architecture decision, and selection source.
- FA4's SM90/SM100/SM110 direct-paged path and SM120 device-materialized path
  are separate contracts. Neither may renumber or mutate Kairyu's page table.
- An auto-policy promotion is a release artifact backed by retained,
  profile-specific correctness/performance evidence, not a runtime
  opportunistic benchmark.

The retained SM120 Qwen3-32B prefill artifact is
`bench/results/attention-backends-qwen3-32b-sm120-2026-07-29.json`. It records
24 alternating AB/BA pairs for both TP4-local (16 Q / 2 KV heads) and TP8-local
(8 Q / 1 KV head) shapes, including every `perf_counter_ns` timestamp. Output
parity passed with maximum absolute error 0.0009765625. FlashInfer was faster
in all 24 pairs for both shapes: median latency was 0.108619 ms versus
0.2637465 ms at TP4-local and 0.083225 ms versus 0.2617595 ms at TP8-local.
The exact one-sided paired sign test therefore supplies no FA4-promotion
evidence, so SM120 `auto` keeps FlashInfer. No effect-size cutoff or discarded
jitter sample is used.

The companion TP8 serving artifact is
`bench/results/attention-backends-serving-qwen3-32b-sm120-2026-07-29.json`.
The same exact-source image and pinned Qwen3-32B revision returned an identical
deterministic response and usage under explicit FlashInfer and FA4 selection.
It retains every request timing and token count from balanced BA/AB ordering:
two runs per backend, eight concurrent requests, and 32 output tokens per
request. Median-of-run throughput was 115.795 versus 116.45 output token/s and
mean TPOT was 39.9785 versus 40.213 ms/token for FlashInfer versus FA4. The
large within/order variation and two-run sample make this diagnostic evidence
of executable serving only, not a backend-performance conclusion; the 24-pair
kernel artifact above remains the binding `auto` decision.
