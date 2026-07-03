# M13 Design: AttentionBackend Seam — Torch, MLA Reference, FlashInfer Adapter

Status: **Implemented** (2026-07-03). Reviewed — APPROVE-WITH-AMENDMENTS (1-reviewer panel with web
verification of the FlashInfer 0.6.x API and the DeepSeek-V2 MLA paper,
2026-07-03; amendments below are binding).
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

### D1 — Protocol: per-request-chunk attend; batching is a recorded GPU-day step

`kairyu/engine/core/attention/__init__.py`:

```python
class AttentionBackend(Protocol):
    def attend(self, query, kv_pool, layer, page_table, seq_len, chunk_start) -> Tensor:
        """query [T, heads, head_dim] -> context [T, heads*head_dim]."""
```

Exactly M12's `paged_attention` signature (designed for this extraction).
Cross-request batched planning (one FlashInfer plan for the whole step) is a
GPU-day optimization recorded in §3 — the adapter plans per chunk (indptr
arrays of length 2), which is correct and keeps the seam minimal.

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
`KAIRYU_ATTENTION_BACKEND` env override (`torch` | `flashinfer`) wins; else
`profile.kernel_tier` — `torch` for CPU/unknown, `flashinfer` for `fa2`/`full`
tiers. `build_engine_loop(model_path=...)` calls it with `probe()` so deploy
day is config-free; CPU environments keep getting the torch backend.

## 3. Non-goals

- Cross-request batched planning (GPU-day; the seam supports it — plan takes
  per-chunk arrays that generalize to batch indptr).
- MLA wired into an architecture (M15); FlashMLA / Triton MLA kernels
  (deploy-day, SM90/100; SM120 fallback is G4 M-B1).
- Attention dtype policies beyond the pool's dtype.

## 4. Phasing

1. Protocol + torch backend extraction (all M12 parity suites must stay green
   — the extraction is behavior-free).
2. MLA reference math + equivalence gate.
3. FlashInfer adapter + fake-module contract tests + `tests/gpu/` mirror.
4. Selector + wiring (`build_engine_loop` uses `probe()`).

## 5. Verification

- Full suite green (501 baseline); M12 hf-parity suites unchanged.
- MLA: two forms ≡ naive oracle (1e-5), shapes for GQA-less MHA latents.
- Fake-flashinfer: indptr math pinned for 1-page, partial-last-page, and
  many-page tables; plan/run ordering; decode-vs-prefill wrapper choice.
- Selector: env override, CPU→torch, fa2/full→flashinfer (constructed lazily —
  no import unless selected).

## 6. Review record (binding amendments)

FlashInfer adapter (all verified against docs.flashinfer.ai 0.6.x + TGI/sglang
issue reports):
- prefill `plan()` takes **`head_dim_qk`** (+`head_dim_vo`), NOT `head_dim`
  (that kwarg is decode-wrapper-only) — the fake pins both spellings.
- Both wrappers take a **workspace buffer** first ctor arg (128 MB uint8,
  named constant); the adapter is stateful — ONE shared instance across all
  layers (DenseDecoder threading provides this; load-bearing).
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
