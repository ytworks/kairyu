# M14 Design: Quantization Compute Paths — CPU References + Triton Stubs

Status: **Reviewed — APPROVE-WITH-AMENDMENTS** (1-reviewer panel; formats
verified against AutoAWQ/AutoGPTQ/vLLM/compressed-tensors source and LIVE
safetensors headers of four real Hub checkpoints, 2026-07-03; §6 binding).
Milestone: M14 (roadmap Track E1/E2 quant half, goal G2-as-amended quant matrix)
Date: 2026-07-03
Depends on: M12 (loader `linear_factory` hook, DenseDecoder), M8
(`detect_quantization` incl. NVFP4/modelopt/INT8). Consumed by: M15 (quantized
MoE experts), deploy day (Triton kernels, `pytest -m gpu`).

## 1. Goal

Make quantized checkpoints **load and RUN on CPU** via dequantize-on-forward
reference implementations — quant correctness becomes locally provable; the DC
only measures speed. Formats: FP8-E4M3 (compressed-tensors W8A8), INT8 W8A8,
AWQ W4A16, GPTQ W4A16, NVFP4 (modelopt). Triton fused kernels are written now
as `kairyu/kernels/*_gpu.py` stubs (deferred import, coverage-omitted,
`@gpu`-tested against the CPU references on deploy day).

## 2. Key design decisions

### D1 — `kairyu/quant/` reference modules: pack + unpack, quantize + dequantize

Each format module ships FOUR functions so round-trips are testable without
real checkpoints: `quantize_*` (reference quantizer — also powers the
integration fixture), `dequantize_*`, plus checkpoint-layout `pack_*`/
`unpack_*` where the storage format differs from the math format.

- **fp8.py**: native `torch.float8_e4m3fn` storage (verified working on CPU
  torch 2.12 for storage+cast; all COMPUTE goes through explicit upcast — the
  fp8 matmul that happens to run on 2.12 is never relied on). Per-tensor and
  per-channel `weight_scale`; compressed-tensors tensor names
  (`{prefix}.weight` fp8 + `{prefix}.weight_scale`).
- **int8.py**: int8 weights + per-channel scales; the CPU reference uses exact
  int32 accumulation (`torch.matmul` on int tensors) so the GPU kernel has a
  bit-exact oracle; dynamic per-token activation quant reference for W8A8.
- **awq.py**: int32 `qweight`/`qzeros` with the AWQ nibble interleave
  `[0, 2, 4, 6, 1, 3, 5, 7]`, per-group `scales` (group_size, default 128) —
  `unpack` to fp; `pack` for round-trip tests; AutoAWQ tensor names
  (`qweight`, `qzeros`, `scales`).
- **gptq.py**: row-packed int32 nibbles + `g_idx` group mapping + `qzeros`
  (+1 offset convention) — GPTQ tensor names.
- **nvfp4.py**: e2m1 4-bit LUT (±{0, .5, 1, 1.5, 2, 3, 4, 6}), two values per
  byte, FP8-E4M3 per-block scales (block 16) + a global fp32 scale; modelopt
  tensor names (`weight` packed uint8, `weight_scale` fp8 blocks,
  `weight_scale_2` global).

Bit-pattern unit tests are transcribed from the reference implementations
(hand-computed byte examples committed in the tests) — format fidelity is the
residual risk, detected loudly on deploy day by a failed load.

### D2 — `QuantizedLinear` modules (`quant/linear.py`)

One nn.Module per scheme, holding the PACKED tensors under the checkpoint's
own names (so the loader assigns by name with zero renaming) and computing
`forward` by dequantize-to-compute-dtype + `F.linear` (CPU-correct, slow).
`forward_fused` is the kernel seam: the base implementation calls the dequant
path; the GPU kernels override it (M14 stubs; wired on deploy day). A
`linear_factory(config, quant) -> Callable[[in, out, bias], nn.Module]`
selects `nn.Linear` (NONE) or the matching QuantizedLinear.

### D3 — Loader integration (M12 hook, no body changes)

`load_model` drops its "arrives in M14" guard: `linear_factory` from
`detect_quantization`; `DenseDecoder` construction takes the factory
(threaded through `Attention`/`SwiGluMlp` — projections only; embeddings,
norms and `lm_head` stay unquantized, matching every target scheme).
Parameter iteration switches from `named_parameters` to
`named_parameters + named_buffers` (packed int tensors are buffers, not
parameters — they must not appear in optimizer-facing APIs).

### D4 — Triton kernel stubs (`kairyu/kernels/`)

`fp8_gemm_gpu.py`, `awq_gemm_gpu.py`, `nvfp4_gemm_gpu.py`: deferred
`import triton`, coverage-omitted, each exposing `linear_forward(x, module)`
matching `forward_fused`; `tests/gpu/test_quant_kernels.py` compares against
the CPU dequant references (bit-exact for INT8's int32 accumulation, tolerance
for FP8/NVFP4). SM120 notes from the roadmap (99 KB smem, Triton-first) are
comments in the kernels.

### D5 — Integration gate: quantized checkpoint runs on CPU

The flagship test: quantize the M12 tiny llama with OUR reference quantizer →
write an HF-format checkpoint (config.json `quantization_config` + safetensors
with the scheme's tensor names) into tmp_path → `load_model` builds
QuantizedLinear modules → **full-engine greedy runs on CPU** → outputs within
a scheme-appropriate tolerance of the fp32 run (FP8/INT8 tight; 4-bit schemes
compared on logits drift + non-degenerate outputs, since 4-bit at hidden-64 is
lossy by construction).

## 3. Non-goals

- Activation quantization kernels beyond the W8A8 reference (deploy-day
  Triton work); KV-cache quantization (G4 E-KV gate).
- Real AWQ/GPTQ/NVFP4 hub checkpoints in CI (an `hf_hub` opt-in test loads a
  real tiny AWQ checkpoint; formats otherwise pinned by bit-pattern tests).
- MoE expert quantization wiring (M15 consumes the same factory).

## 4. Phasing

1. fp8 + int8 references + QuantizedLinear + factory (+ loader integration
   and the D5 gate for FP8/INT8).
2. awq + gptq pack/unpack + modules + D5 gates.
3. nvfp4 pack/unpack + module + D5 gate.
4. Triton stubs + tests/gpu mirror.

## 5. Verification

- Round-trip: quantize→pack→unpack→dequantize ≡ quantize→dequantize for every
  scheme; hand-computed byte examples pin the interleave/offset conventions.
- INT8: reference matmul is exactly int32-accumulated (bit-exact oracle).
- D5 full-engine gates per scheme; loader rejects unknown schemes loudly;
  fp8 storage uses native float8 dtype (upcast-only compute).
- `pytest -m gpu` mirror listed in scripts/gpu_gates (M19).

## 6. Review record (binding amendments — exact formats)

- **A1/A2 (BLOCKING, D3)**: loader iterates ``model.state_dict().keys()``
  (``named_buffers`` includes non-persistent buffers — ``rotary_emb.inv_freq``
  would KeyError every load). Quantized payloads (qweight/qzeros/packed
  uint8/fp8 weights/scales) load VERBATIM in checkpoint dtype; only tensors
  whose constructed dtype is fp32 get ``.to(dtype)``; the global
  ``model.to(dtype)`` is removed (it upcasts float8 buffers — verified);
  ``load_state_dict(assign=True)`` + re-tie lm_head after load.
- **A3 (BLOCKING, fp8)**: torch CPU fp8 cast is NON-saturating (1000.0 → NaN,
  verified) — quantize clamps to ±448 BEFORE the cast; with the clamp, RNE
  rounding matches GPU saturating kernels.
- **A4 (AWQ, verified live)**: qweight int32 ``[in, out//8]`` packed along
  OUT, nibble i of packed col j = original col ``8j + ORDER[i]``,
  ORDER=[0,2,4,6,1,3,5,7], REVERSE=[0,4,1,5,2,6,3,7]; qzeros int32
  ``[in//g, out//8]`` same packing; scales fp16 ``[in//g, out]``;
  ``w=(q-z)*s`` — NO +1 offset; reject ``version != "gemm"``.
- **A5 (GPTQ, verified live)**: qweight int32 ``[in//8, out]`` sequential
  LSB-first along IN (no reorder); qzeros ``[ceil(in/g), out//8]`` stored
  ``z-1`` (+1 restored at dequant); scales ``[ceil(in/g), out]``; ``g_idx``
  int32 ``[in]`` present even with desc_act=False; reject bits != 4 and
  ``checkpoint_format == "gptq_v2"`` (v2 drops the offset).
- **A6 (CT FP8)**: static preset = per-TENSOR weight_scale (1,) +
  input_scale; FP8_DYNAMIC = per-CHANNEL ``[out, 1]`` weight_scale, NO
  input_scale; symmetric — no zero_point tensor; infer variant from shapes.
- **A7 (CT INT8)**: per-channel symmetric ``[out, 1]``, dynamic per-token
  activations (scale = rowmax(|a|)/127); CPU torch.matmul does NOT take int8
  — upcast operands to int32 (int32 matmul verified).
- **A8 (NVFP4, verified live)**: weight uint8 ``[out, in//2]`` packed along
  K, LOW nibble = even element; bit 3 = sign, bits 0-2 = magnitude LUT
  [0,.5,1,1.5,2,3,4,6]; weight_scale fp8-e4m3 ``[out, in//16]`` row-major
  (cutlass swizzle is runtime, not storage); weight_scale_2 fp32 =
  global_amax/(6*448); block scale = cast_fp8(clamp(block_amax/6 / ws2));
  dequant = lut * fp8_scale * ws2 (MULTIPLY); quantizer mirrors vLLM RNE
  boundaries and clamps to ±6; checkpoints carry input_scale (fixture emits
  it). Compressed-tensors FP4 uses DIFFERENT names (weight_packed) and an
  INVERTED global scale — the CT-FP4 detect branch must reject loudly rather
  than flow into the modelopt module.
- **A9**: packed tensors as persistent buffers is OUR convention (AutoAWQ
  buffers, vLLM params — both exist); state_dict iteration covers both.
- **A10**: fixture quantization_config JSONs pinned per scheme (awq incl.
  version/zero_point; gptq incl. desc_act/sym; CT incl. config_groups with
  input_activations + ignore:[lm_head]; modelopt quant_algo NVFP4).
