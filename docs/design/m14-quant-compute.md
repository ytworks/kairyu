# M14 Design: Quantization Compute Paths — CPU References + Production GPU Kernels

Status: **GPU-validated** (2026-07-27). Implemented 2026-07-03; reviewed —
APPROVE-WITH-AMENDMENTS (1-reviewer panel; formats
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

## 7. GPU binding amendment (2026-07-27, issue #205)

The original D2/D4 deploy-day deferral is closed. The following rules supersede
the stub/fallback wording above:

- CUDA `QuantizedLinear.forward` is the production dispatch point. CPU continues
  to use the explicit correctness oracle; CUDA never calls `dequantize()` and
  has no `F.linear` fallback.
- FP8 performs fused dynamic-per-token or checkpoint-static activation
  quantization and selected native scaled GEMM, with a fused Triton ragged-shape
  path. INT8 performs fused dynamic-per-token activation quantization and an
  int32-accumulating tiled Triton GEMM.
- AWQ and GPTQ unpack only the current packed weight tile in registers inside a
  fused W4A16 Triton GEMM. They never materialize an `[out, in]` floating weight.
- ModelOpt NVFP4 is native W4A4: FlashInfer quantizes activations to FP4 and runs
  its FP4 GEMM with the checkpoint `input_scale * weight_scale_2` alpha. Weight
  scale swizzling/padding is cached once as non-persistent device buffers.
- Unsupported dtype, packed layout, compute capability, or missing kernel
  dependency fails loudly. Capability floors are SM89 for FP8, SM80 for
  INT8/AWQ/GPTQ, and SM100 for NVFP4. Quantized MLA is rejected at load time
  because its current latent-attention implementation reads an unquantized
  projection weight directly. Quantized TP remains the separately tracked
  #152 boundary.

GPU oracle tests cover production `module(x)` for all five formats, including
ragged shapes and a monkeypatched `dequantize()` that raises. Separate
full-checkpoint tests load and greedily generate through the whole CUDA engine
for all five formats.

### 7.1 RTX PRO 6000 Blackwell evidence

CUDA-event medians below include activation quantization, shape M×K by K×N with
K=N=4096, 50 warmups and 200 samples. “old” is the removed full-weight
dequantize-plus-BF16 path; “BF16” is unquantized `F.linear`. External results use
the official vLLM 0.26.0 CUDA 12.9 image at the pinned image digest on the same
GPU. vLLM's stable INT8 scaled-MM operation explicitly rejects SM120; its AWQ
and GPTQ selector was not used because this kernel-level comparison does not
construct vLLM model-layer metadata.

| format | M | Kairyu fused ms | old ms (speedup) | BF16 ms | vLLM ms (Kairyu/vLLM) | fused temporary |
|---|---:|---:|---:|---:|---:|---:|
| FP8 | 1 | 0.11891 | 0.15667 (1.32×) | 0.02408 | 0.05229 (2.27×) | 0.021 MiB |
| FP8 | 128 | 0.11754 | 0.16565 (1.41×) | 0.03168 | 0.05325 (2.21×) | 2.501 MiB |
| INT8 | 1 | 0.10899 | 0.15322 (1.41×) | 0.02389 | unsupported on SM120 | 0.012 MiB |
| INT8 | 128 | 0.11770 | 0.16426 (1.40×) | 0.03154 | unsupported on SM120 | 1.500 MiB |
| AWQ | 1 | 0.08890 | 2.59094 (29.15×) | 0.02362 | — | 0.008 MiB |
| AWQ | 128 | 0.29862 | 2.62294 (8.78×) | 0.03349 | — | 1.000 MiB |
| GPTQ | 1 | 0.10749 | 0.92614 (8.62×) | 0.02618 | — | 0.008 MiB |
| GPTQ | 128 | 0.26109 | 0.93640 (3.59×) | 0.03418 | — | 1.000 MiB |
| NVFP4 | 1 | 0.35595 | 0.57914 (1.63×) | 0.02365 | 0.12963 (2.75×) | 0.050 MiB |
| NVFP4 | 128 | 0.35286 | 0.59862 (1.70×) | 0.03136 | 0.13307 (2.65×) | 2.282 MiB |

Correctness is measured on the same matrices. Fused-vs-oracle isolates kernel
error; quantized-vs-BF16 includes the expected format loss:

| format | M | fused vs quantized oracle rel-RMSE | quantized vs BF16 rel-RMSE |
|---|---:|---:|---:|
| FP8 | 1 / 128 | 0.00164 / 0.00166 | 0.03839 / 0.03752 |
| INT8 | 1 / 128 | 0.00166 / 0.00170 | 0.01327 / 0.01261 |
| AWQ | 1 / 128 | 0.00236 / 0.00236 | 0.09812 / 0.10055 |
| GPTQ | 1 / 128 | 0.00236 / 0.00236 | 0.09812 / 0.10055 |
| NVFP4 | 1 / 128 | 0.00276 / 0.00597 | 0.13216 / 0.13436 |

The external vLLM run reports BF16-relative RMSE 0.03743/0.03748 for FP8 and
0.13489/0.13475 for NVFP4 at M=1/128, consistent with Kairyu's format loss.

Raw provenance and p95/minimum data:
`bench/results/quant-gemm-rtxpro6000-2026-07-27.json` and
`bench/results/quant-vllm-rtxpro6000-2026-07-27.json`.

## 8. Contextual construction amendment (2026-07-30, issue #228)

The three-argument factory remains a supported compatibility input, but the
production factory now binds the execution environment once and receives a
typed identity for every projection:

- canonical checkpoint module name and semantic role;
- target/draft model scope, layer index, and routed/shared expert identity;
- logical target device and compute dtype;
- TP mode, rank/world size, shard axis, and local/global dimensions; and
- an immutable snapshot of available linear-kernel families.

Target-model, MoE, MLA, EAGLE, and MTP construction all provide this context.
EAGLE and MTP use checkpoint-canonical identities for policy while retaining
their established local `state_dict` names. Context and the resolved selection
are plain Python attributes, never child modules, parameters, or persistent
buffers, so checkpoint keys and tensor ownership do not change.

Selection is fail-closed. The default policy first applies hard architectural
constraints, then checkpoint `ignore` entries (literal prefixes or explicit
`re:` full matches), then the accuracy-preserving dense defaults for routers,
output heads, and draft fusion projections. Router and vocabulary near-ties can
change selected experts or tokens, while draft-fusion error feeds speculative
acceptance/correction; keeping these small projections dense is therefore a
numerical invariant, not merely compatibility with older construction. All
remaining projections use the checkpoint format. A specialized, evidence-backed
policy may select another compatible format or kernel explicitly; it cannot
override an architectural prohibition.

AWQ/GPTQ W4A16 consumes checkpoint FP16 scales without first rounding the
dequantized weights to the resident activation dtype. Both FP16 and BF16 model
activations enter the tensor-core dot as FP16 operands and accumulate in FP32;
the output is cast back to the resident activation dtype. This preserves the
checkpoint scale precision instead of discarding roughly three mantissa bits
for BF16 serving. Finite BF16 inputs outside FP16's range are saturated to
±65504 before conversion, while existing infinities and NaNs remain non-finite:
W4A16 accepts the narrower exponent range explicitly without hiding an upstream
overflow or turning a finite activation into an infinity at this boundary.
Quantized CUDA selections may choose only a fused CUDA family. They never fall
back to the CPU oracle, dense `F.linear`, full-weight dequantization, or an
emulation kernel.

Capability probing happens once when a runtime factory is created. It records
the target SM, a successful Triton import, and the presence of FlashInfer's
`SfLayout`, `mm_fp4`, and `nvfp4_quantize` symbols. The selected kernel family
and decision reason are exposed on each projection as `linear_selection`;
failures include the canonical name, scope, role, device/dtype, TP placement,
local/global dimensions, and the capability-probe result. The resolved kernel
callable is bound before serving rather than selected on every ordinary
forward.

The single-device loader, P–D role builders, and TP rank builder pass their
resolved device, dtype, and placement into the factory. Offline checkpoint
shape validation deliberately keeps the factory's logical target at CPU while
allocating tensors under `torch.device("meta")`: it performs no CUDA probe or
allocation and does not claim runtime kernel readiness.

## 9. Quantized draft-head amendment (2026-07-31, issue #234)

Draft quantization is a checkpoint-owned policy rather than an accidental copy
of target construction. External EAGLE reads its own standard
`quantization_config`; embedded MTP inherits target quantization unless its
`draft_quantization_config` explicitly overrides it, with `null` or `{}` meaning
dense. Both loaders require semantic `config.json` metadata and fail before
loading tensors when caller and checkpoint geometry disagree.

The first supported draft dialect is deliberately narrow: one
compressed-tensors Linear group with FP8 per-channel weights and dynamic
per-token FP8 activations. EAGLE and MTP projections use their existing
checkpoint-canonical contextual identities. Architectural dense exclusions
remain authoritative; MTP metadata must explicitly ignore the MoE router and
MLA `kv_b_proj`. Other methods, strategies, target sets, or missing exclusions
are rejected rather than silently repacked or routed through a dense fallback.

Runtime loading accepts packed payloads only. Weight shapes and FP8 dtypes are
exact, scale tensors are finite and positive before their value-preserving FP32
ABI normalization, and dense members retain the requested compute dtype.
Offline dense-to-FP8 conversion is a separate explicit helper. CUDA tests call
the fused production modules while making any `dequantize()` path fatal; the
real Qwen3-32B/EAGLE-3 gate separately reports dense-relative acceptance,
latency, memory, target-corrected committed-token goodput, and complete
checkpoint provenance. Its target correctness gate requires exact teacher
prefixes and bounds any cross-shape correction divergence using reciprocal
selected-token log-probabilities within 0.25 nat; it does not silently discard
or relabel divergent target output.

## 10. NVFP4 accuracy-profile amendment (2026-08-08, issue #355)

NVFP4 serving keeps the checkpoint's static activation scale and fused MoE path
by default. An opt-in `nvfp4_accuracy_profile` may select `first:N`, `last:N`,
`down_proj`, or `shared_experts` projections independently for two accuracy
experiments: dynamic per-token NVFP4 activation scaling, or FP8 runtime
execution. FP8 pinning first loads the canonical ModelOpt NVFP4 members, then
dequantizes and requantizes that projection once after TP/EP checkpoint slicing.
It spends additional resident memory and removes FP4 activation error; it does
not pretend to restore weight information already lost in the NVFP4 checkpoint.

Dynamic NVFP4 uses FlashInfer's per-token quantizer with the fixed inverse base
multiplier `1 / (448 * 6)`. The returned per-token decode multiplier is applied
row-wise after `mm_fp4`, while the GEMM alpha retains only the checkpoint weight
global scale. CPU and SM120 tests bind this equation to independent reference
quantization. Row-parallel FP8 pins synchronize the existing per-token FP8 amax
across ranks before quantization.

`saturation_counters: true` adds two non-persistent, device-resident int64
counters to each selected NVFP4-source projection. A block is saturated only
when its group-16 amax exceeds `input_scale * 6 * 448`; snapshots synchronize
only in the explicit operator probe. Accurate per-expert attribution requires
the existing non-fused EP execution path while observation is active. Mixed
FP8 or dynamic routed experts likewise use that already-correct all-to-all
path; untouched homogeneous layers retain FlashInfer fused MoE.

The companion `verification/l1/correctness/g4_ma1_nvfp4_correctness_bench.py` reuses the pinned formal M-A1
reference teacher rollout without changing its retained schema. Each profile
report includes all 1,024 teacher positions, free-running diagnostics, actual
unique resident storage per rank, projection-format counts, and per-projection
saturation. Its `curve` command accepts only reports with identical source,
checkpoint, model, world size, and reference identity.

## 11. Real-checkpoint parity amendment (2026-08-08, issue #356)

The tiny random-model gate remains a format-oracle smoke, not sufficient
real-model evidence. The retained real-checkpoint gate uses the unquantized
`Qwen/Qwen2.5-1.5B-Instruct@989aa798` HF model as one shared reference and
loads three separately published candidates through Kairyu's production CUDA
path: Red Hat AI compressed-tensors dynamic INT8 W8A8, Qwen AWQ GEMM W4A16,
and Qwen GPTQ-v1 W4A16. Repository revisions and complete safetensors digests
are binding.

`verification/l1/correctness/parity_hf.py --reference-model-path` permits the HF reference checkpoint
to differ from the Kairyu candidate only when architecture geometry, vocabulary,
and the actual fixed prompt-token sequences match, the reference is unquantized,
and the candidate declares quantization. Ordinary same-checkpoint A1/A2 behavior
and verdicts are unchanged.

Each arm scores 64 fixed prompts at 16 independent teacher-forced positions and
retains top-64 token-ID logprobs. For cross-checkpoint quantization, exact token
agreement and agreeing-token logprob drift are diagnostics: format loss makes
the A2 same-weight 0.25-nat distribution bound inapplicable. The binding gate
requires all 1,024 positions, no missing candidate token, and zero disagreements
outside the measured BF16 reference tie floor (never below 0.125 nat). The
`verification/l1/correctness/quant_checkpoint_parity_bench.py` assembler independently replays raw
positions, verifies the exact INT8/AWQ/GPTQ checkpoint ABI and revisions, and
rejects dirty or mixed-source arms.

The real INT8 arm exposed an activation-quantization discrepancy that the
small random gate missed. Dynamic W8A8 uses the CPU oracle's FP32
`torch.round(x / scale)` tie-to-even contract exactly. Triton's ordinary `/`
may lower to an approximate reciprocal and move a half-integer across its tie,
so the INT8 branch uses libdevice round-to-nearest division for both the
clamped amax-derived scale and the scaled activation before `rint`.
The fused GEMM also performs bias addition with explicit round-to-nearest
addition, preserving the oracle's rounded scale product instead of allowing
the compiler to fuse the final multiply and bias into one FMA.
FP8 keeps its existing division because it does not perform integer rounding.

The retained SM120 run at clean source `6c4ddc7` completed every required
position and kept the measured 2.875-nat BF16 tie floor unchanged. INT8 has
zero substantive disagreements and passes; AWQ has 5 and GPTQ 2, so both W4
arms and the combined verdict remain formal FAIL. A same-GPU BF16 replay of
all seven W4 substantive positions through the independent CPU unpack/dequant
oracles matched every production token. These W4 residuals are therefore
published checkpoint quantization loss, not relabeled parity or missing data.
The raw authority, combined summary, and SHA-bound `oracle-replay.json` live under
`bench/results/issue-356-qwen25-1.5b-quant-parity-sm120/`.
