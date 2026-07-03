# M15 Design: MoE + MLA Architectures — Qwen3-MoE, DeepSeek-V3 Class

Status: **Reviewed — REVISE-THEN-IMPLEMENT applied** (1-reviewer panel with
empirical verification against transformers 5.12.1 tiny models, 2026-07-03;
§6 amendments are binding — the MLA math itself verified to 3.7e-9 against
DeepseekV3Attention before implementation).
Milestone: M15 (roadmap Track E4 local half; goal G4 architecture base)
Date: 2026-07-03
Depends on: M12 (DenseDecoder pattern, parity harness), M13 (`mla_torch`
reference — the trusted oracle), M14 (`linear_factory` for quantized experts).
Consumed by: M16 (EP dispatch over the same expert modules), G4 gates.

## 1. Goal

Two of the four production model classes are MoE (roadmap §1). Implement the
architectures locally with the same transformers-parity discipline as M12:

- **Qwen3-MoE class** (`Qwen3MoeForCausalLM`): sparse SwiGLU experts with
  softmax top-k routing (`norm_topk_prob`), GQA attention (reuses M12 modules).
- **DeepSeek-V3 class** (`DeepseekV3ForCausalLM`): MLA attention over a
  compressed-latent KV pool variant + sigmoid-routed MoE with grouped top-k
  (`noaux_tc`), shared experts, dense first layers.

Flagship gate: tiny-config fp32 logits < 1e-4 vs transformers AND full-engine
greedy == `hf.generate` — per architecture (the M12 harness extended).

## 2. Key design decisions

### D1 — MoE block (`models/moe.py`), HF module names

`Qwen3MoeSparseBlock`: `gate` (router linear, bias-free), `experts`
(ModuleList of per-expert SwiGLU built via `linear_factory` — M14 quantized
experts come free), routing = softmax over router logits → top-k →
optional re-normalization (`norm_topk_prob`) → weighted sum of expert outputs.
Token-loop reference implementation (gather tokens per expert) — the EP
dispatch/combine (M16) replaces the loop, not the math. Config additions:
`num_experts`, `num_experts_per_tok`, `moe_intermediate_size`,
`norm_topk_prob`, `decoder_sparse_step`/`mlp_only_layers` (which layers are
sparse).

`DeepseekV3MoeBlock`: sigmoid scores + `e_score_correction_bias`, grouped
top-k (`n_group`/`topk_group`: score groups, keep top groups, then top-k
within survivors), `routed_scaling_factor`, `n_shared_experts` (dense shared
expert added unconditionally), `first_k_dense_replace` (leading dense
layers). Weight names mirror HF (`mlp.experts.N.{gate,up,down}_proj`,
`mlp.shared_experts.*`, `mlp.gate.weight`, `mlp.gate.e_score_correction_bias`).

### D2 — MLA attention module (`models/mla.py`) over a latent KV pool

`MlaAttention` wires M13's verified reference into a module: projections
`q_a_proj`/`q_a_layernorm`/`q_b_proj` (q LoRA), `kv_a_proj_with_mqa`/
`kv_a_layernorm`/`kv_b_proj`, `o_proj`. The pool stores the per-token
`[c_kv ‖ k_pe]` (post-RoPE k_pe) — a `PagedKVPool` with `num_kv_heads=1`,
`head_dim = kv_lora_rank + qk_rope_head_dim` (M13 pinned this shape); only
the `k` tensor is used (`v` empty) — recorded so M18's serde knows.
Attention uses the ABSORBED form for decode-shaped chunks and the DECOMPRESS
form for prefill chunks (both verified equal; the split mirrors real serving).
Scale = `mla_scale(...)`, times the YaRN `mscale²` factor when rope_scaling
is yarn (D3).

### D3 — DeepSeek rope: yarn support scoped to what parity needs

DeepSeek-V3 checkpoints ship `rope_scaling.type: yarn`. M15 implements the
HF yarn inv_freq transform + `attention_factor` (mscale) so real checkpoints
load correctly; the tiny parity fixtures run BOTH with rope_scaling None and
with a yarn config (transformers is the oracle either way).

### D4 — Registry + config + engine integration

`parse_model_config` gains the MoE/MLA fields (nullable — dense models
unaffected); `MODEL_BUILDERS` registry maps the two new architectures to a
`SparseDecoder` (same skeleton as DenseDecoder, per-layer mlp chosen sparse
vs dense, attention chosen GQA vs MLA). `PagedModelRunner` is unchanged
(pool shape differences live behind `PagedKVPool.for_cache`, which reads the
MLA fields from config). `validate_tp_degree`: MLA models report
`num_key_value_heads=1` — recorded (TP for MLA is attention-DP, an M16 note).

## 3. Non-goals

- EP dispatch/combine over Communicator (M16 — the expert modules are its
  inputs); expert-parallel weight sharding (M16).
- MTP heads (M17); MLA GPU kernels (deploy day; G4 M-B1 for SM120).
- Aux-loss / load-balancing statistics (inference only).

## 4. Phasing

1. Config fields + Qwen3-MoE block + SparseDecoder + parity (logits +
   full-engine greedy vs transformers tiny).
2. MLA module + latent pool variant + DeepSeek-V3 (rope_scaling None) parity.
3. yarn rope + parity with a yarn fixture; loader/registry wiring
   (`model_path=` works for both archs end-to-end).

## 5. Verification

- Tiny-config parity per arch: fp32 logits < 1e-4; full-engine greedy ==
  hf.generate (chunked prefill + radix reuse + page-crossing decode).
- MoE routing unit tests: top-k selection, norm_topk_prob renormalization,
  grouped top-k group masking, correction bias affects SELECTION but not
  the mixing weights (DeepSeek convention), routed_scaling_factor.
- MLA module output == mla_torch reference forms on random weights; absorbed
  vs decompress split produces identical engine outputs.
- Quantized MoE experts: M14's factory builds expert projections
  (one smoke gate with fp8 experts through the engine).

## 6. Review record (binding amendments, all empirically verified)

- **A1 (BLOCKING)**: transformers 5.12 stores experts FUSED in memory
  (`experts.gate_up_proj [E, 2I, H]`) but save_pretrained/hub use per-expert
  names — parity harnesses must transfer via save_pretrained → load_model
  (bit-exact roundtrip verified), never `load_state_dict(hf.state_dict())`.
- **A2 (BLOCKING)**: DeepSeek rope is INTERLEAVED (`rope_interleave=True`
  default): `x1,x2 = x[...,0::2], x[...,1::2]; out = cat([x1c-x2s, x2c+x1s])`
  with cos/sin truncated to d/2 — the half-split `apply_rope` is wrong for it.
- **A3 (BLOCKING)**: `q_lora_rank=None` → plain `q_proj` (no q_a/q_b modules);
  support both paths.
- **A4 MLA pins**: q split nope-first; kv_a output c_kv-first; pool caches
  c_kv POST-kv_a_layernorm ‖ roped k_pe; w_uk/w_uv from
  `kv_b_proj.weight.view(H, d_nope+d_v, r)` transposed; q_a/kv_a layernorm
  eps HARDCODED 1e-6 in HF (fixtures pin rms_norm_eps=1e-6); o_proj is
  [hidden, H*v_head_dim], v_head_dim independent; attention_bias gates
  q_a/kv_a/o_proj only.
- **A5 yarn**: softmax scale = qk_head_dim^-0.5 × (0.1·mscale_all_dim·ln(factor)+1)²
  (uses mscale_ALL_DIM, applied iff rope_type != default and mscale_all_dim);
  cos/sin attention_factor = get_mscale(f, mscale)/get_mscale(f, mscale_all_dim)
  (= 1.0 for real V3); yarn inv_freq ramp formula with truncate floor/ceil,
  dim = qk_rope_head_dim; rope_scaling keys pinned; dual-generation parse.
- **A6 DeepSeek routing (matched exactly)**: fp32 sigmoid + correction bias;
  group score = sum of TOP-2 corrected per group (hardcoded 2 — fixtures need
  ≥2 experts/group); top-k over corrected masked scores; mixing weights =
  UNCORRECTED sigmoid gathered; norm_topk denominator +1e-20 (Qwen has no
  eps); routed_scaling on routed only; shared = one MLP of
  moe_intermediate_size × n_shared_experts; e_score_correction_bias is a
  persistent BUFFER; dense layers < first_k_dense_replace.
- **A7 config**: DeepSeek saved configs carry head_dim=qk_rope_head_dim (hub
  originals don't — never derive hidden//heads for MLA); Qwen3-MoE hub writes
  num_experts, save_pretrained writes num_local_experts (accept both; DeepSeek
  n_routed_experts likewise); kv-pool mapping via config props: is_mla,
  kv_cache_num_heads (1), kv_cache_head_dim (r + d_rope), rope_dim; PagedKVPool
  gains v_head_dim (0 for MLA — v tensor unused, bytes_per_token honest).
- **A8 Qwen3-MoE**: sparse predicate ((idx+1) % decoder_sparse_step == 0 and
  idx not in mlp_only_layers); routing softmax in fp32 BEFORE top-k; renorm
  without eps; qk_norm extended to Qwen3MoeForCausalLM; attention identical
  to M12 Qwen3.
- **A9 fixtures**: minimum kwargs pinned; DeepSeek needs consistent
  n_routed/n_group/topk_group/top_k, v_head_dim ≠ qk dims (catches
  transposes), q_lora both int and None, num_key_value_heads=heads.
