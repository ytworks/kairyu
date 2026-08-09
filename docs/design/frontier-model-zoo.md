# Frontier text model zoo reference path

Status: **Implemented** (2026-08-09)

## Scope

Kairyu recognizes and serves these released text architectures through its
single-device scheduler:

- Qwen3.6 dense: `Qwen3_5ForCausalLM` and
  `Qwen3_5ForConditionalGeneration`
- Qwen3.6 MoE: `Qwen3_5MoeForCausalLM` and
  `Qwen3_5MoeForConditionalGeneration`
- DeepSeek V4 Flash/Pro: `DeepseekV4ForCausalLM`
- Kimi K3 text tower: `KimiLinearForCausalLM` and
  `KimiK3ForConditionalGeneration`

Qwen3.8 is not included because no public checkpoint/config contract was
available when this support landed.

## Decisions

### FZ-D1 — Recompute complete text sequences

The new families mix full attention with recurrent, compressed, or delta
attention state. They therefore use `RecomputeModelRunner`: each emitted token
runs the complete prompt plus committed completion through the text decoder.
This is slower but preserves Kairyu scheduling and sampling semantics without
pretending that recurrent state is ordinary paged KV.

The reference path is eager and single-device. It rejects CUDA graphs,
speculative verification, DRAM KV tiering, TP/EP, and P-D separation. Those
features require architecture-specific cache and sharding work with separate
correctness/performance evidence.

### FZ-D2 — Load only the text tower

Qwen multimodal wrapper checkpoints map
`model.language_model.*` into the text-only causal decoder. Kimi wrapper
checkpoints expose `language_model`; image inputs and vision weights are not
part of this path.

Qwen3.6 and unquantized local checkpoints use the pinned transformers 5.12
text modules with Kairyu's checkpoint reader. DeepSeek V4 block-FP8 and Kimi
K3 nested MXFP4 public checkpoints are delegated to their official
transformers/custom-code loaders, then wrapped by the same Kairyu runner.
Kimi's published path consequently requires its checkpoint Python files plus
`fla-core` and `compressed-tensors`.

### FZ-D3 — One-GPU validation boundary

Committed tests cover:

- Qwen3.6 dense and MoE outer-checkpoint logits parity at tiny sizes
- DeepSeek V4 tiny checkpoint logits parity
- DeepSeek block-FP8 and Kimi nested-MXFP4 official-loader dispatch contracts
- Kimi public text-config parsing
- complete-history recomputation and backend runner selection

No claim is made for full public-checkpoint fit, throughput, multimodal input,
multi-GPU execution, or optimized recurrent-state caching. The released
checkpoints exceed the available single-GPU validation budget; those gates
remain separate work.
