"""#228: construction-time identity, placement, and kernel policy contracts."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from kairyu.engine.core.quant_config import (
    QuantConfig,
    QuantMethod,
    detect_quantization,
)
from kairyu.models.config import parse_model_config
from kairyu.models.eagle import EagleConfig, EagleDraftHead
from kairyu.models.llama import DenseDecoder
from kairyu.models.mtp import MtpDraftHead, load_mtp_head
from kairyu.models.parallel import build_tp_model
from kairyu.quant.linear import (
    ExpertScope,
    Fp8Linear,
    Int8Linear,
    LinearKernel,
    LinearKernelCapabilities,
    LinearRole,
    LinearSelection,
    ModelScope,
    TensorParallelMode,
    linear_factory,
    make_linear,
)

_DENSE_RAW = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "vocab_size": 96,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
}

_MOE_RAW = {
    **_DENSE_RAW,
    "architectures": ["Qwen3MoeForCausalLM"],
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "norm_topk_prob": True,
}

_DSV3_RAW = {
    **_DENSE_RAW,
    "architectures": ["DeepseekV3ForCausalLM"],
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "kv_lora_rank": 8,
    "q_lora_rank": None,
    "qk_nope_head_dim": 4,
    "qk_rope_head_dim": 4,
    "v_head_dim": 6,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "n_group": 2,
    "topk_group": 1,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
}


def _state_contract(
    module: torch.nn.Module,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {
        name: (tuple(tensor.shape), tensor.dtype) for name, tensor in module.state_dict().items()
    }


def _write_checkpoint(path, raw_config: dict, model: torch.nn.Module) -> None:
    path.mkdir(exist_ok=True)
    (path / "config.json").write_text(json.dumps(raw_config))
    save_file(
        {name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()},
        path / "model.safetensors",
    )


class _IdentityComm:
    def tensor_all_reduce(self, tensor):
        return tensor

    def tensor_all_reduce_max(self, tensor):
        return tensor


def test_dense_and_moe_contexts_use_canonical_checkpoint_identity():
    capabilities = LinearKernelCapabilities(
        frozenset({LinearKernel.TORCH_DENSE}),
        "test-logical-cuda",
    )
    factory = linear_factory(
        QuantConfig(QuantMethod.NONE),
        device="cuda:7",
        dtype=torch.bfloat16,
        capabilities=capabilities,
    )
    dense = DenseDecoder(parse_model_config(_DENSE_RAW), linear_factory=factory)
    query = dense.model.layers[0].self_attn.q_proj.linear_context
    output = dense.model.layers[0].self_attn.o_proj.linear_context
    head = dense.lm_head.linear_context

    assert query.qualified_name == "model.layers.0.self_attn.q_proj"
    assert query.model_scope is ModelScope.TARGET
    assert query.role is LinearRole.ATTENTION_QUERY
    assert query.layer_index == 0
    assert query.device == torch.device("cuda:7")
    assert query.dtype is torch.bfloat16
    assert query.capabilities is capabilities
    assert query.tensor_parallel.mode is TensorParallelMode.COLUMN
    assert output.tensor_parallel.mode is TensorParallelMode.ROW
    assert head.qualified_name == "lm_head"
    assert head.role is LinearRole.OUTPUT_HEAD
    assert head.tensor_parallel.mode is TensorParallelMode.REPLICATED

    moe = DenseDecoder(parse_model_config(_MOE_RAW), linear_factory=factory)
    block = moe.model.layers[0].mlp
    router = block.gate.linear_context
    routed = block.experts[2].up_proj.linear_context
    assert router.qualified_name == "model.layers.0.mlp.gate"
    assert router.role is LinearRole.MOE_ROUTER
    assert routed.qualified_name == "model.layers.0.mlp.experts.2.up_proj"
    assert routed.expert_scope is ExpertScope.ROUTED
    assert routed.expert_index == 2


def test_draft_contexts_keep_canonical_names_without_changing_local_state_keys():
    factory = linear_factory(QuantConfig(QuantMethod.NONE))
    eagle_config = EagleConfig(
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        draft_vocab_size=48,
    )
    plain_eagle = EagleDraftHead(eagle_config)
    contextual_eagle = EagleDraftHead(eagle_config, linear_factory=factory)
    assert _state_contract(contextual_eagle) == _state_contract(plain_eagle)
    assert all("linear_context" not in key for key in contextual_eagle.state_dict())
    assert contextual_eagle.fc.linear_context.qualified_name == "fc"
    assert contextual_eagle.fc.linear_context.model_scope is ModelScope.EAGLE_DRAFT
    eagle_q = contextual_eagle.midlayer.self_attn["q_proj"].linear_context
    assert eagle_q.qualified_name == "midlayer.self_attn.q_proj"
    assert eagle_q.model_scope is ModelScope.EAGLE_DRAFT
    assert contextual_eagle.lm_head.linear_context.role is LinearRole.OUTPUT_HEAD

    config = parse_model_config(_DSV3_RAW)
    plain_mtp = MtpDraftHead(config)
    contextual_mtp = MtpDraftHead(config, linear_factory=factory)
    assert _state_contract(contextual_mtp) == _state_contract(plain_mtp)
    layer = config.num_hidden_layers
    assert contextual_mtp.eh_proj.linear_context.qualified_name == f"model.layers.{layer}.eh_proj"
    mtp_q = contextual_mtp.decoder.self_attn.q_proj.linear_context
    assert mtp_q.qualified_name == f"model.layers.{layer}.self_attn.q_proj"
    assert mtp_q.model_scope is ModelScope.MTP_DRAFT
    assert (
        contextual_mtp.decoder.self_attn.kv_b_proj.linear_selection.reason
        == "architectural-exclusion"
    )
    expert = contextual_mtp.decoder.mlp.experts[1].gate_proj.linear_context
    assert expert.qualified_name == f"model.layers.{layer}.mlp.experts.1.gate_proj"
    assert expert.expert_index == 1
    shared = contextual_mtp.decoder.mlp.shared_experts.up_proj.linear_context
    assert shared.expert_scope is ExpertScope.SHARED
    assert (
        contextual_mtp.head.linear_context.qualified_name
        == f"model.layers.{layer}.shared_head.head"
    )
    # Context identity is canonical, while the loader-facing local keys stay
    # exactly the established MTP contract.
    assert "decoder.self_attn.q_proj.weight" in contextual_mtp.state_dict()
    assert "head.weight" in contextual_mtp.state_dict()


def test_mtp_loader_maps_all_packed_fusion_and_head_payloads(tmp_path):
    def policy(_checkpoint_quant, context):
        quant = (
            QuantConfig(QuantMethod.FP8)
            if context.role in {LinearRole.DRAFT_FUSION, LinearRole.OUTPUT_HEAD}
            else QuantConfig(QuantMethod.NONE)
        )
        return LinearSelection(quant, None, "test-draft-packed")

    factory = linear_factory(
        QuantConfig(QuantMethod.NONE),
        selection_policy=policy,
    )
    config = parse_model_config(_DSV3_RAW)
    source = MtpDraftHead(config, linear_factory=factory)
    prefix = f"model.layers.{config.num_hidden_layers}."
    checkpoint = {}
    for name, tensor in source.state_dict().items():
        if name.startswith("head."):
            target = prefix + "shared_head.head." + name[len("head.") :]
        elif name.startswith("decoder."):
            target = prefix + name[len("decoder.") :]
        else:
            target = prefix + name
        checkpoint[target] = tensor.detach().contiguous()
    save_file(checkpoint, tmp_path / "model.safetensors")

    loaded = load_mtp_head(tmp_path, config, linear_factory=factory)
    assert isinstance(loaded.eh_proj, Fp8Linear)
    assert isinstance(loaded.head, Fp8Linear)
    assert torch.equal(loaded.eh_proj.weight, source.eh_proj.weight)
    assert torch.equal(loaded.eh_proj.weight_scale, source.eh_proj.weight_scale)
    assert torch.equal(loaded.head.weight, source.head.weight)
    assert torch.equal(loaded.head.weight_scale, source.head.weight_scale)


def test_specialized_policy_selects_format_by_context():
    def policy(_checkpoint_quant, context):
        if context.role is LinearRole.ATTENTION_QUERY:
            quant = QuantConfig(QuantMethod.FP8, weight_bits=8)
        elif context.role is LinearRole.MLP_UP:
            quant = QuantConfig(QuantMethod.INT8, weight_bits=8)
        else:
            quant = QuantConfig(QuantMethod.NONE)
        return LinearSelection(quant, None, f"role:{context.role.value}")

    model = DenseDecoder(
        parse_model_config(_DENSE_RAW),
        linear_factory=linear_factory(
            QuantConfig(QuantMethod.NONE),
            selection_policy=policy,
        ),
    )
    assert isinstance(model.model.layers[0].self_attn.q_proj, Fp8Linear)
    assert isinstance(model.model.layers[0].mlp.up_proj, Int8Linear)
    assert isinstance(model.model.layers[0].self_attn.k_proj, torch.nn.Linear)
    assert model.model.layers[0].mlp.up_proj.linear_selection.reason == "role:mlp_up"
    assert model.model.layers[0].mlp.up_proj.linear_selection.kernel is LinearKernel.CPU_REFERENCE


def test_cuda_kernel_is_bound_from_the_selected_kernel_family():
    capabilities = LinearKernelCapabilities(
        frozenset(
            {
                LinearKernel.TORCH_DENSE,
                LinearKernel.CUDA_FP8_AUTO,
            }
        ),
        "test-sm120",
    )
    module = make_linear(
        linear_factory(
            QuantConfig(QuantMethod.FP8),
            device="cuda:3",
            dtype=torch.bfloat16,
            capabilities=capabilities,
        ),
        32,
        16,
        False,
        qualified_name="model.layers.4.self_attn.q_proj",
        role=LinearRole.ATTENTION_QUERY,
        layer_index=4,
        shard_dim=0,
    )
    assert module.linear_selection.kernel is LinearKernel.CUDA_FP8_AUTO
    assert module._fused_forward.__name__ == "fp8_linear_forward"


def test_checkpoint_exclusions_are_dense_and_loadable(tmp_path):
    raw = {
        **_DENSE_RAW,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": {"type": "float", "num_bits": 8},
                    "input_activations": {
                        "type": "float",
                        "num_bits": 8,
                        "dynamic": True,
                    },
                }
            },
            "ignore": [
                "model.layers.0.self_attn.q_proj",
                r"re:model\.layers\.0\.mlp\.(gate|up)_proj",
            ],
        },
    }
    quant = detect_quantization(raw)
    source = DenseDecoder(
        parse_model_config(raw),
        linear_factory=linear_factory(quant),
    )
    _write_checkpoint(tmp_path, raw, source)

    from kairyu.models.loader import load_model

    loaded, _, _ = load_model(tmp_path)
    layer = loaded.model.layers[0]
    assert isinstance(layer.self_attn.q_proj, torch.nn.Linear)
    assert isinstance(layer.mlp.gate_proj, torch.nn.Linear)
    assert isinstance(layer.mlp.up_proj, torch.nn.Linear)
    assert isinstance(layer.self_attn.k_proj, Fp8Linear)
    assert isinstance(layer.mlp.down_proj, Fp8Linear)
    assert isinstance(loaded.lm_head, torch.nn.Linear)
    assert (
        layer.self_attn.q_proj.linear_selection.reason
        == "checkpoint-ignore:model.layers.0.self_attn.q_proj"
    )


def test_unsupported_context_error_is_deterministic_and_actionable():
    capabilities = LinearKernelCapabilities(
        frozenset({LinearKernel.TORCH_DENSE}),
        "test-sm80-no-triton",
    )
    factory = linear_factory(
        QuantConfig(QuantMethod.FP8),
        device="cuda:3",
        dtype=torch.bfloat16,
        tensor_parallel_size=4,
        tensor_parallel_rank=2,
        capabilities=capabilities,
    )
    messages = []
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            make_linear(
                factory,
                32,
                16,
                False,
                qualified_name="model.layers.9.self_attn.q_proj",
                role=LinearRole.ATTENTION_QUERY,
                layer_index=9,
                shard_dim=0,
            )
        messages.append(str(caught.value))
    assert messages[0] == messages[1]
    message = messages[0]
    assert "no production kernel for fp8" in message
    assert "name='model.layers.9.self_attn.q_proj'" in message
    assert "role=attention_query" in message
    assert "device=cuda:3" in message
    assert "dtype=torch.bfloat16" in message
    assert "tp=column:2/4:dim=0" in message
    assert "local=32x16, global=32x64" in message
    assert "capabilities=test-sm80-no-triton" in message


def test_architectural_exclusion_cannot_be_overridden():
    default = make_linear(
        linear_factory(QuantConfig(QuantMethod.FP8)),
        32,
        16,
        False,
        qualified_name="model.layers.0.self_attn.kv_b_proj",
        role=LinearRole.MLA_KV_B,
        allow_quantization=False,
    )
    assert isinstance(default, torch.nn.Linear)
    assert default.linear_selection.reason == "architectural-exclusion"

    def invalid_policy(_checkpoint_quant, _context):
        return LinearSelection(
            QuantConfig(QuantMethod.FP8),
            LinearKernel.CPU_REFERENCE,
            "invalid-override",
        )

    with pytest.raises(ValueError, match="excludes quantization"):
        make_linear(
            linear_factory(
                QuantConfig(QuantMethod.NONE),
                selection_policy=invalid_policy,
            ),
            32,
            16,
            False,
            qualified_name="model.layers.0.self_attn.kv_b_proj",
            role=LinearRole.MLA_KV_B,
            allow_quantization=False,
        )


def test_loader_threads_logical_target_without_allocating_it(tmp_path):
    from kairyu.models.loader import load_model

    config = parse_model_config(_DENSE_RAW)
    _write_checkpoint(tmp_path, _DENSE_RAW, DenseDecoder(config))
    capabilities = LinearKernelCapabilities(
        frozenset({LinearKernel.TORCH_DENSE}),
        "injected-no-probe",
    )
    model, _, _ = load_model(
        tmp_path,
        dtype=torch.bfloat16,
        target_device="cuda:7",
        linear_capabilities=capabilities,
    )
    query = model.model.layers[0].self_attn.q_proj
    assert query.linear_context.device == torch.device("cuda:7")
    assert query.linear_context.dtype is torch.bfloat16
    assert query.linear_context.capabilities is capabilities
    assert query.weight.device.type == "cpu"
    assert query.weight.dtype is torch.bfloat16


def test_tp_builder_threads_rank_and_global_local_projection_shapes(tmp_path):
    config = parse_model_config(_DENSE_RAW)
    _write_checkpoint(tmp_path, _DENSE_RAW, DenseDecoder(config))
    model, _, _ = build_tp_model(
        str(tmp_path),
        tp=2,
        rank=1,
        comm=_IdentityComm(),
    )
    query = model.model.layers[0].self_attn.q_proj.linear_context
    output = model.model.layers[0].self_attn.o_proj.local.linear_context
    gate = model.model.layers[0].mlp.gate_proj.linear_context
    head = model.lm_head.linear_context

    assert query.tensor_parallel.world_size == 2
    assert query.tensor_parallel.rank == 1
    assert query.tensor_parallel.mode is TensorParallelMode.COLUMN
    assert query.tensor_parallel.local_out_features == 16
    assert query.tensor_parallel.global_out_features == 32
    assert output.tensor_parallel.mode is TensorParallelMode.ROW
    assert output.tensor_parallel.local_in_features == 16
    assert output.tensor_parallel.global_in_features == 32
    assert gate.tensor_parallel.local_out_features == 32
    assert gate.tensor_parallel.global_out_features == 64
    assert head.tensor_parallel.mode is TensorParallelMode.REPLICATED
    assert head.tensor_parallel.local_out_features == 96
    assert head.tensor_parallel.global_out_features == 96


def test_legacy_three_argument_factory_remains_compatible():
    calls = []

    def legacy(in_features, out_features, bias):
        calls.append((in_features, out_features, bias))
        return torch.nn.Linear(in_features, out_features, bias=bias)

    model = DenseDecoder(parse_model_config(_DENSE_RAW), linear_factory=legacy)
    assert calls
    assert isinstance(model.model.layers[0].self_attn.q_proj, torch.nn.Linear)

    # The old public factory's returned callable also accepted three arguments
    # directly, outside model construction.
    direct = linear_factory(QuantConfig(QuantMethod.FP8))(32, 16, False)
    assert isinstance(direct, Fp8Linear)
    assert not hasattr(direct, "linear_context")
