"""Real-checkpoint expert-sharded loading contracts for G4 M-A1."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from kairyu.engine.core.quant_config import QuantConfig, QuantMethod
from kairyu.engine.core.weights import CheckpointReader
from kairyu.models.config import parse_model_config
from kairyu.models.llama import DenseDecoder
from kairyu.models.loader import load_model
from kairyu.models.moe_parallel import (
    EpMoeBlock,
    _ExpertShardedLinearFactory,
    _validate_attention_dp_build_envelope,
    build_ep_model,
)
from kairyu.quant.linear import (
    LinearRole,
    NvFp4Linear,
    TensorParallelMode,
    linear_factory,
)

_RAW = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 64,
    "vocab_size": 64,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
}

_ATTENTION_DP_RAW = {
    **_RAW,
    "hidden_size": 4096,
    "num_hidden_layers": 94,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 12_288,
    "vocab_size": 151_936,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": False,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 1536,
}


class _UnusedEpComm:
    def __init__(self, rank: int = 1, world_size: int = 2) -> None:
        self.rank = rank
        self.world_size = world_size


def test_attention_dp_build_envelope_pins_qwen3_235b_nvfp4_ep4() -> None:
    config = parse_model_config(_ATTENTION_DP_RAW)
    quant = QuantConfig(
        QuantMethod.NVFP4,
        weight_bits=4,
        group_size=16,
    )
    comm = _UnusedEpComm(rank=3, world_size=4)

    _validate_attention_dp_build_envelope(
        config,
        quant,
        ep_size=4,
        ep_rank=3,
        comm=comm,
        dtype=torch.bfloat16,
        device="cuda:3",
    )

    with pytest.raises(ValueError, match="ModelOpt NVFP4"):
        _validate_attention_dp_build_envelope(
            config,
            QuantConfig(QuantMethod.NONE),
            ep_size=4,
            ep_rank=3,
            comm=comm,
            dtype=torch.bfloat16,
            device="cuda:3",
        )
    with pytest.raises(ValueError, match="runtime dtype bfloat16"):
        _validate_attention_dp_build_envelope(
            config,
            quant,
            ep_size=4,
            ep_rank=3,
            comm=comm,
            dtype=torch.float16,
            device="cuda:3",
        )


def test_attention_dp_factory_keeps_full_attention_output_replicated() -> None:
    quant = QuantConfig(
        QuantMethod.NVFP4,
        weight_bits=4,
        group_size=16,
    )
    factory = _ExpertShardedLinearFactory(
        linear_factory(quant),
        num_experts=4,
        ep_rank=3,
        ep_size=4,
        replicate_attention_output=True,
    )

    output = factory(
        8192,
        4096,
        False,
        qualified_name="model.layers.0.self_attn.o_proj",
        role=LinearRole.ATTENTION_OUTPUT,
        shard_dim=1,
    )

    assert isinstance(output, NvFp4Linear)
    # Qwen3-235B uses 64 query heads x 128 dimensions while hidden_size is
    # 4096.  Attention-DP replicates this complete 8192 -> 4096 projection.
    assert output.in_features == 8192
    assert output.out_features == 4096
    placement = output.linear_context.tensor_parallel
    assert placement.mode is TensorParallelMode.REPLICATED
    assert placement.rank == 0
    assert placement.world_size == 1
    assert placement.shard_dim is None


def _write_checkpoint(
    path,
    *,
    omit: str | None = None,
    bad_dtype: str | None = None,
    bad_kv_scale: float | None = None,
    expert_input_scales: dict[tuple[int, str], float] | None = None,
    ignored_modules: tuple[str, ...] = (),
    unexpected: bool = False,
) -> dict[str, torch.Tensor]:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_RAW))
    excludes = ["model.layers.0.mlp.gate", "lm_head", *ignored_modules]
    (path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt", "version": "0.33.0"},
                "quantization": {
                    "quant_algo": "NVFP4",
                    "kv_cache_quant_algo": "FP8",
                    "group_size": 16,
                    "exclude_modules": excludes,
                },
            }
        )
    )
    quant = QuantConfig(
        QuantMethod.NVFP4,
        weight_bits=4,
        group_size=16,
        ignored_layers=tuple(excludes),
    )
    model = DenseDecoder(
        parse_model_config(_RAW),
        linear_factory=linear_factory(quant),
    )
    state = {
        name: tensor.detach().clone().contiguous()
        for name, tensor in model.state_dict().items()
    }
    for name, tensor in state.items():
        if name == "model.layers.0.self_attn.o_proj.weight":
            values = torch.arange(tensor.numel(), dtype=torch.int64).reshape(tensor.shape)
            tensor.copy_((values % 251).to(tensor.dtype))
        if name == "model.layers.0.self_attn.o_proj.weight_scale":
            values = torch.arange(1, tensor.numel() + 1).reshape(tensor.shape)
            tensor.copy_(values.to(tensor.dtype))
        if name == "model.layers.0.self_attn.o_proj.weight_scale_2":
            tensor.fill_(0.03125)
        if name == "model.layers.0.self_attn.o_proj.input_scale":
            tensor.fill_(0.125)
        if ".experts." in name and name.endswith(".weight"):
            expert_index = int(name.split(".experts.", 1)[1].split(".", 1)[0])
            tensor.fill_(expert_index + 1)
        if ".experts." in name and name.endswith(".input_scale"):
            expert_index = int(name.split(".experts.", 1)[1].split(".", 1)[0])
            projection = name.rsplit(".", 2)[-2]
            if expert_input_scales is not None:
                tensor.fill_(expert_input_scales[(expert_index, projection)])
    state["model.layers.0.self_attn.k_proj.k_scale"] = torch.tensor(
        0.5 if bad_kv_scale is None else bad_kv_scale
    )
    state["model.layers.0.self_attn.v_proj.v_scale"] = torch.tensor(0.25)
    if omit is not None:
        state.pop(omit)
    if bad_dtype is not None:
        state[bad_dtype] = state[bad_dtype].to(torch.int8)
    if unexpected:
        state["model.layers.0.mlp.experts.4.gate_proj.weight"] = torch.zeros(
            16, 16, dtype=torch.uint8
        )
    save_file(state, path / "model.safetensors")
    return state


def test_build_ep_model_attention_dp_rejects_non_formal_geometry(tmp_path) -> None:
    checkpoint = tmp_path / "tiny-attention-dp"
    _write_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="pinned Qwen3-235B geometry"):
        build_ep_model(
            checkpoint,
            ep_size=4,
            ep_rank=3,
            comm=_UnusedEpComm(rank=3, world_size=4),
            dtype=torch.bfloat16,
            device="cuda:3",
            attention_dp=True,
        )


def test_build_ep_model_loads_only_owned_global_experts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "model"
    full_state = _write_checkpoint(checkpoint)
    selected_calls: list[tuple[str, ...]] = []
    original = CheckpointReader.selected_items

    def recording_selected(self, names):
        selected = tuple(names)
        selected_calls.append(selected)
        yield from original(self, selected)

    monkeypatch.setattr(CheckpointReader, "selected_items", recording_selected)
    model, config, info = build_ep_model(
        checkpoint,
        ep_size=2,
        ep_rank=1,
        comm=_UnusedEpComm(),
    )

    assert config.architecture == "Qwen3MoeForCausalLM"
    assert info.owned_expert_indices == (2, 3)
    assert info.quantization_method == "nvfp4"
    assert info.quantization_source == "hf_quant_config.json"
    assert info.kv_cache_quant_algo == "FP8"
    assert info.auxiliary_kv_scale_count == 2
    assert info.checkpoint_tensor_count == len(full_state)
    block = model.model.layers[0].mlp
    assert isinstance(block, EpMoeBlock)
    assert block.experts[0] is None and block.experts[1] is None
    assert block.experts[2] is not None and block.experts[3] is not None
    rank_names = set(model.state_dict())
    assert any(".experts.2." in name for name in rank_names)
    assert any(".experts.3." in name for name in rank_names)
    assert not any(".experts.0." in name for name in rank_names)
    assert not any(".experts.1." in name for name in rank_names)
    loaded_names = set().union(*(set(call) for call in selected_calls))
    remote_names = {
        name
        for name in loaded_names
        if ".experts.0." in name or ".experts.1." in name
    }
    assert remote_names
    assert all(name.endswith(".input_scale") for name in remote_names)
    owned_weight = model.get_buffer(
        "model.layers.0.mlp.experts.2.gate_proj.weight"
    )
    assert torch.equal(owned_weight, torch.full_like(owned_weight, 3))
    output_name = "model.layers.0.self_attn.o_proj"
    output = model.get_submodule(output_name)
    assert isinstance(output, NvFp4Linear)
    assert output.in_features == 32
    assert output.linear_context.tensor_parallel.rank == 1
    assert output.linear_context.tensor_parallel.world_size == 2
    assert torch.equal(output.weight, full_state[f"{output_name}.weight"][:, 16:])
    assert torch.equal(
        output.weight_scale,
        full_state[f"{output_name}.weight_scale"][:, 2:],
    )
    assert torch.equal(
        output.weight_scale_2,
        full_state[f"{output_name}.weight_scale_2"],
    )
    assert torch.equal(output.input_scale, full_state[f"{output_name}.input_scale"])
    assert f"{output_name}.weight" in loaded_names
    assert f"{output_name}.weight_scale" in loaded_names
    assert not any(
        tensor.device.type == "meta"
        for tensor in (*model.parameters(), *model.buffers())
    )


def test_build_ep_model_propagates_float32_logits_mode(tmp_path) -> None:
    checkpoint = tmp_path / "ep-logits-dtype"
    _write_checkpoint(checkpoint)

    model, _config, _info = build_ep_model(
        checkpoint,
        ep_size=2,
        ep_rank=1,
        comm=_UnusedEpComm(),
        logits_dtype="float32",
    )

    assert model.logits_dtype == "float32"
    logits = model.logits(torch.randn(3, _RAW["hidden_size"]))
    assert logits.dtype is torch.float32


def test_ep4_attention_output_loads_only_rank_contiguous_nvfp4_k_shard(tmp_path):
    checkpoint = tmp_path / "ep4-attention-output"
    full_state = _write_checkpoint(checkpoint)

    model, _config, info = build_ep_model(
        checkpoint,
        ep_size=4,
        ep_rank=3,
        comm=_UnusedEpComm(rank=3, world_size=4),
    )

    assert info.owned_expert_indices == (3,)
    output_name = "model.layers.0.self_attn.o_proj"
    output = model.get_submodule(output_name)
    assert isinstance(output, NvFp4Linear)
    assert output.in_features == 16
    assert output.linear_context.tensor_parallel.rank == 3
    assert output.linear_context.tensor_parallel.world_size == 4
    assert torch.equal(output.weight, full_state[f"{output_name}.weight"][:, 24:32])
    assert torch.equal(
        output.weight_scale,
        full_state[f"{output_name}.weight_scale"][:, 3:4],
    )
    assert torch.equal(
        output.weight_scale_2,
        full_state[f"{output_name}.weight_scale_2"],
    )
    assert torch.equal(output.input_scale, full_state[f"{output_name}.input_scale"])


def test_ep_nvfp4_uses_global_scales_from_owned_and_remote_experts(tmp_path):
    checkpoint = tmp_path / "global-scales"
    scales = {
        (expert, projection): value
        for expert, values in enumerate(
            (
                (0.5, 0.5, 9.0),
                (3.0, 3.0, 2.0),
                (1.5, 1.5, 8.0),
                (2.0, 2.0, 6.0),
            )
        )
        for projection, value in zip(
            ("gate_proj", "up_proj", "down_proj"), values, strict=True
        )
    }
    _write_checkpoint(checkpoint, expert_input_scales=scales)

    model, _config, _info = build_ep_model(
        checkpoint,
        ep_size=2,
        ep_rank=1,
        comm=_UnusedEpComm(),
    )

    block = model.model.layers[0].mlp
    for expert_index in block.owned_expert_indices:
        expert = block.local_expert(expert_index)
        assert expert.gate_proj.input_scale.item() == 3.0
        assert expert.up_proj.input_scale.item() == 3.0
        assert expert.down_proj.input_scale.item() == 9.0

    full_model, _config, _generation = load_model(checkpoint)
    for expert in full_model.model.layers[0].mlp.experts:
        assert expert.gate_proj.input_scale.item() == 3.0
        assert expert.up_proj.input_scale.item() == 3.0
        assert expert.down_proj.input_scale.item() == 9.0


def test_ep_nvfp4_rejects_invalid_remote_input_scale(tmp_path):
    checkpoint = tmp_path / "bad-input-scale"
    scales = {
        (expert, projection): 1.0
        for expert in range(4)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    scales[(0, "down_proj")] = 0.0
    _write_checkpoint(checkpoint, expert_input_scales=scales)

    with pytest.raises(ValueError, match="finite positive FP32"):
        build_ep_model(
            checkpoint,
            ep_size=2,
            ep_rank=1,
            comm=_UnusedEpComm(),
        )


def test_ep_nvfp4_preserves_homogeneous_dense_ignored_projection(tmp_path):
    checkpoint = tmp_path / "ignored-gates"
    ignored = tuple(
        f"model.layers.0.mlp.experts.{expert}.gate_proj"
        for expert in range(_RAW["num_experts"])
    )
    _write_checkpoint(checkpoint, ignored_modules=ignored)

    model, _config, _info = build_ep_model(
        checkpoint,
        ep_size=2,
        ep_rank=1,
        comm=_UnusedEpComm(),
    )

    block = model.model.layers[0].mlp
    for expert_index in block.owned_expert_indices:
        expert = block.local_expert(expert_index)
        assert isinstance(expert.gate_proj, torch.nn.Linear)
        assert not getattr(expert.gate_proj, "is_quantized", False)
        assert getattr(expert.up_proj, "quant_scheme", None) == "nvfp4"
        assert getattr(expert.down_proj, "quant_scheme", None) == "nvfp4"


def test_ep_contract_requires_remote_expert_tensors(tmp_path):
    checkpoint = tmp_path / "missing"
    remote = "model.layers.0.mlp.experts.0.gate_proj.weight"
    _write_checkpoint(checkpoint, omit=remote)

    with pytest.raises(ValueError, match="layout mismatch"):
        build_ep_model(
            checkpoint,
            ep_size=2,
            ep_rank=1,
            comm=_UnusedEpComm(),
        )


def test_ep_contract_rejects_unknown_expert_layout(tmp_path):
    checkpoint = tmp_path / "unknown"
    _write_checkpoint(checkpoint, unexpected=True)

    with pytest.raises(ValueError, match="layout mismatch"):
        build_ep_model(
            checkpoint,
            ep_size=2,
            ep_rank=1,
            comm=_UnusedEpComm(),
        )


def test_ep_nvfp4_abi_dtype_mismatch_fails_closed(tmp_path):
    checkpoint = tmp_path / "dtype"
    owned = "model.layers.0.mlp.experts.2.gate_proj.weight"
    _write_checkpoint(checkpoint, bad_dtype=owned)

    with pytest.raises(ValueError, match="required ABI dtype"):
        build_ep_model(
            checkpoint,
            ep_size=2,
            ep_rank=1,
            comm=_UnusedEpComm(),
        )


def test_ep_nvfp4_global_abi_rejects_remote_expert_dtype(tmp_path):
    checkpoint = tmp_path / "remote-dtype"
    remote = "model.layers.0.mlp.experts.0.gate_proj.weight_scale_2"
    _write_checkpoint(checkpoint, bad_dtype=remote)

    with pytest.raises(ValueError, match="required ABI dtype"):
        build_ep_model(
            checkpoint,
            ep_size=2,
            ep_rank=1,
            comm=_UnusedEpComm(),
        )


@pytest.mark.parametrize("bad_scale", [float("nan"), float("inf"), 0.0, -1.0])
def test_ep_contract_rejects_invalid_auxiliary_kv_scale(tmp_path, bad_scale):
    checkpoint = tmp_path / "bad-kv-scale"
    _write_checkpoint(checkpoint, bad_kv_scale=bad_scale)

    with pytest.raises(ValueError, match="finite positive"):
        build_ep_model(
            checkpoint,
            ep_size=2,
            ep_rank=1,
            comm=_UnusedEpComm(),
        )
