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
from kairyu.models.moe_parallel import EpMoeBlock, build_ep_model
from kairyu.quant.linear import linear_factory

_RAW = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
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


class _UnusedEpComm:
    rank = 1
    world_size = 2


def _write_checkpoint(
    path,
    *,
    omit: str | None = None,
    bad_dtype: str | None = None,
    bad_kv_scale: float | None = None,
    unexpected: bool = False,
) -> dict[str, torch.Tensor]:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_RAW))
    excludes = ["model.layers.0.mlp.gate", "lm_head"]
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
        if ".experts." in name and name.endswith(".weight"):
            expert_index = int(name.split(".experts.", 1)[1].split(".", 1)[0])
            tensor.fill_(expert_index + 1)
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
    assert not any(".experts.0." in name for name in loaded_names)
    assert not any(".experts.1." in name for name in loaded_names)
    owned_weight = model.get_buffer(
        "model.layers.0.mlp.experts.2.gate_proj.weight"
    )
    assert torch.equal(owned_weight, torch.full_like(owned_weight, 3))
    assert not any(
        tensor.device.type == "meta"
        for tensor in (*model.parameters(), *model.buffers())
    )


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
