from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kairyu.engine.core.nvfp4_accuracy import (
    NvFp4AccuracyProfile,
    make_nvfp4_accuracy_policy,
    saturation_snapshot,
)
from kairyu.engine.core.quant_config import QuantConfig, QuantMethod
from kairyu.quant import nvfp4
from kairyu.quant.linear import (
    ExpertScope,
    LinearRole,
    NvFp4Linear,
    NvFp4ToFp8Linear,
    linear_factory,
)


def _factory(profile: NvFp4AccuracyProfile):
    quant = QuantConfig(QuantMethod.NVFP4, weight_bits=4, group_size=16)
    return linear_factory(
        quant,
        selection_policy=make_nvfp4_accuracy_policy(profile, num_layers=4),
    )


def _projection(factory, name: str, role: LinearRole, layer: int, **kwargs):
    return factory(
        16,
        16,
        False,
        qualified_name=name,
        role=role,
        layer_index=layer,
        **kwargs,
    )


def test_profile_parses_and_rejects_unknown_or_malformed_selectors() -> None:
    profile = NvFp4AccuracyProfile.parse(
        {
            "fp8": ["first:1", "down_proj"],
            "dynamic_activation": ["last:2", "shared_experts"],
            "saturation_counters": True,
        }
    )
    assert profile.as_dict() == {
        "fp8": ["first:1", "down_proj"],
        "dynamic_activation": ["last:2", "shared_experts"],
        "saturation_counters": True,
    }
    with pytest.raises(ValueError, match="unknown NVFP4 accuracy selector"):
        NvFp4AccuracyProfile.parse({"fp8": ["middle:2"]})
    with pytest.raises(ValueError, match="positive count"):
        NvFp4AccuracyProfile.parse({"fp8": ["first:0"]})
    with pytest.raises(ValueError, match="unsupported fields"):
        NvFp4AccuracyProfile.parse({"fp16": []})
    with pytest.raises(ValueError, match="exceeds 4 layers"):
        make_nvfp4_accuracy_policy(
            NvFp4AccuracyProfile(fp8=("last:5",)),
            num_layers=4,
        )


def test_native_backend_profile_validation_requires_a_real_model() -> None:
    from kairyu.engine.config_validation import validate_backend_options

    profile = {"dynamic_activation": ["last:1"]}
    with pytest.raises(ValueError, match="requires a real model_path"):
        validate_backend_options("kairyu", {"nvfp4_accuracy_profile": profile})
    validate_backend_options(
        "kairyu-proc",
        {
            "model_path": "/models/nvfp4",
            "nvfp4_accuracy_profile": profile,
        },
    )


def test_policy_selects_fp8_dynamic_and_shared_projection_roles() -> None:
    factory = _factory(
        NvFp4AccuracyProfile(
            fp8=("first:1", "down_proj"),
            dynamic_activation=("last:1", "shared_experts"),
            saturation_counters=True,
        )
    )
    first = _projection(factory, "model.layers.0.self_attn.q_proj", LinearRole.ATTENTION_QUERY, 0)
    down = _projection(factory, "model.layers.1.mlp.down_proj", LinearRole.MLP_DOWN, 1)
    last = _projection(factory, "model.layers.3.self_attn.q_proj", LinearRole.ATTENTION_QUERY, 3)
    shared = _projection(
        factory,
        "model.layers.1.mlp.shared_experts.up_proj",
        LinearRole.MLP_UP,
        1,
        expert_scope=ExpertScope.SHARED,
    )
    ordinary = _projection(
        factory,
        "model.layers.1.self_attn.q_proj",
        LinearRole.ATTENTION_QUERY,
        1,
    )

    assert isinstance(first, NvFp4ToFp8Linear)
    assert isinstance(down, NvFp4ToFp8Linear)
    assert isinstance(last, NvFp4Linear) and last.activation_dynamic
    assert isinstance(shared, NvFp4Linear) and shared.activation_dynamic
    assert type(ordinary) is NvFp4Linear and not ordinary.activation_dynamic
    assert all(
        module.observe_activation_saturation
        for module in (first, down, last, shared, ordinary)
    )


def test_nvfp4_to_fp8_loads_source_abi_then_discards_packed_weight() -> None:
    module = NvFp4ToFp8Linear(16, 8, False)
    source = torch.linspace(-2, 2, 128, dtype=torch.float32).reshape(8, 16)
    packed, scale, scale_2 = nvfp4.quantize_nvfp4(source)
    module.load_state_dict(
        {
            "weight": packed,
            "weight_scale": scale,
            "weight_scale_2": scale_2,
            "input_scale": torch.ones(()),
        },
        assign=True,
    )
    expected = module.dequantize().clone()
    actual = module(torch.randn(2, 16))
    assert actual.shape == (2, 8)
    assert module.weight.dtype is torch.float8_e4m3fn
    assert module.weight.shape == (8, 16)
    assert "weight_scale_2" not in module._buffers
    torch.testing.assert_close(module.dequantize(), expected, rtol=0.04, atol=0.05)


def test_static_saturation_is_counted_and_dynamic_scaling_avoids_it() -> None:
    static = NvFp4Linear(16, 4, False, observe_activation_saturation=True)
    dynamic = NvFp4Linear(
        16,
        4,
        False,
        activation_dynamic=True,
        observe_activation_saturation=True,
    )
    static.input_scale.fill_(1.0 / (6.0 * 448.0))
    values = torch.zeros(2, 16)
    values[0, 0] = 2.0
    static(values)
    dynamic(values)
    assert saturation_snapshot(static) == [
        {
            "projection": "",
            "runtime_format": "nvfp4",
            "activation_scale": "static",
            "blocks": 2,
            "saturated_blocks": 1,
            "saturation_rate": 0.5,
        }
    ]
    dynamic_row = saturation_snapshot(dynamic, reset=True)[0]
    assert dynamic_row["blocks"] == 2
    assert dynamic_row["saturated_blocks"] == 0
    assert dynamic.activation_blocks.item() == 0


def test_ep_accuracy_probe_reports_unique_resident_bytes_and_resets() -> None:
    from kairyu.engine.core.worker import _nvfp4_accuracy_reply

    model = torch.nn.Sequential(
        NvFp4Linear(16, 4, False, observe_activation_saturation=True)
    )
    model(torch.ones(2, 16))
    reply = _nvfp4_accuracy_reply(
        SimpleNamespace(rank=1),
        SimpleNamespace(_model=model),
        reset=True,
    )
    assert reply.error is None
    assert reply.row["rank"] == 1
    assert reply.row["resident_model_bytes"] > 0
    assert reply.row["projection_formats"] == {"nvfp4": 1}
    assert reply.row["saturation"][0]["blocks"] == 2
    assert model[0].activation_blocks.item() == 0
