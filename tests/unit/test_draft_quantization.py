"""#234: draft-format metadata, policy, packed load, and numerical gates."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from kairyu.engine.core.quant_config import QuantConfig, QuantMethod
from kairyu.models.config import parse_model_config
from kairyu.models.draft_quant import (
    MTP_DRAFT_QUANTIZATION_KEY,
    DraftQuantization,
    draft_linear_factory,
    load_draft_quantization,
    pack_dynamic_fp8_draft_state,
    parse_draft_quantization,
)
from kairyu.models.eagle import EagleConfig, EagleDraftHead, load_eagle_head
from kairyu.models.layers import RotaryEmbedding
from kairyu.models.mtp import MtpDraftHead, load_mtp_head
from kairyu.quant.linear import (
    Fp8Linear,
    LinearKernel,
    LinearKernelCapabilities,
    ModelScope,
)

_FP8_CONFIG = {
    "quant_method": "compressed-tensors",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "type": "float",
                "num_bits": 8,
                "strategy": "channel",
                "symmetric": True,
            },
            "input_activations": {
                "type": "float",
                "num_bits": 8,
                "dynamic": True,
                "strategy": "token",
                "symmetric": True,
            },
        }
    },
}

_EAGLE = EagleConfig(
    hidden_size=32,
    num_attention_heads=4,
    intermediate_size=64,
    draft_vocab_size=48,
    num_key_value_heads=2,
    head_dim=8,
)

_DSV3_RAW = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 128,
    "max_position_embeddings": 256,
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
    "rms_norm_eps": 1e-6,
}


def _mtp_fp8_config(*extra_ignore: str) -> dict:
    layer = _DSV3_RAW["num_hidden_layers"]
    return {
        **_FP8_CONFIG,
        "ignore": [
            f"model.layers.{layer}.mlp.gate",
            f"model.layers.{layer}.self_attn.kv_b_proj",
            *extra_ignore,
        ],
    }


def _raw(
    scope: ModelScope,
    *,
    quantization: dict | None = None,
) -> dict:
    key = "quantization_config" if scope is ModelScope.EAGLE_DRAFT else MTP_DRAFT_QUANTIZATION_KEY
    return {key: quantization or _FP8_CONFIG}


def _metadata(scope: ModelScope) -> DraftQuantization:
    if scope is ModelScope.MTP_DRAFT:
        raw = {
            **_DSV3_RAW,
            **_raw(scope, quantization=_mtp_fp8_config()),
        }
    else:
        raw = _raw(scope)
    return parse_draft_quantization(
        raw,
        expected_scope=scope,
        target_quantization=QuantConfig(QuantMethod.NONE),
    )


def _eagle_checkpoint_raw(
    *, quantization: dict | None = None, dense: bool = False, **overrides
) -> dict:
    raw = {
        "hidden_size": _EAGLE.hidden_size,
        "intermediate_size": _EAGLE.intermediate_size,
        "draft_vocab_size": _EAGLE.draft_vocab_size,
        "num_attention_heads": _EAGLE.num_attention_heads,
        "num_key_value_heads": _EAGLE.num_key_value_heads,
        "head_dim": _EAGLE.head_dim,
        "rope_theta": _EAGLE.rope_theta,
        "rms_norm_eps": _EAGLE.rms_norm_eps,
    }
    if not dense:
        raw["quantization_config"] = quantization or _FP8_CONFIG
    raw.update(overrides)
    return raw


def _relative_rmse(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    reference = reference.to(torch.float32)
    candidate = candidate.to(torch.float32)
    return (candidate - reference).square().mean().sqrt() / reference.square().mean().sqrt()


def _mtp_checkpoint_state(config, local: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{config.num_hidden_layers}."
    checkpoint: dict[str, torch.Tensor] = {}
    for name, tensor in local.items():
        if name.startswith("head."):
            target = prefix + "shared_head.head." + name[len("head.") :]
        elif name.startswith("decoder."):
            target = prefix + name[len("decoder.") :]
        else:
            target = prefix + name
        checkpoint[target] = tensor
    return checkpoint


def test_missing_metadata_is_dense_and_never_inherits_target_quantization():
    parsed = parse_draft_quantization(
        {},
        expected_scope=ModelScope.EAGLE_DRAFT,
        target_quantization=QuantConfig(QuantMethod.FP8),
    )
    assert parsed.quantization.method is QuantMethod.NONE
    assert parsed.source == "dense-default"


def test_external_eagle_uses_its_own_standard_quantization_config(tmp_path):
    raw = _raw(ModelScope.EAGLE_DRAFT)
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    parsed = load_draft_quantization(
        tmp_path,
        expected_scope=ModelScope.EAGLE_DRAFT,
        target_quantization=QuantConfig(QuantMethod.NVFP4),
    )
    assert parsed.quantization.method is QuantMethod.FP8
    assert parsed.quantization.activation_dynamic is True
    assert parsed.source == "quantization_config"


def test_mtp_inherits_target_or_uses_explicit_dense_override():
    inherited = parse_draft_quantization(
        {**_DSV3_RAW, "quantization_config": _mtp_fp8_config()},
        expected_scope=ModelScope.MTP_DRAFT,
    )
    assert inherited.quantization.method is QuantMethod.FP8
    assert inherited.source == "target-quantization-inherited"

    overridden = parse_draft_quantization(
        {MTP_DRAFT_QUANTIZATION_KEY: {}},
        expected_scope=ModelScope.MTP_DRAFT,
        target_quantization=QuantConfig(QuantMethod.FP8),
    )
    assert overridden.quantization.method is QuantMethod.NONE
    assert overridden.source == MTP_DRAFT_QUANTIZATION_KEY


def test_quantized_mtp_requires_canonical_dense_projection_ignores():
    raw = {**_DSV3_RAW, MTP_DRAFT_QUANTIZATION_KEY: _FP8_CONFIG}
    with pytest.raises(ValueError, match="must ignore dense architectural projections"):
        parse_draft_quantization(raw, expected_scope=ModelScope.MTP_DRAFT)


def test_quantized_mtp_accepts_regex_coverage_for_dense_projection_ignores():
    layer = _DSV3_RAW["num_hidden_layers"]
    quantization = {
        **_FP8_CONFIG,
        "ignore": [
            rf"re:model\.layers\.{layer}\.mlp\.gate",
            rf"re:model\.layers\.{layer}\.self_attn\.kv_b_proj",
        ],
    }
    parsed = parse_draft_quantization(
        {**_DSV3_RAW, MTP_DRAFT_QUANTIZATION_KEY: quantization},
        expected_scope=ModelScope.MTP_DRAFT,
    )
    assert parsed.quantization.method is QuantMethod.FP8


@pytest.mark.parametrize(
    ("raw", "scope", "match"),
    [
        (
            {},
            ModelScope.TARGET,
            "not a draft model scope",
        ),
        (
            {MTP_DRAFT_QUANTIZATION_KEY: "fp8"},
            ModelScope.MTP_DRAFT,
            "must be a JSON object or null",
        ),
        (
            _raw(
                ModelScope.EAGLE_DRAFT,
                quantization={
                    "quant_method": "modelopt",
                    "quant_algo": "FP8",
                    "activation_scheme": "dynamic",
                },
            ),
            ModelScope.EAGLE_DRAFT,
            "only compressed-tensors",
        ),
        (
            _raw(
                ModelScope.EAGLE_DRAFT,
                quantization={
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                },
            ),
            ModelScope.EAGLE_DRAFT,
            "only compressed-tensors",
        ),
    ],
)
def test_invalid_or_unvalidated_metadata_fails_closed(raw, scope, match):
    with pytest.raises(ValueError, match=match):
        parse_draft_quantization(
            raw,
            expected_scope=scope,
        )


@pytest.mark.parametrize(
    ("quantization", "match"),
    [
        (
            {
                **_FP8_CONFIG,
                "config_groups": {
                    "group_0": _FP8_CONFIG["config_groups"]["group_0"],
                    "group_1": _FP8_CONFIG["config_groups"]["group_0"],
                },
            },
            "exactly one config group",
        ),
        (
            {
                **_FP8_CONFIG,
                "config_groups": {
                    "group_0": {
                        **_FP8_CONFIG["config_groups"]["group_0"],
                        "targets": ["Conv2d"],
                    }
                },
            },
            "targets",
        ),
        (
            {
                **_FP8_CONFIG,
                "config_groups": {
                    "group_0": {
                        **_FP8_CONFIG["config_groups"]["group_0"],
                        "weights": {
                            "type": "float",
                            "num_bits": 8,
                            "strategy": "tensor",
                        },
                    }
                },
            },
            "channel strategy",
        ),
        (
            {
                **_FP8_CONFIG,
                "config_groups": {
                    "group_0": {
                        **_FP8_CONFIG["config_groups"]["group_0"],
                        "weights": {"type": "float", "num_bits": 8},
                    }
                },
            },
            "channel strategy",
        ),
        (
            {
                **_FP8_CONFIG,
                "config_groups": {
                    "group_0": {
                        **_FP8_CONFIG["config_groups"]["group_0"],
                        "input_activations": {
                            "type": "float",
                            "num_bits": 8,
                            "dynamic": True,
                            "strategy": "channel",
                        },
                    }
                },
            },
            "dynamic FP8",
        ),
        (
            {
                **_FP8_CONFIG,
                "config_groups": {
                    "group_0": {
                        **_FP8_CONFIG["config_groups"]["group_0"],
                        "input_activations": {
                            "type": "float",
                            "num_bits": 8,
                            "dynamic": True,
                        },
                    }
                },
            },
            "token strategy",
        ),
        (
            {
                **_FP8_CONFIG,
                "config_groups": {
                    "group_0": {
                        **_FP8_CONFIG["config_groups"]["group_0"],
                        "weights": {
                            **_FP8_CONFIG["config_groups"]["group_0"]["weights"],
                            "symmetric": False,
                        },
                    }
                },
            },
            "symmetric=true",
        ),
        (
            {
                **_FP8_CONFIG,
                "config_groups": {
                    "group_0": {
                        **_FP8_CONFIG["config_groups"]["group_0"],
                        "weights": {
                            **_FP8_CONFIG["config_groups"]["group_0"]["weights"],
                            "dynamic": True,
                        },
                    }
                },
            },
            "static weights",
        ),
        (
            {
                **_FP8_CONFIG,
                "config_groups": {
                    "group_0": {
                        **_FP8_CONFIG["config_groups"]["group_0"],
                        "input_activations": {
                            **_FP8_CONFIG["config_groups"]["group_0"]["input_activations"],
                            "symmetric": False,
                        },
                    }
                },
            },
            "symmetric=true",
        ),
        (
            {
                **_FP8_CONFIG,
                "config_groups": {
                    "group_0": {
                        **_FP8_CONFIG["config_groups"]["group_0"],
                        "output_activations": {
                            "type": "float",
                            "num_bits": 8,
                            "dynamic": True,
                            "strategy": "token",
                            "symmetric": True,
                        },
                    }
                },
            },
            "output activations",
        ),
    ],
)
def test_compressed_tensors_draft_dialect_fails_closed(quantization, match):
    with pytest.raises(ValueError, match=match):
        parse_draft_quantization(
            _raw(ModelScope.EAGLE_DRAFT, quantization=quantization),
            expected_scope=ModelScope.EAGLE_DRAFT,
        )


def test_compressed_tensors_draft_dialect_rejects_implicit_linear_target():
    group = dict(_FP8_CONFIG["config_groups"]["group_0"])
    group.pop("targets")
    with pytest.raises(ValueError, match="targets"):
        parse_draft_quantization(
            _raw(
                ModelScope.EAGLE_DRAFT,
                quantization={
                    **_FP8_CONFIG,
                    "config_groups": {"group_0": group},
                },
            ),
            expected_scope=ModelScope.EAGLE_DRAFT,
        )


def test_eagle_fp8_policy_packs_fusion_attention_mlp_and_head():
    metadata = _metadata(ModelScope.EAGLE_DRAFT)
    dense_hidden_rows: list[torch.Tensor] = []
    quant_hidden_rows: list[torch.Tensor] = []
    dense_logits_rows: list[torch.Tensor] = []
    quant_logits_rows: list[torch.Tensor] = []

    # Quantization quality is an aggregate contract, not a single element-wise
    # allclose draw.  Cover several independently initialized heads and contexts,
    # then require both bounded signal error and stable greedy draft decisions.
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        for model_seed in range(4):
            torch.manual_seed(model_seed)
            dense = EagleDraftHead(_EAGLE).eval()
            quantized = EagleDraftHead(
                _EAGLE,
                linear_factory=draft_linear_factory(metadata),
            ).eval()

            assert isinstance(quantized.fc, Fp8Linear)
            assert isinstance(quantized.midlayer.self_attn["q_proj"], Fp8Linear)
            assert isinstance(quantized.midlayer.self_attn["k_proj"], Fp8Linear)
            assert isinstance(quantized.midlayer.mlp["down_proj"], Fp8Linear)
            assert isinstance(quantized.lm_head, Fp8Linear)

            packed = pack_dynamic_fp8_draft_state(dense.state_dict(), quantized)
            quantized.load_state_dict(packed)
            assert packed["fc.weight"].dtype is torch.float8_e4m3fn
            assert packed["fc.weight_scale"].shape == (32, 1)

            for input_seed in range(8):
                generator = torch.Generator().manual_seed(1_000 + input_seed)
                aux = torch.randn(5, 96, generator=generator)
                embeds = torch.randn(5, 32, generator=generator)
                dense_hidden = dense.midlayer(embeds, dense.fuse(aux))
                quant_hidden = quantized.midlayer(embeds, quantized.fuse(aux))
                dense_hidden_rows.append(dense_hidden)
                quant_hidden_rows.append(quant_hidden)
                dense_logits_rows.append(dense.lm_head(dense.norm(dense_hidden[-1])))
                quant_logits_rows.append(quantized.lm_head(quantized.norm(quant_hidden[-1])))

    dense_hidden = torch.cat(dense_hidden_rows)
    quant_hidden = torch.cat(quant_hidden_rows)
    dense_logits = torch.stack(dense_logits_rows)
    quant_logits = torch.stack(quant_logits_rows)
    token_agreement = (dense_logits.argmax(-1) == quant_logits.argmax(-1)).float().mean()
    assert _relative_rmse(dense_hidden, quant_hidden) < 0.05
    assert _relative_rmse(dense_logits, quant_logits) < 0.065
    assert token_agreement >= 0.80


def test_eagle_offline_pack_preserves_dense_bf16_members(tmp_path):
    metadata = _metadata(ModelScope.EAGLE_DRAFT)
    dense = EagleDraftHead(_EAGLE).to(torch.bfloat16).eval()
    quantized = EagleDraftHead(
        _EAGLE,
        linear_factory=draft_linear_factory(metadata),
    ).eval()

    packed = pack_dynamic_fp8_draft_state(dense.state_dict(), quantized)

    assert packed["fc.weight"].dtype is torch.float8_e4m3fn
    assert packed["fc.weight_scale"].dtype is torch.float32
    assert packed["norm.weight"].dtype is torch.bfloat16
    (tmp_path / "config.json").write_text(
        json.dumps(_eagle_checkpoint_raw()),
        encoding="utf-8",
    )
    save_file(packed, tmp_path / "model.safetensors")
    loaded = load_eagle_head(tmp_path, _EAGLE, dtype=torch.bfloat16)
    assert loaded.norm.weight.dtype is torch.bfloat16
    assert torch.equal(loaded.norm.weight, packed["norm.weight"])


def test_eagle_loader_respects_dense_fusion_ignore_in_mixed_fp8_checkpoint(tmp_path):
    quantization = {**_FP8_CONFIG, "ignore": ["fc"]}
    metadata = parse_draft_quantization(
        _raw(ModelScope.EAGLE_DRAFT, quantization=quantization),
        expected_scope=ModelScope.EAGLE_DRAFT,
    )
    dense = EagleDraftHead(_EAGLE).eval()
    quantized = EagleDraftHead(
        _EAGLE,
        linear_factory=draft_linear_factory(metadata),
    ).eval()
    packed = pack_dynamic_fp8_draft_state(dense.state_dict(), quantized)
    assert "fc.weight_scale" not in packed
    (tmp_path / "config.json").write_text(
        json.dumps(_eagle_checkpoint_raw(quantization=quantization)),
        encoding="utf-8",
    )
    save_file(packed, tmp_path / "model.safetensors")

    loaded = load_eagle_head(tmp_path, _EAGLE)
    assert isinstance(loaded.fc, torch.nn.Linear)
    assert isinstance(loaded.midlayer.self_attn["q_proj"], Fp8Linear)
    assert isinstance(loaded.lm_head, Fp8Linear)
    assert torch.equal(loaded.fc.weight, packed["fc.weight"])
    assert torch.equal(
        loaded.midlayer.self_attn["q_proj"].weight,
        packed["midlayer.self_attn.q_proj.weight"],
    )


def test_eagle_loader_rejects_packed_weight_dtype_drift(tmp_path):
    metadata = _metadata(ModelScope.EAGLE_DRAFT)
    dense = EagleDraftHead(_EAGLE).eval()
    quantized = EagleDraftHead(
        _EAGLE,
        linear_factory=draft_linear_factory(metadata),
    ).eval()
    packed = pack_dynamic_fp8_draft_state(dense.state_dict(), quantized)
    packed["fc.weight"] = packed["fc.weight"].to(torch.float32)
    (tmp_path / "config.json").write_text(
        json.dumps(_eagle_checkpoint_raw()),
        encoding="utf-8",
    )
    save_file(packed, tmp_path / "model.safetensors")

    with pytest.raises(TypeError, match="packed draft tensor 'fc.weight' has dtype"):
        load_eagle_head(tmp_path, _EAGLE)


def test_eagle_loader_accepts_bf16_fp8_scale_and_registers_fp32(tmp_path):
    metadata = _metadata(ModelScope.EAGLE_DRAFT)
    dense = EagleDraftHead(_EAGLE).eval()
    quantized = EagleDraftHead(
        _EAGLE,
        linear_factory=draft_linear_factory(metadata),
    ).eval()
    packed = pack_dynamic_fp8_draft_state(dense.state_dict(), quantized)
    packed["fc.weight_scale"] = packed["fc.weight_scale"].to(torch.bfloat16)
    (tmp_path / "config.json").write_text(
        json.dumps(_eagle_checkpoint_raw()),
        encoding="utf-8",
    )
    save_file(packed, tmp_path / "model.safetensors")

    loaded = load_eagle_head(tmp_path, _EAGLE)

    assert loaded.fc.weight.dtype is torch.float8_e4m3fn
    assert loaded.fc.weight_scale.dtype is torch.float32
    assert torch.equal(loaded.fc.weight_scale, packed["fc.weight_scale"].float())


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (float("nan"), "must be finite"),
        (0.0, "must be strictly positive"),
        (-1.0, "must be strictly positive"),
    ],
)
def test_eagle_loader_rejects_invalid_fp8_scale_values(tmp_path, value, match):
    metadata = _metadata(ModelScope.EAGLE_DRAFT)
    dense = EagleDraftHead(_EAGLE).eval()
    quantized = EagleDraftHead(
        _EAGLE,
        linear_factory=draft_linear_factory(metadata),
    ).eval()
    packed = pack_dynamic_fp8_draft_state(dense.state_dict(), quantized)
    packed["fc.weight_scale"] = torch.full_like(packed["fc.weight_scale"], value)
    (tmp_path / "config.json").write_text(
        json.dumps(_eagle_checkpoint_raw()),
        encoding="utf-8",
    )
    save_file(packed, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match=match):
        load_eagle_head(tmp_path, _EAGLE)


def test_eagle_loader_rejects_packed_scale_shape_drift(tmp_path):
    metadata = _metadata(ModelScope.EAGLE_DRAFT)
    dense = EagleDraftHead(_EAGLE).eval()
    quantized = EagleDraftHead(
        _EAGLE,
        linear_factory=draft_linear_factory(metadata),
    ).eval()
    packed = pack_dynamic_fp8_draft_state(dense.state_dict(), quantized)
    packed["fc.weight_scale"] = packed["fc.weight_scale"][:1].contiguous()
    (tmp_path / "config.json").write_text(
        json.dumps(_eagle_checkpoint_raw()),
        encoding="utf-8",
    )
    save_file(packed, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="tensor 'fc.weight_scale' has shape"):
        load_eagle_head(tmp_path, _EAGLE)


def test_eagle_loader_threads_logical_target_without_allocating_it(tmp_path):
    metadata = _metadata(ModelScope.EAGLE_DRAFT)
    dense = EagleDraftHead(_EAGLE).eval()
    quantized = EagleDraftHead(
        _EAGLE,
        linear_factory=draft_linear_factory(metadata),
    ).eval()
    packed = pack_dynamic_fp8_draft_state(dense.state_dict(), quantized)
    (tmp_path / "config.json").write_text(
        json.dumps(_eagle_checkpoint_raw()),
        encoding="utf-8",
    )
    save_file(packed, tmp_path / "model.safetensors")
    capabilities = LinearKernelCapabilities(
        frozenset({LinearKernel.TORCH_DENSE, LinearKernel.CUDA_FP8_AUTO}),
        "draft-loader-device-contract",
    )

    loaded = load_eagle_head(
        tmp_path,
        _EAGLE,
        dtype=torch.bfloat16,
        target_device="cuda:7",
        linear_capabilities=capabilities,
    )

    assert loaded.fc.linear_context.device == torch.device("cuda:7")
    assert loaded.fc.linear_context.dtype is torch.bfloat16
    assert loaded.fc.linear_context.capabilities is capabilities
    assert all(tensor.device.type == "cpu" for tensor in loaded.parameters())
    assert all(tensor.device.type == "cpu" for tensor in loaded.buffers())
    assert loaded.fc.weight.dtype is torch.float8_e4m3fn
    assert loaded.fc.weight_scale.dtype is torch.float32
    assert loaded.norm.weight.dtype is torch.bfloat16
    assert loaded.midlayer.inv_freq.device.type == "cpu"


@pytest.mark.parametrize(
    ("overrides", "fields"),
    [
        (
            {
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
            },
            ("num_attention_heads", "num_key_value_heads", "head_dim"),
        ),
        ({"rope_theta": _EAGLE.rope_theta * 2}, ("rope_theta",)),
        ({"rms_norm_eps": _EAGLE.rms_norm_eps * 2}, ("rms_norm_eps",)),
    ],
)
def test_eagle_loader_rejects_semantic_geometry_mismatch(tmp_path, overrides, fields):
    head = EagleDraftHead(_EAGLE).eval()
    save_file(
        {name: tensor.contiguous() for name, tensor in head.state_dict().items()},
        tmp_path / "model.safetensors",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(_eagle_checkpoint_raw(dense=True, **overrides)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint semantics") as caught:
        load_eagle_head(tmp_path, _EAGLE)
    for field in fields:
        assert field in str(caught.value)


def test_mtp_fp8_policy_keeps_router_and_absorbed_kv_dense():
    config = parse_model_config(_DSV3_RAW)
    metadata = _metadata(ModelScope.MTP_DRAFT)
    dense_hidden_rows: list[torch.Tensor] = []
    quant_hidden_rows: list[torch.Tensor] = []
    dense_logits_rows: list[torch.Tensor] = []
    quant_logits_rows: list[torch.Tensor] = []

    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        for model_seed in range(4):
            torch.manual_seed(model_seed)
            dense = MtpDraftHead(config).eval()
            quantized = MtpDraftHead(
                config,
                linear_factory=draft_linear_factory(metadata),
            ).eval()

            assert isinstance(quantized.eh_proj, Fp8Linear)
            assert isinstance(quantized.head, Fp8Linear)
            assert isinstance(quantized.decoder.self_attn.q_proj, Fp8Linear)
            assert isinstance(quantized.decoder.mlp.experts[0].up_proj, Fp8Linear)
            assert isinstance(quantized.decoder.mlp.gate, torch.nn.Linear)
            assert isinstance(quantized.decoder.self_attn.kv_b_proj, torch.nn.Linear)

            packed = pack_dynamic_fp8_draft_state(dense.state_dict(), quantized)
            quantized.load_state_dict(packed)
            rotary = RotaryEmbedding(config)
            for input_seed in range(8):
                generator = torch.Generator().manual_seed(2_000 + input_seed)
                token_ids = torch.randint(
                    0,
                    config.vocab_size,
                    (5,),
                    generator=generator,
                )
                target_hidden = torch.randn(5, config.hidden_size, generator=generator)
                dense_out = dense.forward_chain(token_ids, target_hidden, rotary)
                quant_out = quantized.forward_chain(token_ids, target_hidden, rotary)
                dense_hidden_rows.append(dense_out)
                quant_hidden_rows.append(quant_out)
                dense_logits_rows.append(dense.logits(dense_out[-1]))
                quant_logits_rows.append(quantized.logits(quant_out[-1]))

    dense_hidden = torch.cat(dense_hidden_rows)
    quant_hidden = torch.cat(quant_hidden_rows)
    dense_logits = torch.stack(dense_logits_rows)
    quant_logits = torch.stack(quant_logits_rows)
    token_agreement = (dense_logits.argmax(-1) == quant_logits.argmax(-1)).float().mean()
    assert _relative_rmse(dense_hidden, quant_hidden) < 0.08
    assert _relative_rmse(dense_logits, quant_logits) < 0.07
    assert token_agreement >= 0.85


def test_mtp_loader_respects_dense_fusion_ignore_in_mixed_fp8_checkpoint(tmp_path):
    config = parse_model_config(_DSV3_RAW)
    fusion_name = f"model.layers.{config.num_hidden_layers}.eh_proj"
    quantization = _mtp_fp8_config(fusion_name)
    raw = {
        **_DSV3_RAW,
        **_raw(ModelScope.MTP_DRAFT, quantization=quantization),
    }
    metadata = parse_draft_quantization(
        raw,
        expected_scope=ModelScope.MTP_DRAFT,
    )
    dense = MtpDraftHead(config).eval()
    quantized = MtpDraftHead(
        config,
        linear_factory=draft_linear_factory(metadata),
    ).eval()
    local = pack_dynamic_fp8_draft_state(dense.state_dict(), quantized)
    assert "eh_proj.weight_scale" not in local
    checkpoint = _mtp_checkpoint_state(config, local)
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    save_file(checkpoint, tmp_path / "model.safetensors")

    loaded = load_mtp_head(tmp_path, config)
    assert isinstance(loaded.eh_proj, torch.nn.Linear)
    assert isinstance(loaded.head, Fp8Linear)
    assert isinstance(loaded.decoder.self_attn.q_proj, Fp8Linear)
    assert isinstance(loaded.decoder.mlp.gate, torch.nn.Linear)
    assert torch.equal(loaded.head.weight, local["head.weight"])
    assert torch.equal(loaded.head.weight_scale, local["head.weight_scale"])


@pytest.mark.parametrize(
    "extra_member",
    ("mlp.gate.weight_scale", "self_attn.kv_b_proj.weight_scale"),
)
def test_mtp_loader_rejects_packed_payload_for_dense_architectural_exclusion(
    tmp_path, extra_member
):
    config = parse_model_config(_DSV3_RAW)
    metadata = _metadata(ModelScope.MTP_DRAFT)
    dense = MtpDraftHead(config).eval()
    quantized = MtpDraftHead(
        config,
        linear_factory=draft_linear_factory(metadata),
    ).eval()
    local = pack_dynamic_fp8_draft_state(dense.state_dict(), quantized)
    checkpoint = _mtp_checkpoint_state(config, local)
    prefix = f"model.layers.{config.num_hidden_layers}."
    checkpoint[prefix + extra_member] = torch.ones(1)
    raw = {
        **_DSV3_RAW,
        **_raw(ModelScope.MTP_DRAFT, quantization=_mtp_fp8_config()),
    }
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    save_file(checkpoint, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="unexpected=.*weight_scale"):
        load_mtp_head(tmp_path, config)


def test_mtp_loader_rejects_caller_checkpoint_semantic_mismatch(tmp_path):
    config = parse_model_config(_DSV3_RAW)
    head = MtpDraftHead(config).eval()
    checkpoint = _mtp_checkpoint_state(config, dict(head.state_dict()))
    raw = {**_DSV3_RAW, "rms_norm_eps": _DSV3_RAW["rms_norm_eps"] * 2}
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    save_file(checkpoint, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="MTP caller config.*rms_norm_eps"):
        load_mtp_head(tmp_path, config)


def test_draft_factory_rejects_cross_scope_construction():
    metadata = _metadata(ModelScope.EAGLE_DRAFT)
    factory = draft_linear_factory(metadata)
    with pytest.raises(ValueError, match="cannot construct mtp_draft"):
        MtpDraftHead(
            parse_model_config(_DSV3_RAW),
            linear_factory=factory,
        )
