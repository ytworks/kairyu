from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from kairyu.models.config import parse_model_config
from kairyu.models.loader import load_model


def _save_qwen_outer(
    directory,
    model,
    text_config,
    architecture: str,
    *,
    per_expert: bool = False,
) -> None:
    from transformers import Qwen3_5Config

    implementation = model.config._attn_implementation
    outer = Qwen3_5Config(text_config=text_config)
    outer.architectures = [architecture]
    outer.save_pretrained(directory)
    model.config._attn_implementation = implementation
    state = {}
    for name, tensor in model.state_dict().items():
        checkpoint_name = (
            "model.language_model." + name.removeprefix("model.")
            if name.startswith("model.")
            else name
        )
        if per_expert and checkpoint_name.endswith(".experts.gate_up_proj"):
            prefix = checkpoint_name.removesuffix(".gate_up_proj")
            for expert_index, packed in enumerate(tensor):
                gate, up = packed.chunk(2, dim=0)
                state[f"{prefix}.{expert_index}.gate_proj.weight"] = gate.detach().clone()
                state[f"{prefix}.{expert_index}.up_proj.weight"] = up.detach().clone()
        elif per_expert and checkpoint_name.endswith(".experts.down_proj"):
            prefix = checkpoint_name.removesuffix(".down_proj")
            for expert_index, down in enumerate(tensor):
                state[f"{prefix}.{expert_index}.down_proj.weight"] = down.detach().clone()
        else:
            state[checkpoint_name] = tensor.detach().clone()
    save_file(state, directory / "model.safetensors")


@pytest.mark.parametrize(
    ("moe", "per_expert"),
    [
        pytest.param(False, False, id="dense-packed"),
        pytest.param(True, False, id="moe-packed"),
        pytest.param(True, True, id="moe-per-expert"),
    ],
)
def test_qwen36_outer_checkpoint_logits_parity(
    tmp_path,
    moe: bool,
    per_expert: bool,
) -> None:
    if moe:
        from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

        text_config = Qwen3_5MoeTextConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention"],
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=32,
        )
        text_config._attn_implementation = "eager"
        oracle = Qwen3_5MoeForCausalLM(text_config).eval()
        architecture = "Qwen3_5MoeForConditionalGeneration"
    else:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        text_config = Qwen3_5TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=32,
        )
        text_config._attn_implementation = "eager"
        oracle = Qwen3_5ForCausalLM(text_config).eval()
        architecture = "Qwen3_5ForConditionalGeneration"

    _save_qwen_outer(
        tmp_path,
        oracle,
        text_config,
        architecture,
        per_expert=per_expert,
    )
    loaded, config, _ = load_model(tmp_path, dtype=torch.float32)
    tokens = torch.tensor([1, 2, 3, 4])
    with torch.inference_mode():
        expected = oracle(tokens.unsqueeze(0), use_cache=False).logits[0]
        actual = loaded.forward_sequence(tokens)
    assert config.architecture == architecture
    assert config.requires_full_recompute
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_deepseek_v4_checkpoint_logits_parity(tmp_path) -> None:
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

    hf_config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        max_position_embeddings=32,
        sliding_window=4,
        hc_mult=2,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        num_nextn_predict_layers=0,
    )
    oracle = DeepseekV4ForCausalLM(hf_config).eval()
    oracle.save_pretrained(tmp_path, safe_serialization=True)

    loaded, config, _ = load_model(tmp_path, dtype=torch.float32)
    tokens = torch.tensor([1, 2, 3, 4])
    with torch.inference_mode():
        expected = oracle(tokens.unsqueeze(0), use_cache=False).logits[0]
        actual = loaded.forward_sequence(tokens)
    assert config.architecture == "DeepseekV4ForCausalLM"
    assert config.requires_full_recompute
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    bf16_loaded, _, _ = load_model(tmp_path, dtype=torch.bfloat16)
    hf_model = bf16_loaded.hf_model
    assert hf_model.model.embed_tokens.weight.dtype is torch.bfloat16
    assert hf_model.model.layers[0].input_layernorm.weight.dtype is torch.float32


def test_deepseek_v4_public_block_fp8_uses_official_loader(tmp_path, monkeypatch) -> None:
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

    raw = DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        max_position_embeddings=32,
        sliding_window=4,
        hc_mult=2,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        num_nextn_predict_layers=0,
    ).to_dict()
    raw["architectures"] = ["DeepseekV4ForCausalLM"]
    raw["quantization_config"] = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 32))

    seen = {}

    def fake_from_pretrained(path, **kwargs):
        seen.update(path=path, kwargs=kwargs)
        return _Model()

    monkeypatch.setattr(
        DeepseekV4ForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )
    loaded, config, _ = load_model(tmp_path)
    assert config.architecture == "DeepseekV4ForCausalLM"
    assert seen["kwargs"]["attn_implementation"] == "eager"
    assert loaded.forward_sequence(torch.tensor([1, 2])).shape == (2, 32)

    def invalid_checkpoint(*_args, **_kwargs):
        raise ValueError("invalid DeepSeek checkpoint")

    monkeypatch.setattr(DeepseekV4ForCausalLM, "from_pretrained", invalid_checkpoint)
    with pytest.raises(ValueError, match="invalid DeepSeek checkpoint"):
        load_model(tmp_path)


def test_kimi_k3_public_text_config_is_recognized() -> None:
    raw = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "dtype": "bfloat16",
        "text_config": {
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "sliding_window": 16,
            "moe_intermediate_size": 16,
            "num_experts": 8,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "linear_attn_config": {
                "kda_layers": [1, 2, 3],
                "full_attn_layers": [4],
            },
        },
    }
    config = parse_model_config(raw)
    assert config.architecture == "KimiK3ForConditionalGeneration"
    assert config.requires_full_recompute
    assert config.moe is not None
    assert config.moe.num_experts_per_tok == 2


def test_kimi_k3_missing_public_custom_code_fails_clearly(tmp_path) -> None:
    raw = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": {
            "auto_map": {
                "AutoConfig": "configuration_kimi_k3.KimiLinearConfig",
                "AutoModelForCausalLM": ("modeling_kimi_linear.KimiLinearForCausalLM"),
            },
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "moe_intermediate_size": 8,
            "num_experts": 4,
            "num_experts_per_token": 2,
            "max_position_embeddings": 32,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    save_file(
        {"unused": torch.zeros(1)},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(RuntimeError, match="custom Python files"):
        load_model(tmp_path, dtype=torch.float32)


def test_kimi_k3_public_mxfp4_uses_official_loader(tmp_path, monkeypatch) -> None:
    import transformers

    raw = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "moe_intermediate_size": 8,
            "num_experts": 4,
            "num_experts_per_token": 2,
            "max_position_embeddings": 32,
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "mxfp4-pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "type": "float",
                            "num_bits": 4,
                        }
                    }
                },
            },
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))

    class _TextModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, **_kwargs):
            logits = torch.nn.functional.one_hot(input_ids, num_classes=32).to(torch.float32)
            return SimpleNamespace(logits=logits)

    text_model = _TextModel()
    seen = {}

    def fake_from_pretrained(path, **kwargs):
        seen.update(path=path, kwargs=kwargs)
        return SimpleNamespace(language_model=text_model)

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )
    loaded, config, _ = load_model(tmp_path)
    actual = loaded.forward_sequence(torch.tensor([1, 2, 3]))
    assert config.architecture == "KimiK3ForConditionalGeneration"
    assert seen["kwargs"]["trust_remote_code"] is True
    assert actual.shape == (3, 32)

    def invalid_checkpoint(*_args, **_kwargs):
        raise ValueError("invalid Kimi checkpoint")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", invalid_checkpoint)
    with pytest.raises(ValueError, match="invalid Kimi checkpoint"):
        load_model(tmp_path)


def test_qwen36_quantized_reference_is_rejected_by_load_and_validation(
    tmp_path,
) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from kairyu.deploy.validation import _validate_model

    raw = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 24,
        "vocab_size": 32,
        "sliding_window": 8,
        "quantization_config": {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 16,
            "version": "gemm",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    save_file({"unused": torch.zeros(1)}, tmp_path / "model.safetensors")
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "token": 1}, unk_token="[UNK]"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    with pytest.raises(ValueError, match="requires an unquantized checkpoint"):
        load_model(tmp_path)
    findings = _validate_model(
        str(tmp_path),
        field="engines.local.options.model_path",
        tokenizer=None,
        tensor_parallel_size=1,
        reference_artifact=tmp_path / "deploy.yaml",
        generation_config="none",
    )
    assert any(finding.code == "schema.incompatible_model" for finding in findings)


def test_recompute_runner_uses_complete_committed_history() -> None:
    from kairyu.engine.core.recompute_runner import RecomputeModelRunner
    from kairyu.engine.core.scheduler import ScheduledChunk

    class _SequenceModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward_sequence(self, token_ids):
            logits = torch.zeros(len(token_ids), 16)
            logits[-1, int(token_ids.sum().item()) % 16] = 1
            return logits

    runner = RecomputeModelRunner(_SequenceModel())
    request = SimpleNamespace(
        request_id="r",
        prompt_token_ids=(1, 2),
    )
    state = SimpleNamespace(
        request=request,
        outputs=[],
        prefill_done=True,
        computed_prompt=2,
    )
    first = runner.execute(
        (ScheduledChunk("r", 2, True),),
        {"r": state},
    )["r"][0]
    assert first.token_id == 3

    state.outputs.append(first.token_id)
    second = runner.execute(
        (ScheduledChunk("r", 1, False, position=1),),
        {"r": state},
    )["r"][0]
    assert second.token_id == 6

    assert runner._future_tokens == {"r": {1: 6}}


def test_backend_selects_recompute_runner_for_qwen36(tmp_path, monkeypatch) -> None:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from kairyu.engine.core import hw_profile
    from kairyu.engine.core.recompute_runner import RecomputeModelRunner
    from kairyu.engine.kairyu_backend import build_engine_loop

    text_config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=32,
    )
    text_config._attn_implementation = "eager"
    oracle = Qwen3_5ForCausalLM(text_config).eval()
    _save_qwen_outer(
        tmp_path,
        oracle,
        text_config,
        "Qwen3_5ForConditionalGeneration",
    )

    class _Tokenizer:
        eos_token_id = None

        def encode(self, text):
            return (1, 2)

        def decode(self, token_ids):
            return " ".join(map(str, token_ids))

        def vocab(self):
            return [str(index) for index in range(32)]

    monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "torch")
    monkeypatch.setattr(
        hw_profile,
        "probe",
        lambda: hw_profile.HardwareProfile(arch="cpu"),
    )
    with pytest.raises(ValueError, match="single-device only"):
        build_engine_loop(
            model_path=str(tmp_path),
            tokenizer=_Tokenizer(),
            decode_mode="eager",
            tensor_parallel_size=2,
        )

    with pytest.raises(ValueError, match="does not use a KV cache"):
        build_engine_loop(
            model_path=str(tmp_path),
            tokenizer=_Tokenizer(),
            decode_mode="eager",
            kv_cache_dtype="bfloat16",
        )

    loop, _cache, _scheduler = build_engine_loop(
        model_path=str(tmp_path),
        tokenizer=_Tokenizer(),
        decode_mode="eager",
        num_pages=16,
        page_size=4,
    )
    try:
        assert isinstance(loop._runner, RecomputeModelRunner)
        from kairyu import SamplingParams
        from kairyu.engine.prompt import TokensPrompt

        assert loop.kv_cache_dtype_resolved == "not-applicable"

        loop.submit(
            "reference",
            TokensPrompt((1, 2, 3)),
            SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True),
        )
        updates = []
        while loop.has_work():
            updates.extend(loop.step())
        final = next(
            update
            for request_id, update in updates
            if request_id == "reference" and update.finished
        )
        assert len(final.outputs) == 2
    finally:
        loop.close()
