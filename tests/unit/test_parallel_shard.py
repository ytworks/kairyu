"""m16 in-proc gates: shard math, tp_view guards, fake-comm wrappers."""

import pytest
import torch

from kairyu.engine.core.pp_worker import stage_layer_bounds
from kairyu.models.config import parse_model_config
from kairyu.models.parallel import (
    RowParallelLinear,
    checkpoint_member_specs,
    shard_bounds,
    tp_view,
)

TINY = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "vocab_size": 256,
    "max_position_embeddings": 512,
}


class _FakeTensorComm:
    """world=1 tensor comm: all_reduce identity (partial sum IS the sum)."""

    def tensor_all_reduce(self, tensor):
        return tensor

    def tensor_all_reduce_max(self, tensor):
        return tensor


class TestShardMath:
    def test_shard_bounds(self):
        assert shard_bounds(8, 2, 0) == (0, 4)
        assert shard_bounds(8, 2, 1) == (4, 8)
        with pytest.raises(ValueError, match="divide"):
            shard_bounds(9, 2, 0)

    def test_tp_view_divides_heads_and_intermediate(self):
        config = parse_model_config(TINY)
        local = tp_view(config, 2, 0)
        assert local.num_attention_heads == 2
        assert local.num_key_value_heads == 1
        assert local.intermediate_size == 64
        assert local.head_dim == config.head_dim  # NOT divided
        assert local.vocab_size == config.vocab_size  # replicated head

    def test_tp_view_rejects_indivisible_and_mla(self):
        config = parse_model_config(TINY)
        with pytest.raises(ValueError):
            tp_view(config, 3, 0)  # 4 heads % 3
        mla_raw = {
            **TINY,
            "architectures": ["DeepseekV3ForCausalLM"],
            "kv_lora_rank": 16,
            "q_lora_rank": None,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 4,
            "v_head_dim": 12,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "n_group": 2,
            "topk_group": 1,
        }
        with pytest.raises(ValueError, match="attention-DP"):
            tp_view(parse_model_config(mla_raw), 2, 0)

    def test_shard_spec_dims(self):
        from kairyu.engine.core.quant_config import QuantConfig, QuantMethod
        from kairyu.models.llama import DenseDecoder
        from kairyu.quant.linear import linear_factory

        config = parse_model_config(TINY)
        model = DenseDecoder(
            tp_view(config, 2, 0),
            linear_factory=linear_factory(
                QuantConfig(QuantMethod.NONE),
                tensor_parallel_size=2,
                tensor_parallel_rank=0,
            ),
        )
        spec = checkpoint_member_specs(model)
        assert spec["model.layers.0.self_attn.q_proj.weight"].shard_dim == 0
        assert spec["model.layers.1.self_attn.o_proj.weight"].shard_dim == 1
        assert spec["model.layers.0.mlp.down_proj.weight"].shard_dim == 1
        assert spec["model.norm.weight"].shard_dim is None
        assert spec["lm_head.weight"].shard_dim is None  # replicated

    def test_stage_layer_bounds_early_stages_take_remainder(self):
        assert stage_layer_bounds(5, 2, 0) == (0, 3)
        assert stage_layer_bounds(5, 2, 1) == (3, 5)
        assert stage_layer_bounds(4, 2, 0) == (0, 2)


class TestRowParallelLinear:
    def test_bias_added_once_after_reduce(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(8, 4, bias=True)
        expected = linear(torch.ones(2, 8))
        bias = linear.bias
        wrapped = RowParallelLinear(linear, _FakeTensorComm())
        assert wrapped is linear
        assert linear.bias is bias  # canonical owner stays on the projection
        out = wrapped(torch.ones(2, 8))
        assert torch.allclose(out, expected)

    def test_dynamic_fp8_uses_global_per_token_scale(self):
        from kairyu.quant.linear import Fp8Linear

        class RecordingFp8(Fp8Linear):
            def __init__(self):
                super().__init__(4, 3, bias=False)
                self.seen_scale = None

            def activation_amax(self, x):
                return torch.tensor([[2.0], [6.0]])

            def forward_with_activation_scale(self, x, activation_scale):
                self.seen_scale = activation_scale.clone()
                return x.new_zeros((x.shape[0], self.out_features))

        class GlobalMaxComm:
            def __init__(self):
                self.local_amax = None

            def tensor_all_reduce_max(self, tensor):
                self.local_amax = tensor.clone()
                return torch.tensor([[4.0], [8.0]])

            def tensor_all_reduce(self, tensor):
                return tensor

        local = RecordingFp8()
        comm = GlobalMaxComm()
        wrapped = RowParallelLinear(local, comm)
        wrapped(torch.ones(2, 4))

        assert torch.equal(comm.local_amax, torch.tensor([[2.0], [6.0]]))
        torch.testing.assert_close(
            local.seen_scale,
            torch.tensor([[4.0 / 448.0], [8.0 / 448.0]]),
        )


class TestTp1Equivalence:
    """build_tp_model(tp=1) with an identity comm == plain load_model."""

    def test_logits_identical(self, tmp_path):
        transformers = pytest.importorskip("transformers")
        torch.manual_seed(5)
        hf = transformers.Qwen2ForCausalLM(
            transformers.Qwen2Config(
                vocab_size=256, hidden_size=64, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=512,
            )
        ).to(torch.float32).eval()
        hf.save_pretrained(tmp_path, safe_serialization=True)

        from kairyu.engine.core.kv_pool import PagedKVPool
        from kairyu.engine.core.radix_kv import RadixKVCache
        from kairyu.models.loader import load_model
        from kairyu.models.parallel import build_tp_model

        plain, config, _ = load_model(tmp_path)
        sharded, local_config, full_config = build_tp_model(
            str(tmp_path), tp=1, rank=0, comm=_FakeTensorComm()
        )
        assert local_config.num_attention_heads == full_config.num_attention_heads

        torch.manual_seed(7)
        prompt = torch.randint(0, 256, (10,))
        cache = RadixKVCache(num_pages=16, page_size=4)

        def logits(model, cfg):
            pool = PagedKVPool.for_cache(cache, cfg)
            hidden = model.forward_tokens(
                prompt, torch.arange(10), pool, [0, 1, 2], seq_len=10
            )
            return model.logits(hidden)

        assert torch.equal(logits(plain, config), logits(sharded, local_config))

    def test_non_fp8_quantized_checkpoint_rejected(self, tmp_path):
        import json

        (tmp_path / "config.json").write_text(json.dumps({
            **TINY, "quantization_config": {"quant_method": "awq", "bits": 4,
                                            "group_size": 128, "version": "gemm"},
        }))
        from kairyu.models.parallel import build_tp_model

        with pytest.raises(ValueError, match="supports dense and FP8"):
            build_tp_model(str(tmp_path), tp=2, rank=0, comm=_FakeTensorComm())

    def test_fp8_checkpoint_shards_weights_and_output_scales(self, tmp_path):
        transformers = pytest.importorskip("transformers")
        from kairyu.engine.core.weights import CheckpointReader
        from kairyu.models.parallel import build_tp_model
        from kairyu.quant.linear import Fp8Linear
        from tests.quant_checkpoint_fixtures import write_quantized_checkpoint

        torch.manual_seed(17)
        hf = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY))
        source = tmp_path / "source"
        target = tmp_path / "fp8"
        source.mkdir()
        target.mkdir()
        hf.to(torch.float32).eval().save_pretrained(
            source, safe_serialization=True
        )
        write_quantized_checkpoint("fp8", source, hf, target)
        from safetensors.torch import load_file, save_file

        checkpoint = target / "model.safetensors"
        state = load_file(checkpoint)
        for name, tensor in tuple(state.items()):
            if name.endswith(".weight_scale"):
                state[name] = tensor.to(torch.bfloat16)
        save_file(state, checkpoint)

        rank0, _, _ = build_tp_model(
            str(target), tp=2, rank=0, comm=_FakeTensorComm()
        )
        rank1, _, _ = build_tp_model(
            str(target), tp=2, rank=1, comm=_FakeTensorComm()
        )
        q0 = rank0.model.layers[0].self_attn.q_proj
        q1 = rank1.model.layers[0].self_attn.q_proj
        o0 = rank0.model.layers[0].self_attn.o_proj
        o1 = rank1.model.layers[0].self_attn.o_proj
        assert all(isinstance(layer, Fp8Linear) for layer in (q0, q1, o0, o1))

        reader = CheckpointReader(target)
        q_name = "model.layers.0.self_attn.q_proj"
        o_name = "model.layers.0.self_attn.o_proj"
        assert torch.equal(
            torch.cat((q0.weight, q1.weight), dim=0),
            reader.tensor(f"{q_name}.weight"),
        )
        assert torch.equal(
            torch.cat((q0.weight_scale, q1.weight_scale), dim=0),
            reader.tensor(f"{q_name}.weight_scale"),
        )
        assert torch.equal(
            torch.cat((o0.weight, o1.weight), dim=1),
            reader.tensor(f"{o_name}.weight"),
        )
        assert torch.equal(o0.weight_scale, reader.tensor(f"{o_name}.weight_scale"))
        assert torch.equal(o1.weight_scale, reader.tensor(f"{o_name}.weight_scale"))
        assert q0.weight.dtype is torch.float8_e4m3fn
        assert q0.weight_scale.dtype is torch.float32
