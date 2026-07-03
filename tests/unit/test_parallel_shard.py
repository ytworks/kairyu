"""m16 in-proc gates: shard math, tp_view guards, fake-comm wrappers."""

import pytest
import torch

from kairyu.engine.core.pp_worker import stage_layer_bounds
from kairyu.models.config import parse_model_config
from kairyu.models.parallel import (
    RowParallelLinear,
    shard_bounds,
    shard_dim_for,
    tp_shard_spec,
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
        spec = tp_shard_spec(parse_model_config(TINY))
        assert shard_dim_for("model.layers.0.self_attn.q_proj.weight", spec) == 0
        assert shard_dim_for("model.layers.0.self_attn.q_proj.bias", spec) == 0
        assert shard_dim_for("model.layers.1.self_attn.o_proj.weight", spec) == 1
        assert shard_dim_for("model.layers.0.mlp.down_proj.weight", spec) == 1
        assert shard_dim_for("model.norm.weight", spec) is None
        assert shard_dim_for("lm_head.weight", spec) is None  # replicated

    def test_stage_layer_bounds_early_stages_take_remainder(self):
        assert stage_layer_bounds(5, 2, 0) == (0, 3)
        assert stage_layer_bounds(5, 2, 1) == (3, 5)
        assert stage_layer_bounds(4, 2, 0) == (0, 2)


class TestRowParallelLinear:
    def test_bias_added_once_after_reduce(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(8, 4, bias=True)
        expected = linear(torch.ones(2, 8))
        wrapped = RowParallelLinear(linear, _FakeTensorComm())
        assert linear.bias is None  # detached from the local matmul
        out = wrapped(torch.ones(2, 8))
        assert torch.allclose(out, expected)


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

    def test_quantized_checkpoint_rejected(self, tmp_path):
        import json

        (tmp_path / "config.json").write_text(json.dumps({
            **TINY, "quantization_config": {"quant_method": "awq", "bits": 4,
                                            "group_size": 128, "version": "gemm"},
        }))
        from kairyu.models.parallel import build_tp_model

        with pytest.raises(ValueError, match="tensor parallelism"):
            build_tp_model(str(tmp_path), tp=2, rank=0, comm=_FakeTensorComm())
