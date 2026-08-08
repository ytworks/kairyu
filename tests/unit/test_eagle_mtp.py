"""m17 D4/D5 gates: EAGLE-3/MTP invariants + synthetic checkpoint round-trips."""

import json

import pytest
import torch

from kairyu.models.config import parse_model_config
from kairyu.models.eagle import (
    EagleConfig,
    EagleDraftHead,
    default_eagle3_aux_layer_ids,
    load_eagle_head,
)
from kairyu.models.mtp import MtpDraftHead, load_mtp_head

EAGLE = EagleConfig(
    hidden_size=32, num_attention_heads=4, intermediate_size=64, draft_vocab_size=48
)

DSV3_RAW = {
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


def _eagle_raw(config: EagleConfig) -> dict:
    return {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "draft_vocab_size": config.draft_vocab_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.rms_norm_eps,
    }


class TestEagleHead:
    def test_default_aux_layer_ids_match_sglang_qwen3_32b_capture(self):
        assert default_eagle3_aux_layer_ids(64) == (1, 31, 60)
        with pytest.raises(ValueError, match="at least 7"):
            default_eagle3_aux_layer_ids(6)

    def test_head_dim_must_be_even_for_rope(self):
        with pytest.raises(ValueError, match="must be even"):
            EagleConfig(
                hidden_size=32,
                num_attention_heads=4,
                intermediate_size=64,
                draft_vocab_size=48,
                num_key_value_heads=2,
                head_dim=7,
            )

    def test_fuse_and_rollout_shapes_and_determinism(self):
        torch.manual_seed(0)
        head = EagleDraftHead(EAGLE).eval()
        embed = torch.randn(128, 32)  # stand-in target embedding

        aux = torch.randn(5, 96)  # [T, 3H]
        fused = head.fuse(aux)
        assert fused.shape == (5, 32)

        embeds = embed[torch.arange(5)]
        first = head.rollout(embeds, fused, lambda t: embed[t], k=3)
        second = head.rollout(embeds, fused, lambda t: embed[t], k=3)
        assert first == second and len(first) == 3

    def test_cached_rollout_matches_dense_reference(self):
        torch.manual_seed(17)
        head = EagleDraftHead(EAGLE).eval()
        embed = torch.randn(128, 32)
        hidden = torch.randn(6, 32)
        dense = head.rollout(embed[:6], hidden, lambda token: embed[token], k=4)
        cached = head.rollout_cached(
            embed[:6], hidden, lambda token: embed[token], k=4
        )
        assert cached == dense

    def test_cached_rollout_appends_into_fixed_kv_buffers(self, monkeypatch):
        torch.manual_seed(18)
        head = EagleDraftHead(EAGLE).eval()
        embed = torch.randn(128, 32)
        hidden = torch.randn(6, 32)
        buffers = []
        original = head.midlayer.decode_cached

        def capture(embed_row, feedback, position, key, value):
            buffers.append(
                (key.data_ptr(), value.data_ptr(), key.shape[1], position)
            )
            return original(embed_row, feedback, position, key, value)

        monkeypatch.setattr(head.midlayer, "decode_cached", capture)
        head.rollout_cached(embed[:6], hidden, lambda token: embed[token], k=4)

        assert len(buffers) == 3
        assert {row[:3] for row in buffers} == {
            (buffers[0][0], buffers[0][1], 9)
        }
        assert [row[3] for row in buffers] == [6, 7, 8]

    def test_d2t_offsets_map_to_target_ids(self):
        torch.manual_seed(1)
        head = EagleDraftHead(EAGLE).eval()
        head.d2t.fill_(1000)  # every draft id maps to draft_id + 1000
        embed = torch.randn(2048, 32)
        drafted = head.rollout(embed[:4], torch.randn(4, 32), lambda t: embed[t], k=2)
        assert all(token >= 1000 for token in drafted)

    def test_midlayer_qkv_takes_2h(self):
        head = EagleDraftHead(EAGLE)
        assert head.midlayer.self_attn["q_proj"].in_features == 64  # 2H (A2)
        assert head.fc.in_features == 96  # 3H

    def test_midlayer_gqa_uses_distinct_query_and_kv_geometry(self):
        config = EagleConfig(
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=64,
            draft_vocab_size=48,
            num_key_value_heads=2,
            head_dim=8,
        )
        head = EagleDraftHead(config).eval()
        attention = head.midlayer.self_attn
        assert attention["q_proj"].weight.shape == (32, 64)
        assert attention["k_proj"].weight.shape == (16, 64)
        assert attention["v_proj"].weight.shape == (16, 64)
        assert attention["o_proj"].weight.shape == (32, 32)

        output = head.midlayer(
            torch.randn(3, 32),
            torch.randn(3, 32),
            torch.tensor([0, 1, 2]),
        )
        assert output.shape == (3, 32)

    def test_public_qwen3_32b_eagle3_header_shapes(self):
        # thoughtworks/Qwen3-32B-Eagle3@ba10360 uses the target model's
        # explicit 64Q/8KV x 128 geometry; hidden_size / heads (=80) is not the
        # attention head width.  Construct on meta so this exact 1.56 GB
        # checkpoint contract costs no test memory.
        config = EagleConfig.from_checkpoint_config(
            {
                "hidden_size": 5120,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 25600,
                "draft_vocab_size": 32000,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000,
            }
        )
        with torch.device("meta"):
            state = EagleDraftHead(config).state_dict()

        expected_shapes = {
            "d2t": (32000,),
            "fc.weight": (5120, 15360),
            "lm_head.weight": (32000, 5120),
            "midlayer.hidden_norm.weight": (5120,),
            "midlayer.input_layernorm.weight": (5120,),
            "midlayer.mlp.down_proj.weight": (5120, 25600),
            "midlayer.mlp.gate_proj.weight": (25600, 5120),
            "midlayer.mlp.up_proj.weight": (25600, 5120),
            "midlayer.post_attention_layernorm.weight": (5120,),
            "midlayer.self_attn.k_proj.weight": (1024, 10240),
            "midlayer.self_attn.o_proj.weight": (5120, 8192),
            "midlayer.self_attn.q_proj.weight": (8192, 10240),
            "midlayer.self_attn.v_proj.weight": (1024, 10240),
            "norm.weight": (5120,),
        }
        assert {name: tuple(tensor.shape) for name, tensor in state.items()} == expected_shapes

    def test_midlayer_applies_rope_and_is_position_sensitive(self):
        # H1: the midlayer is RoPE-trained. RoPE encodes RELATIVE position, so a
        # uniform shift is (correctly) invariant, but changing the relative gaps
        # between tokens must change the last token's attention output.
        torch.manual_seed(3)
        head = EagleDraftHead(EAGLE).eval()
        embeds = torch.randn(3, 32)
        hidden = torch.randn(3, 32)
        base = head.midlayer(embeds, hidden, torch.tensor([0, 1, 2]))
        uniform_shift = head.midlayer(embeds, hidden, torch.tensor([10, 11, 12]))
        assert torch.allclose(base, uniform_shift, atol=1e-5)  # relative -> invariant
        spread = head.midlayer(embeds, hidden, torch.tensor([0, 3, 9]))  # different gaps
        assert not torch.allclose(base[-1], spread[-1], atol=1e-4)  # RoPE is applied

    def test_rollout_freezes_context_hidden_inputs(self):
        # H2: each draft step must APPEND one feedback row and leave the context
        # rows' input hidden untouched — not overwrite the whole hidden with the
        # midlayer output. We spy on the hidden passed into the midlayer.
        torch.manual_seed(4)
        head = EagleDraftHead(EAGLE).eval()
        embed = torch.randn(128, 32)
        initial = torch.randn(4, 32)
        seen_hidden = []
        original = head.midlayer.forward

        def spy(embeds, hidden, positions=None):
            seen_hidden.append(hidden.clone())
            return original(embeds, hidden, positions)

        head.midlayer.forward = spy  # type: ignore[method-assign]
        head.rollout(embed[:4], initial, lambda t: embed[t], k=3)
        # every step's first 4 (context) hidden rows equal the ORIGINAL fuse rows
        for hidden in seen_hidden:
            assert torch.equal(hidden[:4], initial)

    def test_rollout_requires_shifted_embeddings_to_pair_with_aux_rows(self):
        head = EagleDraftHead(EAGLE).eval()
        embed = torch.randn(128, 32)
        initial = torch.randn(4, 32)

        with pytest.raises(ValueError, match="pair one-for-one"):
            head.rollout(embed[:3], initial, lambda token: embed[token], k=1)
        with pytest.raises(ValueError, match="nonempty rank-2"):
            head.rollout(embed[:0], initial[:0], lambda token: embed[token], k=1)
        with pytest.raises(ValueError, match="nonempty rank-2"):
            head.rollout(embed[:4], torch.tensor(1.0), lambda token: embed[token], k=1)
        with pytest.raises(ValueError, match="positive integer"):
            head.rollout(embed[:4], initial, lambda token: embed[token], k=0)

    def test_loader_round_trip_and_format_guards(self, tmp_path):
        from safetensors.torch import save_file

        torch.manual_seed(2)
        head = EagleDraftHead(EAGLE)
        state = dict(head.state_dict())
        state["d2t"] = (torch.arange(EAGLE.draft_vocab_size) - 24).to(torch.int32)
        # SpecForge extras: t2d present, embed absent (target-aliased)
        state["t2d"] = torch.zeros(128, dtype=torch.bool)
        (tmp_path / "config.json").write_text(json.dumps(_eagle_raw(EAGLE)), encoding="utf-8")
        save_file({k: v.contiguous() for k, v in state.items()}, tmp_path / "model.safetensors")
        loaded = load_eagle_head(tmp_path, EAGLE)
        assert torch.equal(loaded.fc.weight, head.fc.weight)
        assert loaded.d2t.dtype is torch.int64
        assert torch.equal(loaded.d2t, state["d2t"].to(torch.int64))

        for rogue_name in (
            "mystery.weight",
            "rogue.t2d",
            "rogue.embed_tokens.weight",
        ):
            rogue = dict(state)
            rogue[rogue_name] = torch.zeros(1)
            save_file(
                {name: tensor.contiguous() for name, tensor in rogue.items()},
                tmp_path / "model.safetensors",
            )
            with pytest.raises(KeyError, match="format drift"):
                load_eagle_head(tmp_path, EAGLE)

    def test_loader_requires_config_json(self, tmp_path):
        from safetensors.torch import save_file

        save_file(
            {
                name: tensor.contiguous()
                for name, tensor in EagleDraftHead(EAGLE).state_dict().items()
            },
            tmp_path / "model.safetensors",
        )
        with pytest.raises(ValueError, match="no config.json"):
            load_eagle_head(tmp_path, EAGLE)


class TestMtpHead:
    def test_mla_partial_cached_suffix_uses_host_metadata(self, monkeypatch):
        from kairyu.engine.core.kv_pool import PagedKVPool
        from kairyu.engine.core.radix_kv import RadixKVCache
        from kairyu.models.llama import DenseDecoder

        torch.manual_seed(319)
        config = parse_model_config(DSV3_RAW)
        model = DenseDecoder(config).eval()
        pool = PagedKVPool.for_cache(
            RadixKVCache(num_pages=4, page_size=4), config
        )
        attention = model.model.layers[0].self_attn
        positions = torch.arange(2)
        cos, sin = model.model.rotary_emb(positions)
        attention(
            torch.randn(2, config.hidden_size),
            cos,
            sin,
            pool,
            0,
            [0],
            positions,
            2,
            0,
            chunk_start=0,
            has_writable=True,
        )
        snapshot = pool.k.clone()

        def fail(name):
            def method(self, *args, **kwargs):
                raise AssertionError(f"host tensor scalar read through {name}")

            return method

        monkeypatch.setattr(torch.Tensor, "any", fail("any"))
        monkeypatch.setattr(torch.Tensor, "item", fail("item"))
        monkeypatch.setattr(torch.Tensor, "__bool__", fail("bool"))
        attention(
            torch.randn(2, config.hidden_size),
            cos,
            sin,
            pool,
            0,
            [0],
            positions,
            2,
            1,
            chunk_start=0,
            has_writable=True,
        )

        assert torch.equal(pool.k[0, 0, 0], snapshot[0, 0, 0])
        assert not torch.equal(pool.k[0, 0, 1], snapshot[0, 0, 1])

    def test_forward_chain_passes_host_metadata_to_mla(self, monkeypatch):
        from kairyu.models.layers import RotaryEmbedding

        config = parse_model_config(DSV3_RAW)
        head = MtpDraftHead(config).eval()
        original = head.decoder.self_attn.forward
        calls = []

        def capture(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(head.decoder.self_attn, "forward", capture)
        head.forward_chain(
            torch.tensor([1, 2]),
            torch.randn(2, config.hidden_size),
            RotaryEmbedding(config),
        )

        assert len(calls) == 1
        assert calls[0]["chunk_start"] == 0
        assert calls[0]["has_writable"] is True

    def test_forward_chain_shapes_and_moe_block(self):
        from kairyu.models.moe import DeepseekV3MoeBlock

        torch.manual_seed(3)
        config = parse_model_config(DSV3_RAW)
        head = MtpDraftHead(config).eval()
        # layer_index = num_hidden_layers >= first_k_dense_replace -> MoE (A8)
        assert isinstance(head.decoder.mlp, DeepseekV3MoeBlock)

        from kairyu.models.layers import RotaryEmbedding

        rotary = RotaryEmbedding(config)
        tokens = torch.randint(0, 128, (6,))
        target_hidden = torch.randn(6, 32)
        out = head.forward_chain(tokens, target_hidden, rotary)
        assert out.shape == (6, 32)
        logits = head.logits(out[-1])
        assert logits.shape == (128,)
        again = head.forward_chain(tokens, target_hidden, rotary)
        assert torch.allclose(out, again)

    def test_cached_rollout_reuses_context_kv(self, monkeypatch):
        from kairyu.models.layers import RotaryEmbedding

        torch.manual_seed(31)
        config = parse_model_config(DSV3_RAW)
        head = MtpDraftHead(config).eval()
        rotary = RotaryEmbedding(config)
        tokens = torch.randint(0, 128, (5,))
        hidden = torch.randn(5, 32)
        expected_first = int(
            torch.argmax(head.logits(head.forward_chain(tokens, hidden, rotary)[-1]))
        )
        widths = []
        pools = []
        original = head.decoder.forward

        def capture(*args, **kwargs):
            widths.append(args[0].shape[0])
            pools.append(args[3])
            return original(*args, **kwargs)

        monkeypatch.setattr(head.decoder, "forward", capture)
        drafted = head.rollout_cached(tokens, hidden, rotary, 3)

        assert drafted[0] == expected_first
        assert widths == [5, 1, 1]
        assert len({id(pool) for pool in pools}) == 1

    def test_cached_rollout_recycles_post_shared_head_norm(self):
        from kairyu.models.layers import RotaryEmbedding

        torch.manual_seed(32)
        config = parse_model_config(DSV3_RAW)
        head = MtpDraftHead(config).eval()
        rotary = RotaryEmbedding(config)
        tokens = torch.randint(0, 128, (4,))
        hidden = torch.randn(4, 32)
        shared_outputs = []
        hidden_inputs = []

        shared_hook = head.shared_head["norm"].register_forward_hook(
            lambda _module, _inputs, output: shared_outputs.append(output.detach().clone())
        )
        hidden_hook = head.hnorm.register_forward_pre_hook(
            lambda _module, inputs: hidden_inputs.append(inputs[0].detach().clone())
        )
        try:
            head.rollout_cached(tokens, hidden, rotary, 2)
        finally:
            shared_hook.remove()
            hidden_hook.remove()

        assert len(shared_outputs) == 2
        assert len(hidden_inputs) == 2
        assert torch.equal(hidden_inputs[1], shared_outputs[0][None])

    def test_loader_maps_extra_layer_names(self, tmp_path):
        from safetensors.torch import save_file

        torch.manual_seed(4)
        config = parse_model_config(DSV3_RAW)
        head = MtpDraftHead(config)
        prefix = f"model.layers.{config.num_hidden_layers}."
        checkpoint: dict[str, torch.Tensor] = {}
        for name, tensor in head.state_dict().items():
            if name == "head.weight":
                checkpoint[prefix + "shared_head.head.weight"] = tensor.contiguous()
            elif name.startswith("decoder."):
                checkpoint[prefix + name[len("decoder.") :]] = tensor.contiguous()
            else:
                checkpoint[prefix + name] = tensor.contiguous()
        checkpoint["model.layers.0.ignored.weight"] = torch.zeros(1)  # non-MTP
        (tmp_path / "config.json").write_text(json.dumps(DSV3_RAW), encoding="utf-8")
        save_file(checkpoint, tmp_path / "model.safetensors")
        loaded = load_mtp_head(tmp_path, config)
        assert torch.equal(loaded.eh_proj.weight, head.eh_proj.weight)
        assert torch.equal(loaded.head.weight, head.head.weight)

    def test_loader_fails_on_missing_tensors(self, tmp_path):
        from safetensors.torch import save_file

        config = parse_model_config(DSV3_RAW)
        (tmp_path / "config.json").write_text(json.dumps(DSV3_RAW), encoding="utf-8")
        save_file({"model.layers.2.enorm.weight": torch.ones(32)}, tmp_path / "model.safetensors")
        with pytest.raises(KeyError, match="missing"):
            load_mtp_head(tmp_path, config)

    def test_loader_requires_config_json(self, tmp_path):
        from safetensors.torch import save_file

        config = parse_model_config(DSV3_RAW)
        save_file(
            {"model.layers.2.enorm.weight": torch.ones(32)},
            tmp_path / "model.safetensors",
        )
        with pytest.raises(ValueError, match="no config.json"):
            load_mtp_head(tmp_path, config)
