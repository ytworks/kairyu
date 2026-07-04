"""m17 D4/D5 gates: EAGLE-3/MTP invariants + synthetic checkpoint round-trips."""

import pytest
import torch

from kairyu.models.config import parse_model_config
from kairyu.models.eagle import EagleConfig, EagleDraftHead, load_eagle_head
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


class TestEagleHead:
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

    def test_d2t_offsets_map_to_target_ids(self):
        torch.manual_seed(1)
        head = EagleDraftHead(EAGLE).eval()
        head.d2t.fill_(1000)  # every draft id maps to draft_id + 1000
        embed = torch.randn(2048, 32)
        drafted = head.rollout(
            embed[:4], torch.randn(4, 32), lambda t: embed[t], k=2
        )
        assert all(token >= 1000 for token in drafted)

    def test_midlayer_qkv_takes_2h(self):
        head = EagleDraftHead(EAGLE)
        assert head.midlayer.self_attn["q_proj"].in_features == 64  # 2H (A2)
        assert head.fc.in_features == 96  # 3H

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

    def test_loader_round_trip_and_format_guards(self, tmp_path):
        from safetensors.torch import save_file

        torch.manual_seed(2)
        head = EagleDraftHead(EAGLE)
        state = dict(head.state_dict())
        # SpecForge extras: t2d present, embed absent (target-aliased)
        state["t2d"] = torch.zeros(128, dtype=torch.bool)
        save_file({k: v.contiguous() for k, v in state.items()},
                  tmp_path / "model.safetensors")
        loaded = load_eagle_head(tmp_path, EAGLE)
        assert torch.equal(loaded.fc.weight, head.fc.weight)
        assert torch.equal(loaded.d2t, head.d2t)

        rogue = dict(state)
        rogue["mystery.weight"] = torch.zeros(1)
        save_file(rogue, tmp_path / "model.safetensors")
        with pytest.raises(KeyError, match="format drift"):
            load_eagle_head(tmp_path, EAGLE)


class TestMtpHead:
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
                checkpoint[prefix + name[len("decoder."):]] = tensor.contiguous()
            else:
                checkpoint[prefix + name] = tensor.contiguous()
        checkpoint["model.layers.0.ignored.weight"] = torch.zeros(1)  # non-MTP
        save_file(checkpoint, tmp_path / "model.safetensors")
        loaded = load_mtp_head(tmp_path, config)
        assert torch.equal(loaded.eh_proj.weight, head.eh_proj.weight)
        assert torch.equal(loaded.head.weight, head.head.weight)

    def test_loader_fails_on_missing_tensors(self, tmp_path):
        from safetensors.torch import save_file

        config = parse_model_config(DSV3_RAW)
        save_file({"model.layers.2.enorm.weight": torch.ones(32)},
                  tmp_path / "model.safetensors")
        with pytest.raises(KeyError, match="missing"):
            load_mtp_head(tmp_path, config)
