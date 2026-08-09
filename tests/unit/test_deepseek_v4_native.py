from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import kairyu.kernels.deepseek_v4_moe_gpu as moe_gpu
import kairyu.models.deepseek_v4_cache as cache_module
from kairyu.engine.core.deepseek_v4_runner import DeepseekV4DistributedRunner
from kairyu.models.deepseek_v4_cache import (
    CompressedAttentionState,
    PackedFP4IndexerCache,
    _compressed_attention,
    _fp4_index_topk,
    make_deepseek_v4_cache,
)
from kairyu.quant.nvfp4 import E2M1_LUT


def test_attention_dp_empty_rank_still_participates_and_pads(monkeypatch) -> None:
    runner = object.__new__(DeepseekV4DistributedRunner)
    began: list[bool] = []
    counts: list[int] = []
    padding: list[bool] = []
    runner._model = SimpleNamespace(
        _kairyu_begin_attention_dp_phase=lambda: began.append(True)
    )
    monkeypatch.setattr(
        runner,
        "_phase_forward_count",
        lambda local: counts.append(local) or 2,
    )
    monkeypatch.setattr(runner, "_padding_forward", lambda: padding.append(True))

    runner._execute_phase((), {}, {}, prefill=True)

    assert counts == [0]
    assert began == [True]
    assert padding == [True, True]


def _dequant(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    raw = packed.view(torch.uint8)
    nibbles = torch.stack((raw & 0xF, raw >> 4), dim=-1).reshape(raw.shape[0], -1)
    magnitudes = E2M1_LUT.to(raw.device)[(nibbles & 0x7).long()]
    signs = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    values = magnitudes * signs
    return (
        values.view(values.shape[0], scales.shape[1], 32)
        * scales.float().unsqueeze(-1)
    ).reshape(values.shape)


def test_fp4_indexer_cache_is_block32_packed_and_bounded() -> None:
    values = torch.linspace(-6, 6, 3 * 32).view(1, 3, 32)
    cache = PackedFP4IndexerCache(1, 32)
    cache.append(values[:, :2])
    cache.append(values[:, 2:])

    assert cache.shape == (1, 3, 32)
    assert len(cache.packed_chunks) == 1
    assert cache.packed_chunks[0].shape == (3, 16)
    assert cache.packed_chunks[0].dtype is torch.int8
    assert cache.scale_chunks[0].dtype is torch.float8_e8m0fnu
    restored = _dequant(cache.packed_chunks[0], cache.scale_chunks[0])
    assert (restored - values.reshape(3, 32)).abs().max() <= 1.0


def test_dynamic_cache_selects_kairyu_fp4_csa_layer() -> None:
    from transformers import DeepseekV4Config

    config = DeepseekV4Config(num_hidden_layers=1, layer_types=["compressed_sparse_attention"])
    cache = make_deepseek_v4_cache(config)
    assert type(cache.layers[0]).__name__ == "DeepseekV4FP4CSACache"


def test_fp4_indexer_streaming_topk_matches_dense_quantized_scores(monkeypatch) -> None:
    torch.manual_seed(3)
    values = torch.randn(1, 9, 32)
    cache = PackedFP4IndexerCache(1, 32)
    cache.append(values[:, :4])
    cache.append(values[:, 4:])
    restored = _dequant(cache.packed_chunks[0], cache.scale_chunks[0])

    def linear(hidden, packed, scales):
        return hidden.float() @ _dequant(packed, scales).t()

    monkeypatch.setattr(cache_module, "_fp4_linear", linear)
    scorer = SimpleNamespace(
        weights_proj=lambda hidden: hidden[..., :2],
        weights_scaling=2**-0.5,
        softmax_scale=32**-0.5,
    )
    q = torch.randn(1, 3, 2, 32)
    hidden = torch.randn(1, 3, 4)
    threshold = torch.tensor([[2, 6, 9]])
    actual = _fp4_index_topk(scorer, q, cache, hidden, threshold, 2)

    scores = torch.matmul(q.float(), restored.t()).relu() * scorer.softmax_scale
    query_weights = scorer.weights_proj(hidden) * scorer.weights_scaling
    scores = (scores * query_weights.unsqueeze(-1)).sum(dim=2)
    positions = torch.arange(9)
    scores.masked_fill_(positions.view(1, 1, -1) >= threshold.unsqueeze(-1), -torch.inf)
    expected = scores.topk(2, dim=-1).indices
    torch.testing.assert_close(actual, expected)


def test_fixed_ep_dispatch_restores_topk_slot_order(monkeypatch) -> None:
    class Experts:
        gate_up_proj = torch.zeros(2, 8, 4)
        gate_up_proj[1].fill_(1)
        gate_up_proj_scale_inv = torch.ones(2, 8, 1)
        down_proj = torch.zeros(2, 4, 4)
        down_proj[1].fill_(2)
        down_proj_scale_inv = torch.ones(2, 4, 1)

        @staticmethod
        def _apply_gate(value):
            return value[..., :4]

    def fake_linear(hidden, weight, _scale):
        return hidden.float().mean(dim=-1, keepdim=True).expand(
            hidden.shape[0], weight.shape[0]
        ) + weight.flatten()[0]

    monkeypatch.setattr(moe_gpu, "_fp4_linear", fake_linear)
    monkeypatch.setattr(moe_gpu.dist, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(moe_gpu.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        moe_gpu.dist,
        "all_to_all_single",
        lambda output, input, group=None: output.copy_(input),
    )
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)
    indices = torch.tensor([[1, 0], [0, 1]])
    weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.bfloat16)
    output = moe_gpu.deepseek_v4_fp4_ep_forward(
        Experts(), hidden, indices, weights, process_group=object()
    )
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output.float(),
        torch.tensor([[1.75] * 4, [2.2] * 4]),
        atol=2e-2,
        rtol=0,
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_memory_bounded_compressed_attention_matches_dense_math(sparse: bool) -> None:
    torch.manual_seed(11)
    batch, heads, sequence, head_dim = 1, 2, 3, 4
    query = torch.randn(batch, heads, sequence, head_dim)
    sliding = torch.randn(batch, 1, 5, head_dim)
    compressed_keys = torch.randn(batch, 4, head_dim)
    position_ids = torch.tensor([[2, 3, 4]])
    module = SimpleNamespace(sinks=torch.tensor([0.2, -0.1]), scaling=0.5)
    selected = (
        torch.tensor([[[0, -1], [1, 0], [2, 1]]]) if sparse else None
    )
    state = CompressedAttentionState(compressed_keys, 2, selected)

    actual = _compressed_attention(
        module,
        query,
        sliding,
        state,
        attention_mask=None,
        position_ids=position_ids,
    )

    expected = torch.empty(batch, sequence, heads, head_dim)
    for token in range(sequence):
        for head in range(heads):
            keys = [*sliding[0, 0]]
            if sparse:
                keys.extend(
                    compressed_keys[0, index]
                    for index in selected[0, token].tolist()
                    if index >= 0
                )
            else:
                threshold = int((position_ids[0, token] + 1) // 2)
                keys.extend(compressed_keys[0, :threshold])
            values = torch.stack(keys)
            logits = query[0, head, token].matmul(values.t()) * module.scaling
            logits = torch.cat((module.sinks[head].view(1), logits))
            weights = logits.softmax(dim=0)[1:]
            expected[0, token, head] = weights.matmul(values)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
