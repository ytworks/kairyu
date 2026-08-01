"""Target hidden-state capture contract used by EAGLE-3 verification."""

from __future__ import annotations

import pytest
import torch

from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.models.config import parse_model_config
from kairyu.models.llama import DenseDecoder

_CONFIG = parse_model_config(
    {
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
    }
)


def _pool() -> PagedKVPool:
    return PagedKVPool(
        num_layers=_CONFIG.num_hidden_layers,
        num_pages=8,
        page_size=4,
        num_kv_heads=_CONFIG.kv_cache_num_heads,
        head_dim=_CONFIG.kv_cache_head_dim,
        v_head_dim=_CONFIG.kv_cache_v_head_dim,
    )


def test_aux_capture_matches_same_pass_layer_outputs_and_final_hidden():
    torch.manual_seed(234)
    model = DenseDecoder(_CONFIG).eval()
    token_ids = torch.tensor([3, 5, 8, 13])
    positions = torch.arange(token_ids.numel())
    page_table = [0, 1]
    observed: dict[int, torch.Tensor] = {}
    handles = [
        model.model.layers[index].register_forward_hook(
            lambda _module, _args, output, layer_id=index: observed.__setitem__(
                layer_id, output.detach().clone()
            )
        )
        for index in (0, 2, 3)
    ]
    try:
        ordinary = model.forward_tokens(
            token_ids,
            positions,
            _pool(),
            page_table,
            seq_len=token_ids.numel(),
        )
    finally:
        for handle in handles:
            handle.remove()

    captured, aux = model.forward_tokens_with_aux(
        token_ids,
        positions,
        _pool(),
        page_table,
        seq_len=token_ids.numel(),
        aux_layer_ids=(0, 2, 3),
    )
    assert torch.equal(captured, ordinary)
    assert torch.equal(aux, torch.cat([observed[0], observed[2], observed[3]], dim=-1))


@pytest.mark.parametrize(
    ("layer_ids", "match"),
    [
        ((), "at least one"),
        ((1, 1), "unique"),
        ((2, 0), "strictly increasing"),
        ((-1,), "within"),
        ((4,), "within"),
    ],
)
def test_aux_capture_rejects_ambiguous_or_invalid_layer_ids(layer_ids, match):
    model = DenseDecoder(_CONFIG).eval()
    with pytest.raises(ValueError, match=match):
        model.forward_tokens_with_aux(
            torch.tensor([1]),
            torch.tensor([0]),
            _pool(),
            [0],
            seq_len=1,
            aux_layer_ids=layer_ids,
        )
