"""Small CPU contracts for model hot-path setup."""

import torch

from kairyu.models.config import parse_model_config
from kairyu.models.layers import RotaryEmbedding

_TINY = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "vocab_size": 256,
    "max_position_embeddings": 37,
}


def test_rotary_embedding_forward_only_gathers_precomputed_fp32_rows() -> None:
    rotary = RotaryEmbedding(parse_model_config(_TINY))
    positions = torch.tensor([36, 0, 7, 7], dtype=torch.int64)
    frequencies = positions.float()[:, None] * rotary.inv_freq[None, :]
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    expected = embedding.cos(), embedding.sin()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        acc_events=True,
    ) as profile:
        actual = rotary(positions)

    assert rotary.cos_cached.shape == (37, 16)
    assert rotary.sin_cached.shape == (37, 16)
    assert rotary.cos_cached.dtype == rotary.sin_cached.dtype == torch.float32
    assert rotary.state_dict() == {}
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    operations = {event.key for event in profile.key_averages()}
    assert "aten::index_select" in operations
    assert {"aten::cos", "aten::sin", "aten::cat"}.isdisjoint(operations)
