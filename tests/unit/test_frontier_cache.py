from __future__ import annotations

import pytest
import torch

from kairyu.engine.core.frontier_cache import (
    CacheHandle,
    PrefixStateStore,
    cache_descriptor_for_model,
)
from kairyu.models.config import FrontierCacheConfig, ModelConfig


def _config(frontier: FrontierCacheConfig, *, architecture: str) -> ModelConfig:
    return ModelConfig(
        architecture=architecture,
        hidden_size=16,
        num_hidden_layers=len(frontier.layer_types),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        vocab_size=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        dtype="bfloat16",
        max_position_embeddings=1_048_576,
        frontier_cache=frontier,
    )


def test_qwen_frontier_descriptor_exposes_recurrent_and_paged_state() -> None:
    descriptor = cache_descriptor_for_model(
        _config(
            FrontierCacheConfig(
                family="qwen3.5-hybrid-deltanet",
                layer_types=("linear_attention", "full_attention") * 2,
                block_size=256,
                recurrent_state_dtype="float32",
                linear_conv_kernel_dim=4,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                linear_key_head_dim=4,
                linear_value_head_dim=4,
            ),
            architecture="Qwen3_5ForCausalLM",
        )
    )

    assert descriptor.prefix_policy == "complete-state-snapshots"
    assert descriptor.speculative_policy == "copy-on-write"
    assert [component.kind for component in descriptor.components] == [
        "gated-deltanet-recurrent",
        "paged-kv",
    ]
    assert descriptor.components[0].dtype == "float32"


def test_deepseek_frontier_descriptor_exposes_compressed_sparse_layout() -> None:
    descriptor = cache_descriptor_for_model(
        _config(
            FrontierCacheConfig(
                family="deepseek-v4-hca-csa",
                layer_types=(
                    "heavily_compressed_attention",
                    "compressed_sparse_attention",
                ),
                block_size=256,
                sliding_window=4096,
                compress_rate_hca=4,
                compress_rate_csa=128,
                index_topk=2048,
                index_n_heads=64,
                index_head_dim=128,
                hc_mult=4,
                hc_sinkhorn_iters=20,
                fp4_indexer_cache=True,
            ),
            architecture="DeepseekV4ForCausalLM",
        )
    )

    assert descriptor.supported_expert_parallel_sizes == (1, 2, 4, 8)
    assert [component.kind for component in descriptor.components] == [
        "heavily-compressed-attention",
        "compressed-sparse-attention",
        "mhc-state",
    ]
    assert descriptor.components[1].block_size == 256
    assert descriptor.components[1].metadata["fp4_indexer_cache"] is True


def test_cache_handle_transaction_clones_state_for_rollback() -> None:
    original = torch.tensor([1.0, 2.0])
    handle = CacheHandle(
        request_id="r",
        token_ids=(1, 2),
        state={"recurrent": original},
        last_logits=torch.tensor([3.0]),
        output_epoch=7,
    )

    handle.begin_transaction()
    original.add_(10)
    handle.replace((1, 2, 3), {"recurrent": torch.tensor([9.0])}, None, 8)
    handle.rollback()

    assert handle.token_ids == (1, 2)
    torch.testing.assert_close(handle.state["recurrent"], torch.tensor([1.0, 2.0]))
    assert handle.output_epoch == 7
    with pytest.raises(RuntimeError, match="no cache transaction"):
        handle.rollback()


def test_prefix_state_store_is_complete_copying_byte_bounded_lru() -> None:
    store = PrefixStateStore(capacity_bytes=16)
    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([3.0, 4.0])
    third = torch.tensor([5.0, 6.0])

    assert store.put((1,), first)
    assert store.put((1, 2), second)
    assert store.longest_prefix((1, 2, 3))[0] == (1, 2)
    assert store.put((9,), third)
    first.add_(100)

    assert store.longest_prefix((1, 2, 4))[0] == (1, 2)
    assert store.longest_prefix((1, 8)) is None
    restored = store.longest_prefix((9, 10))[1]
    torch.testing.assert_close(restored, torch.tensor([5.0, 6.0]))
