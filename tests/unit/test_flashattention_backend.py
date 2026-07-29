from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from kairyu.engine.core.attention import graph_capture_gap
from kairyu.engine.core.attention.flashattention_gpu import (
    FlashAttentionBackend,
)
from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
from kairyu.engine.core.kv_pool import PagedKVPool


def _bottom_right_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Small CPU oracle for the fake upstream modules (BSHD inputs)."""
    groups = query.shape[2] // keys.shape[2]
    keys = keys.repeat_interleave(groups, dim=2)
    values = values.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bthd,bshd->bhts", query, keys)
    scores = scores / math.sqrt(query.shape[-1])
    if causal:
        query_positions = (
            torch.arange(query.shape[1], device=query.device) + keys.shape[1] - query.shape[1]
        )
        key_positions = torch.arange(keys.shape[1], device=query.device)
        keep = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~keep[None, None], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bhts,bshd->bthd", probabilities, values)


class _FakeFA3:
    __version__ = "3.fake"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def flash_attn_with_kvcache(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "query": query,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "cache_seqlens": cache_seqlens,
                "page_table": page_table,
                "causal": causal,
            }
        )
        length = int(cache_seqlens[0])
        pages = page_table[0].long()
        keys = k_cache[pages].flatten(0, 1)[:length].unsqueeze(0)
        values = v_cache[pages].flatten(0, 1)[:length].unsqueeze(0)
        return _bottom_right_attention(query, keys, values, causal=causal)


class _FakeFA4:
    __version__ = "4.0.0b24.fake"

    def __init__(self) -> None:
        self.paged_calls: list[dict[str, object]] = []
        self.dense_calls: list[dict[str, object]] = []

    def flash_attn_varlen_func(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        max_seqlen_q: int,
        max_seqlen_k: int,
        seqused_k: torch.Tensor,
        page_table: torch.Tensor,
        causal: bool,
    ) -> tuple[torch.Tensor, None]:
        self.paged_calls.append(
            {
                "query": query,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "seqused_k": seqused_k,
                "page_table": page_table,
                "causal": causal,
            }
        )
        length = int(seqused_k[0])
        pages = page_table[0].long()
        keys = k_cache[pages].flatten(0, 1)[:length].unsqueeze(0)
        values = v_cache[pages].flatten(0, 1)[:length].unsqueeze(0)
        return _bottom_right_attention(query, keys, values, causal=causal), None

    def flash_attn_func(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        causal: bool,
    ) -> tuple[torch.Tensor, None]:
        self.dense_calls.append(
            {
                "query": query,
                "keys": keys,
                "values": values,
                "causal": causal,
            }
        )
        return _bottom_right_attention(query, keys, values, causal=causal), None


class _FakeDecodeBackend:
    __version__ = "0.6.fake"
    supports_graph_capture = True

    def __init__(self) -> None:
        self.attend_calls: list[tuple] = []
        self.batch_calls: list[tuple] = []
        self.plan_calls: list[tuple] = []
        self.decode_calls: list[tuple] = []

    def attend(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_table: list[int],
        seq_len: int,
        chunk_start: int,
    ) -> torch.Tensor:
        self.attend_calls.append((query, kv_pool, layer, page_table, seq_len, chunk_start))
        return query.reshape(query.shape[0], -1) + 10

    def attend_batched(
        self,
        queries: list[torch.Tensor],
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: list[list[int]],
        seq_lens: list[int],
        chunk_starts: list[int],
    ) -> list[torch.Tensor]:
        self.batch_calls.append((queries, kv_pool, layer, page_tables, seq_lens, chunk_starts))
        return [query.reshape(query.shape[0], -1) + 20 for query in queries]

    def plan_decode(
        self,
        kv_pool: PagedKVPool,
        page_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        num_qo_heads: int,
        q_dtype: torch.dtype,
    ) -> None:
        self.plan_calls.append((kv_pool, page_tables, seq_lens, num_qo_heads, q_dtype))

    def attend_decode(
        self,
        query: torch.Tensor,
        kv_pool: PagedKVPool,
        layer: int,
        page_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        self.decode_calls.append((query, kv_pool, layer, page_tables, seq_lens))
        return query.reshape(query.shape[0], -1) + 30


def _case() -> tuple[PagedKVPool, list[int], torch.Tensor]:
    generator = torch.Generator().manual_seed(277)
    pool = PagedKVPool(
        num_layers=1,
        num_pages=5,
        page_size=2,
        num_kv_heads=2,
        head_dim=8,
    )
    page_table = [3, 1, 4]
    keys = torch.randn(5, 2, 8, generator=generator)
    values = torch.randn(5, 2, 8, generator=generator)
    pool.write(
        0,
        page_table,
        torch.arange(5),
        keys,
        values,
    )
    query = torch.randn(2, 4, 8, generator=generator)
    return pool, page_table, query


def _backend(
    generation: int,
    module: object,
    *,
    sm: int,
    decode: _FakeDecodeBackend | None = None,
) -> FlashAttentionBackend:
    return FlashAttentionBackend(
        generation,
        flashattention_module=module,
        decode_backend=decode or _FakeDecodeBackend(),
        capability=sm,
    )


@pytest.mark.parametrize("sm", [80, 90])
def test_fa3_paged_api_preserves_page_identity_gqa_and_causal_contract(sm):
    pool, page_table, query = _case()
    module = _FakeFA3()
    backend = _backend(3, module, sm=sm)

    output = backend.attend(query, pool, 0, page_table, 5, 3)
    reference = TorchAttentionBackend().attend(query, pool, 0, page_table, 5, 3)

    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
    assert len(module.calls) == 1
    call = module.calls[0]
    assert call["query"].shape == (1, 2, 4, 8)
    assert call["k_cache"].data_ptr() == pool.k[0].data_ptr()
    assert call["v_cache"].data_ptr() == pool.v[0].data_ptr()
    assert call["cache_seqlens"].dtype == torch.int32
    assert call["cache_seqlens"].tolist() == [5]
    assert call["page_table"].dtype == torch.int32
    assert call["page_table"].tolist() == [[3, 1, 4]]
    assert call["causal"] is True
    assert backend.components == {
        "prefill": "flashattention3",
        "decode": "flashinfer",
        "kv_mode": "paged-direct",
    }


def test_fa4_sm100_uses_varlen_paged_api_with_logical_cache_length():
    pool, page_table, query = _case()
    module = _FakeFA4()
    backend = _backend(4, module, sm=100)

    output = backend.attend(query, pool, 0, page_table, 5, 3)
    reference = TorchAttentionBackend().attend(query, pool, 0, page_table, 5, 3)

    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
    assert not module.dense_calls
    assert len(module.paged_calls) == 1
    call = module.paged_calls[0]
    assert call["k_cache"].data_ptr() == pool.k[0].data_ptr()
    assert call["v_cache"].data_ptr() == pool.v[0].data_ptr()
    assert call["page_table"].tolist() == [[3, 1, 4]]
    assert call["page_table"].dtype == torch.int32
    assert call["seqused_k"].tolist() == [5]
    assert call["seqused_k"].dtype == torch.int32
    assert call["max_seqlen_q"] == 2
    # The compile-time maximum is page-aligned and bound to table width;
    # ``seqused_k`` above carries the exact, unaligned logical length.
    assert call["max_seqlen_k"] == 6
    assert call["causal"] is True
    assert backend.components["kv_mode"] == "paged-direct"


@pytest.mark.parametrize("sm", [90, 120])
def test_fa4_sm90_and_sm120_materialize_selected_pages_for_dense_api(sm):
    pool, page_table, query = _case()
    module = _FakeFA4()
    backend = _backend(4, module, sm=sm)

    output = backend.attend(query, pool, 0, page_table, 5, 3)
    reference = TorchAttentionBackend().attend(query, pool, 0, page_table, 5, 3)

    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
    assert not module.paged_calls
    assert len(module.dense_calls) == 1
    call = module.dense_calls[0]
    expected_keys, expected_values = pool.gather(0, page_table, 5)
    torch.testing.assert_close(call["keys"], expected_keys.unsqueeze(0))
    torch.testing.assert_close(call["values"], expected_values.unsqueeze(0))
    assert call["keys"].shape == (1, 5, 2, 8)
    assert call["causal"] is True
    assert backend.components["kv_mode"] == "paged-materialized"


def test_decode_list_tensor_and_graph_contracts_are_explicitly_delegated():
    pool, page_table, _ = _case()
    module = _FakeFA4()
    delegate = _FakeDecodeBackend()
    backend = _backend(4, module, sm=120, decode=delegate)
    query = torch.randn(1, 4, 8)

    eager = backend.attend(query, pool, 0, page_table, 5, 4)
    torch.testing.assert_close(eager, query.reshape(1, -1) + 10)
    assert len(delegate.attend_calls) == 1
    assert not module.dense_calls and not module.paged_calls

    page_tables = torch.tensor([page_table], dtype=torch.int32)
    seq_lens = torch.tensor([5], dtype=torch.int32)
    backend.plan_decode(
        pool,
        page_tables,
        seq_lens,
        num_qo_heads=4,
        q_dtype=query.dtype,
    )
    graph_output = backend.attend_decode(query, pool, 0, page_tables, seq_lens)
    torch.testing.assert_close(graph_output, query.reshape(1, -1) + 30)
    assert len(delegate.plan_calls) == 1
    assert len(delegate.decode_calls) == 1
    assert backend.supports_graph_capture is True
    assert graph_capture_gap(backend) is None
    assert backend.components["decode"] == "flashinfer"
    assert not isinstance(backend, torch.nn.Module)


def test_mixed_batch_groups_decode_and_keeps_prefill_output_order():
    pool, page_table, prefill = _case()
    module = _FakeFA4()
    delegate = _FakeDecodeBackend()
    backend = _backend(4, module, sm=120, decode=delegate)
    decode_a = torch.zeros(1, 4, 8)
    decode_b = torch.ones(1, 4, 8)

    outputs = backend.attend_batched(
        [decode_a, prefill, decode_b],
        pool,
        0,
        [page_table, page_table, page_table],
        [5, 5, 5],
        [4, 3, 4],
    )

    assert len(outputs) == 3
    torch.testing.assert_close(outputs[0], decode_a.reshape(1, -1) + 20)
    torch.testing.assert_close(
        outputs[1],
        TorchAttentionBackend().attend(prefill, pool, 0, page_table, 5, 3),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(outputs[2], decode_b.reshape(1, -1) + 20)
    assert len(delegate.batch_calls) == 1
    assert len(delegate.batch_calls[0][0]) == 2
    assert len(module.dense_calls) == 1


def test_parallel_lists_must_have_identical_lengths():
    pool, page_table, query = _case()
    backend = _backend(3, _FakeFA3(), sm=90)

    with pytest.raises(ValueError, match="same length"):
        backend.attend_batched(
            [query],
            pool,
            0,
            [page_table],
            [5],
            [],
        )


def test_prefill_rejects_non_tail_bad_gqa_and_missing_pages():
    pool, page_table, query = _case()
    backend = _backend(3, _FakeFA3(), sm=90)

    with pytest.raises(ValueError, match="bottom-right"):
        backend.attend(query, pool, 0, page_table, 5, 2)
    with pytest.raises(ValueError, match="GQA"):
        backend.attend(torch.randn(2, 3, 8), pool, 0, page_table, 5, 3)
    with pytest.raises(ValueError, match="requires 3"):
        backend.attend(query, pool, 0, page_table[:2], 5, 3)


def test_model_config_validation_fails_before_serving_unsupported_shapes():
    backend = _backend(4, _FakeFA4(), sm=100)
    valid = SimpleNamespace(
        is_mla=False,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        kv_cache_v_head_dim=128,
    )
    backend.validate_model_config(valid)

    with pytest.raises(ValueError, match="MLA"):
        backend.validate_model_config(SimpleNamespace(**{**vars(valid), "is_mla": True}))
    with pytest.raises(ValueError, match="GQA"):
        backend.validate_model_config(
            SimpleNamespace(
                **{
                    **vars(valid),
                    "num_attention_heads": 30,
                    "num_key_value_heads": 8,
                }
            )
        )
    with pytest.raises(ValueError, match="head_dim=256"):
        backend.validate_model_config(
            SimpleNamespace(
                **{
                    **vars(valid),
                    "head_dim": 256,
                    "kv_cache_v_head_dim": 256,
                }
            )
        )


@pytest.mark.parametrize(
    ("generation", "sm", "message"),
    [(3, 100, "SM100"), (4, 80, "SM80")],
)
def test_unsupported_generation_architecture_pairs_fail_closed(generation, sm, message):
    module = _FakeFA3() if generation == 3 else _FakeFA4()
    with pytest.raises(RuntimeError, match=message):
        _backend(generation, module, sm=sm)


def test_generation_and_injected_module_api_mismatches_are_actionable():
    with pytest.raises(ValueError, match="expected 3 or 4"):
        _backend(2, object(), sm=90)
    with pytest.raises(RuntimeError, match="flash_attn_varlen_func"):
        _backend(
            4,
            SimpleNamespace(flash_attn_func=lambda *args, **kwargs: None),
            sm=120,
        )


def test_fa3_default_loader_prefers_official_top_level_module(monkeypatch):
    imported: list[str] = []
    module = _FakeFA3()

    def import_module(name: str):
        imported.append(name)
        if name == "flash_attn_interface":
            return module
        raise AssertionError(f"unexpected compatibility import: {name}")

    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.importlib.import_module",
        import_module,
    )
    backend = FlashAttentionBackend(
        3,
        flashattention_module=None,
        decode_backend=_FakeDecodeBackend(),
        capability=90,
    )

    assert imported == ["flash_attn_interface"]
    assert backend.component_versions["prefill"] == "3.fake"


def test_missing_upstream_module_has_install_action(monkeypatch):
    def missing(_name: str):
        raise ModuleNotFoundError("not installed")

    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.importlib.import_module",
        missing,
    )
    with pytest.raises(RuntimeError, match="flash-attention/hopper"):
        FlashAttentionBackend(
            3,
            decode_backend=_FakeDecodeBackend(),
            capability=90,
        )
