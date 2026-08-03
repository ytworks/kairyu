from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from functools import cache
from types import SimpleNamespace

import pytest
import torch

from kairyu.engine.core.attention import graph_capture_gap
from kairyu.engine.core.attention.flashattention_gpu import (
    FlashAttentionBackend,
)
from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
from kairyu.engine.core.kv_pool import PagedKVPool

_FA3_BUILD_FLAGS = {
    "FLASHATTENTION_DISABLE_PAGEDKV": False,
    "FLASHATTENTION_DISABLE_VARLEN": False,
    "FLASHATTENTION_DISABLE_FP16": False,
    "FLASHATTENTION_DISABLE_HDIM64": False,
    "FLASHATTENTION_DISABLE_HDIM96": False,
    "FLASHATTENTION_DISABLE_HDIM128": False,
    "FLASHATTENTION_DISABLE_HDIM192": False,
    "FLASHATTENTION_DISABLE_HDIM256": False,
}


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
    def __init__(
        self,
        *,
        version: str = "3.0.0",
        build_flags: dict[str, bool] | None = None,
    ) -> None:
        self.__version__ = version
        self.CONFIG = {
            "build_flags": dict(_FA3_BUILD_FLAGS if build_flags is None else build_flags)
        }
        self.calls: list[dict[str, object]] = []

    def flash_attn_with_kvcache(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "query": q,
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
        return _bottom_right_attention(q, keys, values, causal=causal)


class _FakeFA4:
    def __init__(self, *, version: str = "4.0.0b24") -> None:
        self.__version__ = version
        self.paged_calls: list[dict[str, object]] = []
        self.dense_calls: list[dict[str, object]] = []

    def flash_attn_varlen_func(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        max_seqlen_q: int,
        max_seqlen_k: int,
        seqused_k: torch.Tensor,
        page_table: torch.Tensor,
        causal: bool,
    ) -> tuple[torch.Tensor, None]:
        self.paged_calls.append(
            {
                "query": q,
                "k_cache": k,
                "v_cache": v,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "seqused_k": seqused_k,
                "page_table": page_table,
                "causal": causal,
            }
        )
        length = int(seqused_k[0])
        pages = page_table[0].long()
        keys = k[pages].flatten(0, 1)[:length].unsqueeze(0)
        values = v[pages].flatten(0, 1)[:length].unsqueeze(0)
        return _bottom_right_attention(q, keys, values, causal=causal), None

    def flash_attn_func(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
    ) -> tuple[torch.Tensor, None]:
        self.dense_calls.append(
            {
                "query": q,
                "keys": k,
                "values": v,
                "causal": causal,
            }
        )
        return _bottom_right_attention(q, k, v, causal=causal), None


class _FakeCudaContexts:
    def __init__(self, capability_by_device: dict[torch.device, int]) -> None:
        self.capability_by_device = capability_by_device
        self.current: torch.device | None = None
        self.events: list[tuple[str, torch.device]] = []

    @contextmanager
    def device(self, device: object):
        selected = torch.device(device)
        previous = self.current
        self.current = selected
        self.events.append(("enter", selected))
        try:
            yield
        finally:
            self.events.append(("exit", selected))
            self.current = previous

    def get_device_capability(self, device: object) -> tuple[int, int]:
        selected = torch.device(device)
        assert self.current == selected
        return divmod(self.capability_by_device[selected], 10)


class _FakeFA4Interface:
    def __init__(
        self,
        contexts: _FakeCudaContexts,
        architecture_by_device: dict[torch.device, int],
    ) -> None:
        @cache
        def get_device_arch() -> int:
            assert contexts.current is not None
            return architecture_by_device[contexts.current]

        self._get_device_arch = get_device_arch

    @staticmethod
    def _parse_arch_str(value: str) -> int:
        normalised = value.lower()
        if normalised.startswith("sm_"):
            normalised = normalised[3:]
        if normalised.endswith("a"):
            normalised = normalised[:-1]
        return int(normalised)


class _FakeDecodeBackend:
    __version__ = "0.6.fake"
    supports_graph_capture = True

    def __init__(self) -> None:
        self.attend_calls: list[tuple] = []
        self.batch_calls: list[tuple] = []
        self.plan_calls: list[tuple] = []
        self.decode_calls: list[tuple] = []
        self.preflight_calls: list[tuple] = []

    def preflight_runtime(
        self,
        config: object,
        kv_pool: PagedKVPool,
        *,
        q_dtype: torch.dtype,
    ) -> None:
        self.preflight_calls.append((config, kv_pool, q_dtype))

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
        device="cpu",
        capability=sm,
    )


def test_fa3_sm90_paged_api_preserves_page_identity_gqa_and_causal_contract():
    pool, page_table, query = _case()
    module = _FakeFA3()
    backend = _backend(3, module, sm=90)

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


@pytest.mark.parametrize("sm", [90, 100, 110])
def test_fa4_sm90_sm100_and_sm110_use_direct_paged_api(sm):
    pool, page_table, query = _case()
    module = _FakeFA4()
    backend = _backend(4, module, sm=sm)

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


def test_fa4_sm120_materializes_selected_pages_on_device_for_dense_api():
    pool, page_table, query = _case()
    module = _FakeFA4()
    backend = _backend(4, module, sm=120)

    output = backend.attend(query, pool, 0, page_table, 5, 3)
    reference = TorchAttentionBackend().attend(query, pool, 0, page_table, 5, 3)

    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
    assert not module.paged_calls
    assert len(module.dense_calls) == 1
    call = module.dense_calls[0]
    expected_keys, expected_values = pool.gather(0, page_table, 5)
    torch.testing.assert_close(call["keys"], expected_keys.unsqueeze(0))
    torch.testing.assert_close(call["values"], expected_values.unsqueeze(0))
    assert call["keys"].device == pool.k.device
    assert call["values"].device == pool.v.device
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
    assert backend.supports_batched_prefill is False
    assert backend.device == torch.device("cpu")
    assert graph_capture_gap(backend) is None
    assert backend.components["decode"] == "flashinfer"
    assert not isinstance(backend, torch.nn.Module)


def test_runtime_preflight_reaches_the_delegated_flashinfer_backend():
    pool, _, query = _case()
    delegate = _FakeDecodeBackend()
    backend = _backend(4, _FakeFA4(), sm=120, decode=delegate)
    config = object()

    backend.preflight_runtime(config, pool, q_dtype=query.dtype)

    assert delegate.preflight_calls == [(config, pool, query.dtype)]


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
        dtype="bfloat16",
    )
    backend.validate_model_config(valid, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="MLA"):
        backend.validate_model_config(
            SimpleNamespace(**{**vars(valid), "is_mla": True}),
            dtype=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="GQA"):
        backend.validate_model_config(
            SimpleNamespace(
                **{
                    **vars(valid),
                    "num_attention_heads": 30,
                    "num_key_value_heads": 8,
                }
            ),
            dtype=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="head_dim=256"):
        backend.validate_model_config(
            SimpleNamespace(
                **{
                    **vars(valid),
                    "head_dim": 256,
                    "kv_cache_v_head_dim": 256,
                }
            ),
            dtype=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="fp16 or bf16"):
        backend.validate_model_config(valid, dtype=torch.float32)


@pytest.mark.parametrize(
    ("generation", "sm", "message"),
    [
        (3, 80, "SM80"),
        (3, 86, "SM86"),
        (3, 89, "SM89"),
        (3, 100, "SM100"),
        (4, 80, "SM80"),
    ],
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

    def dense(q, k, v, *, causal):
        return q

    def incomplete_paged(q, k, v, *, page_table, causal):
        return q

    with pytest.raises(RuntimeError, match="missing max_seqlen_q"):
        _backend(
            4,
            SimpleNamespace(
                __version__="4.0.0b24",
                flash_attn_func=dense,
                flash_attn_varlen_func=incomplete_paged,
            ),
            sm=100,
        )


def test_fa4_cuda_preflight_requires_pinned_private_arch_api():
    with pytest.raises(RuntimeError, match=r"interface\._get_device_arch"):
        FlashAttentionBackend(
            4,
            flashattention_module=_FakeFA4(),
            decode_backend=_FakeDecodeBackend(),
            device="cuda:0",
            capability=90,
        )


def test_fa4_cuda_preflight_rejects_environment_arch_mismatch(monkeypatch):
    architectures = {torch.device("cuda:1"): 90}
    contexts = _FakeCudaContexts(architectures)
    module = _FakeFA4()
    module.interface = _FakeFA4Interface(contexts, architectures)
    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.torch.cuda.device",
        contexts.device,
    )
    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.torch.cuda.get_device_capability",
        contexts.get_device_capability,
    )
    monkeypatch.setenv("FLASH_ATTENTION_ARCH", "sm_120")

    with pytest.raises(RuntimeError, match=r"FLASH_ATTENTION_ARCH='sm_120'.*SM120.*SM90"):
        FlashAttentionBackend(
            4,
            flashattention_module=module,
            decode_backend=_FakeDecodeBackend(),
            device="cuda:1",
            capability=90,
        )


def test_fa4_cuda_preflight_rejects_selected_device_capability_mismatch(
    monkeypatch,
):
    device = torch.device("cuda:1")
    contexts = _FakeCudaContexts({device: 100})
    module = _FakeFA4()
    module.interface = _FakeFA4Interface(contexts, {device: 90})
    with contexts.device(device):
        assert module.interface._get_device_arch() == 90
    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.torch.cuda.device",
        contexts.device,
    )
    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.torch.cuda.get_device_capability",
        contexts.get_device_capability,
    )
    monkeypatch.setenv("FLASH_ATTENTION_ARCH", "sm_90")

    with pytest.raises(
        RuntimeError,
        match=r"selected device cuda:1 reports SM100.*requires SM90",
    ):
        FlashAttentionBackend(
            4,
            flashattention_module=module,
            decode_backend=_FakeDecodeBackend(),
            device=device,
            capability=90,
        )


def test_fa4_global_arch_cache_rejects_second_heterogeneous_backend(monkeypatch):
    architectures = {
        torch.device("cuda:0"): 90,
        torch.device("cuda:1"): 100,
    }
    contexts = _FakeCudaContexts(architectures)
    module = _FakeFA4()
    module.interface = _FakeFA4Interface(contexts, architectures)
    monkeypatch.delenv("FLASH_ATTENTION_ARCH", raising=False)
    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.torch.cuda.device",
        contexts.device,
    )
    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.torch.cuda.get_device_capability",
        contexts.get_device_capability,
    )

    FlashAttentionBackend(
        4,
        flashattention_module=module,
        decode_backend=_FakeDecodeBackend(),
        device="cuda:0",
        capability=90,
    )
    with pytest.raises(
        RuntimeError,
        match=r"pre-existing global _get_device_arch cache.*SM90.*SM100",
    ):
        FlashAttentionBackend(
            4,
            flashattention_module=module,
            decode_backend=_FakeDecodeBackend(),
            device="cuda:1",
            capability=100,
        )


def test_fa4_runtime_call_uses_selected_cuda_device_context(monkeypatch):
    architectures = {torch.device("cuda:1"): 120}
    contexts = _FakeCudaContexts(architectures)
    module = _FakeFA4()
    module.interface = _FakeFA4Interface(contexts, architectures)
    call_devices: list[torch.device | None] = []
    dense = module.flash_attn_func

    def recording_dense(q, k, v, *, causal):
        call_devices.append(contexts.current)
        return dense(q, k, v, causal=causal)

    module.flash_attn_func = recording_dense
    monkeypatch.delenv("FLASH_ATTENTION_ARCH", raising=False)
    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.torch.cuda.device",
        contexts.device,
    )
    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.torch.cuda.get_device_capability",
        contexts.get_device_capability,
    )
    backend = FlashAttentionBackend(
        4,
        flashattention_module=module,
        decode_backend=_FakeDecodeBackend(),
        device="cuda:1",
        capability=120,
    )
    contexts.events.clear()
    pool, page_table, query = _case()

    backend.attend(query, pool, 0, page_table, 5, 3)

    assert call_devices == [torch.device("cuda:1")]
    assert contexts.events == [
        ("enter", torch.device("cuda:1")),
        ("exit", torch.device("cuda:1")),
    ]


@pytest.mark.parametrize(
    ("generation", "version", "sm", "expected"),
    [
        (3, "3.1.0", 90, "expected 3.0.0"),
        (4, "4.0.0b23", 100, "expected 4.0.0b24"),
    ],
)
def test_upstream_version_preflight_is_fail_closed(generation, version, sm, expected):
    module = _FakeFA3(version=version) if generation == 3 else _FakeFA4(version=version)
    with pytest.raises(RuntimeError, match=expected):
        _backend(generation, module, sm=sm)


def test_fa4_beta_wrapper_uses_distribution_version_for_preflight(monkeypatch):
    requested: list[str] = []

    def distribution_version(name: str) -> str:
        requested.append(name)
        return "4.0.0b24"

    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.metadata.version",
        distribution_version,
    )
    backend = _backend(4, _FakeFA4(version="0.0.0"), sm=120)

    assert requested == ["flash-attn-4"]
    assert backend.component_versions["prefill"] == "4.0.0b24"


def test_fa3_build_flags_reject_missing_paged_kv_and_disabled_shapes():
    no_paged = {**_FA3_BUILD_FLAGS, "FLASHATTENTION_DISABLE_PAGEDKV": True}
    with pytest.raises(RuntimeError, match="DISABLE_PAGEDKV=True"):
        _backend(3, _FakeFA3(build_flags=no_paged), sm=90)

    no_varlen = {**_FA3_BUILD_FLAGS, "FLASHATTENTION_DISABLE_VARLEN": True}
    with pytest.raises(RuntimeError, match="DISABLE_VARLEN=True"):
        _backend(3, _FakeFA3(build_flags=no_varlen), sm=90)

    missing_varlen = dict(_FA3_BUILD_FLAGS)
    del missing_varlen["FLASHATTENTION_DISABLE_VARLEN"]
    with pytest.raises(RuntimeError, match="missing FLASHATTENTION_DISABLE_VARLEN"):
        _backend(3, _FakeFA3(build_flags=missing_varlen), sm=90)

    non_boolean_varlen = {
        **_FA3_BUILD_FLAGS,
        "FLASHATTENTION_DISABLE_VARLEN": 0,
    }
    with pytest.raises(RuntimeError, match="non-boolean FLASHATTENTION_DISABLE_VARLEN"):
        _backend(3, _FakeFA3(build_flags=non_boolean_varlen), sm=90)

    missing_flag = dict(_FA3_BUILD_FLAGS)
    del missing_flag["FLASHATTENTION_DISABLE_HDIM128"]
    with pytest.raises(RuntimeError, match="missing FLASHATTENTION_DISABLE_HDIM128"):
        _backend(3, _FakeFA3(build_flags=missing_flag), sm=90)

    limited_dims = {
        **_FA3_BUILD_FLAGS,
        "FLASHATTENTION_DISABLE_HDIM128": True,
        "FLASHATTENTION_DISABLE_HDIM192": True,
        "FLASHATTENTION_DISABLE_HDIM256": True,
        "FLASHATTENTION_DISABLE_FP16": True,
    }
    backend = _backend(3, _FakeFA3(build_flags=limited_dims), sm=90)
    config = SimpleNamespace(
        is_mla=False,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=96,
        kv_cache_v_head_dim=96,
    )
    backend.validate_model_config(config, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim=128"):
        backend.validate_model_config(
            SimpleNamespace(
                **{
                    **vars(config),
                    "head_dim": 128,
                    "kv_cache_v_head_dim": 128,
                }
            ),
            dtype=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="DISABLE_FP16=True"):
        backend.validate_model_config(config, dtype=torch.float16)


def test_fa3_default_loader_reads_official_sibling_build_config(monkeypatch):
    imported: list[str] = []
    distributions: list[str] = []
    module = _FakeFA3()
    del module.CONFIG
    del module.__version__
    config_module = SimpleNamespace(CONFIG={"build_flags": dict(_FA3_BUILD_FLAGS)})

    def import_module(name: str):
        imported.append(name)
        if name == "flash_attn_interface":
            return module
        if name == "flash_attn_config":
            return config_module
        raise AssertionError(f"unexpected compatibility import: {name}")

    def distribution_version(name: str) -> str:
        distributions.append(name)
        return "3.0.0+cu13torch2.12"

    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.metadata.version",
        distribution_version,
    )
    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.importlib.import_module",
        import_module,
    )
    backend = FlashAttentionBackend(
        3,
        flashattention_module=None,
        decode_backend=_FakeDecodeBackend(),
        device="cpu",
        capability=90,
    )

    assert imported == ["flash_attn_interface", "flash_attn_config"]
    assert distributions == ["flash_attn_3"]
    assert backend.component_versions["prefill"] == "3.0.0+cu13torch2.12"


@pytest.mark.parametrize(
    ("generation", "sm", "action"),
    [
        (3, 90, "flash-attention/hopper"),
        (4, 120, "uv sync --extra flashattention4"),
    ],
)
def test_missing_upstream_module_has_install_action(monkeypatch, generation, sm, action):
    def missing(_name: str):
        raise ModuleNotFoundError("not installed")

    monkeypatch.setattr(
        "kairyu.engine.core.attention.flashattention_gpu.importlib.import_module",
        missing,
    )
    with pytest.raises(RuntimeError, match=action):
        FlashAttentionBackend(
            generation,
            decode_backend=_FakeDecodeBackend(),
            capability=sm,
        )


def test_fa4_with_upstream_present_but_missing_decode_dependency_is_actionable(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "flashinfer", None)

    with pytest.raises(RuntimeError, match=r"uv sync --extra gpu"):
        FlashAttentionBackend(
            4,
            flashattention_module=_FakeFA4(),
            decode_backend=None,
            device="cpu",
            capability=120,
        )
