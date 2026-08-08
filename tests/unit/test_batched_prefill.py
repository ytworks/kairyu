"""Issue #224: ragged prefill is a correctness-preserving batch transform."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kairyu.engine.core.attention import TorchAttentionBackend
from kairyu.engine.core.engine_core import token_ids
from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.model_runner import PagedModelRunner
from kairyu.engine.core.prefill import (
    PrefillSequence,
    build_prefill_batch,
)
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import EngineSampling
from kairyu.engine.core.scheduler import (
    EngineRequest,
    ScheduledChunk,
    Scheduler,
)
from kairyu.models.config import parse_model_config
from kairyu.models.llama import DenseDecoder

PAGE = 4
TINY = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 64,
    "vocab_size": 64,
    "rms_norm_eps": 1e-6,
}


class _NativeTorchBackend(TorchAttentionBackend):
    """CPU oracle for the native flat backend contract.

    The production Torch backend deliberately does not opt in because this
    reference loops over rows.  Here it lets model/runner contract tests execute
    without faking the attention result.
    """

    supports_batched_prefill = True

    def __init__(self) -> None:
        self.prefill_calls = 0

    def attend_prefill(
        self,
        query,
        kv_pool,
        layer,
        page_tables,
        seq_lens,
        chunk_starts,
        qo_indptr,
    ):
        self.prefill_calls += 1
        queries = [query[qo_indptr[index] : qo_indptr[index + 1]] for index in range(len(seq_lens))]
        contexts = [
            self.attend(
                row,
                kv_pool,
                layer,
                list(page_tables[index]),
                seq_lens[index],
                chunk_starts[index],
            )
            for index, row in enumerate(queries)
        ]
        return torch.cat(contexts)


def _model(backend=None) -> DenseDecoder:
    torch.manual_seed(224)
    return DenseDecoder(parse_model_config(TINY), attention_backend=backend).eval()


def _pool(model: DenseDecoder) -> PagedKVPool:
    return PagedKVPool.for_cache(RadixKVCache(num_pages=12, page_size=PAGE), model.config)


def _row(
    request_id: str,
    tokens,
    pages,
    *,
    start: int,
    seq_len: int,
    write_from: int,
) -> PrefillSequence:
    return PrefillSequence(
        request_id=request_id,
        token_ids=tuple(tokens),
        page_table=tuple(pages),
        chunk_start=start,
        seq_len=seq_len,
        write_from=write_from,
    )


class _CountingPageTable(tuple):
    integer_reads: int

    def __new__(cls, pages):
        instance = super().__new__(cls, pages)
        instance.integer_reads = 0
        return instance

    def __getitem__(self, index):
        if isinstance(index, int):
            self.integer_reads += 1
        return super().__getitem__(index)


def test_prefill_plan_preserves_row_metadata_and_shared_cached_page():
    batch = build_prefill_batch(
        [
            _row("a", [5, 6, 7], [0, 1], start=4, seq_len=7, write_from=4),
            _row("cached", [4], [0], start=3, seq_len=4, write_from=4),
            _row("cold", [8, 9, 10, 11, 12], [2, 3], start=0, seq_len=5, write_from=0),
        ],
        page_size=PAGE,
    )

    assert batch.request_ids == ("a", "cached", "cold")
    assert batch.token_ids.tolist() == [5, 6, 7, 4, 8, 9, 10, 11, 12]
    assert batch.positions.tolist() == [4, 5, 6, 3, 0, 1, 2, 3, 4]
    assert batch.row_ids.tolist() == [0, 0, 0, 1, 2, 2, 2, 2, 2]
    assert batch.query_lens == (3, 1, 5)
    assert batch.qo_indptr == (0, 3, 4, 9)
    assert batch.seq_lens == (7, 4, 5)
    assert batch.chunk_starts == (4, 3, 0)
    assert batch.write_from.tolist() == [4, 4, 0]
    assert batch.page_lists == ((0, 1), (0,), (2, 3))
    assert batch.token_ids.dtype is torch.int64
    assert batch.positions.dtype is torch.int64
    assert batch.row_ids.dtype is torch.int64
    assert batch.write_from.dtype is torch.int64
    assert batch.page_tables.dtype is torch.int32
    assert len(
        {
            tensor.untyped_storage().data_ptr()
            for tensor in (
                batch.token_ids,
                batch.positions,
                batch.row_ids,
                batch.write_from,
                batch.page_tables,
            )
        }
    ) == 1


def test_prefill_plan_rejects_page_alias_and_cross_request_writable_page():
    with pytest.raises(ValueError, match="aliases one physical page"):
        build_prefill_batch(
            [_row("alias", [5], [0, 0], start=4, seq_len=5, write_from=4)],
            page_size=PAGE,
        )

    # Both rows write different slots of physical page 1. Slot-only collision
    # detection would miss this, but either sequence could then read the
    # other's KV through that shared page.
    with pytest.raises(ValueError, match="writable pages.*overlap"):
        build_prefill_batch(
            [
                _row("a", [5, 6], [0, 1], start=4, seq_len=6, write_from=4),
                _row("b", [7], [2, 1], start=6, seq_len=7, write_from=6),
            ],
            page_size=PAGE,
        )

    with pytest.raises(RuntimeError, match="overflow"):
        build_prefill_batch(
            [_row("overflow", [5], [2**31], start=0, seq_len=1, write_from=0)],
            page_size=PAGE,
        )


@pytest.mark.parametrize("reverse", [False, True], ids=["reader-first", "writer-first"])
def test_prefill_plan_rejects_reader_writer_page_sharing_in_either_order(
    reverse: bool,
):
    reader = _row("reader", [4], [1], start=3, seq_len=4, write_from=4)
    writer = _row("writer", [5], [1], start=0, seq_len=1, write_from=0)
    rows = [writer, reader] if reverse else [reader, writer]

    with pytest.raises(ValueError, match="writable pages.*overlap"):
        build_prefill_batch(rows, page_size=PAGE)


def test_prefill_plan_allows_shared_read_only_partial_page_and_unused_aliases():
    shared = build_prefill_batch(
        [
            _row("a", [3], [0], start=2, seq_len=3, write_from=3),
            _row("b", [3], [0], start=2, seq_len=3, write_from=3),
        ],
        page_size=PAGE,
    )
    assert shared.page_lists == ((0,), (0,))

    trailing = build_prefill_batch(
        [_row("tail", [5], [0, 1, 0], start=4, seq_len=5, write_from=4)],
        page_size=PAGE,
    )
    assert trailing.page_lists == ((0, 1, 0),)


def test_prefill_ownership_page_lookups_do_not_scale_with_prompt_tokens():
    def integer_reads(*, tokens: int, page_size: int) -> int:
        pages = _CountingPageTable(range(64))
        row = PrefillSequence(
            request_id=str(tokens),
            token_ids=tuple(range(tokens)),
            page_table=pages,
            chunk_start=0,
            seq_len=tokens,
            write_from=0,
        )
        build_prefill_batch([row], page_size=page_size)
        return pages.integer_reads

    short = integer_reads(tokens=1024, page_size=16)
    long = integer_reads(tokens=8192, page_size=128)

    assert short == long
    assert long <= 64


def test_batched_prefill_matches_sequential_and_retains_cached_kv():
    backend = _NativeTorchBackend()
    model = _model(backend)
    sequential_pool = _pool(model)
    batched_pool = _pool(model)

    prefix = torch.tensor([1, 2, 3, 4])
    for pool in (sequential_pool, batched_pool):
        model.forward_tokens(
            prefix,
            torch.arange(4),
            pool,
            [0],
            seq_len=4,
            write_from=0,
        )
        torch.manual_seed(99)
        pool.k[:, 9].copy_(torch.randn_like(pool.k[:, 9]))
        pool.v[:, 9].copy_(torch.randn_like(pool.v[:, 9]))
    cached_k = batched_pool.k[:, 0].clone()
    cached_v = batched_pool.v[:, 0].clone()
    poison_k = batched_pool.k[:, 9].clone()
    poison_v = batched_pool.v[:, 9].clone()

    rows = [
        _row("a", [5, 6, 7], [0, 1], start=4, seq_len=7, write_from=4),
        _row("cached", [4], [0], start=3, seq_len=4, write_from=4),
        _row("cold", [8, 9, 10, 11, 12], [2, 3], start=0, seq_len=5, write_from=0),
    ]
    sequential = [
        model.forward_tokens(
            torch.tensor(row.token_ids),
            torch.arange(row.chunk_start, row.seq_len),
            sequential_pool,
            list(row.page_table),
            seq_len=row.seq_len,
            write_from=row.write_from,
        )
        for row in rows
    ]

    backend.prefill_calls = 0
    batch = build_prefill_batch(rows, page_size=PAGE)
    actual = batch.split(model.forward_prefill_batch(batch, batched_pool))

    assert backend.prefill_calls == model.config.num_hidden_layers
    for got, expected in zip(actual, sequential, strict=True):
        assert torch.allclose(got, expected, atol=1e-5)
    assert torch.allclose(batched_pool.k, sequential_pool.k, atol=1e-6)
    assert torch.allclose(batched_pool.v, sequential_pool.v, atol=1e-6)
    assert torch.equal(batched_pool.k[:, 0], cached_k)
    assert torch.equal(batched_pool.v[:, 0], cached_v)
    assert torch.equal(batched_pool.k[:, 9], poison_k)
    assert torch.equal(batched_pool.v[:, 9], poison_v)


def _state(
    request_id: str,
    prompt: tuple[int, ...],
    pages: tuple[int, ...],
    *,
    computed: int | None = None,
    cached: int = 0,
    decode_pages: tuple[int, ...] = (),
    outputs: tuple[int, ...] = (),
):
    end = len(prompt) if computed is None else computed
    return SimpleNamespace(
        request=SimpleNamespace(
            request_id=request_id,
            sampling_identity=request_id,
            prompt_token_ids=prompt,
            sampling=EngineSampling(),
            eos_token_id=None,
        ),
        allocation=SimpleNamespace(pages=pages, num_cached_tokens=cached),
        decode_pages=list(decode_pages),
        computed_prompt=end,
        prefill_done=end >= len(prompt),
        outputs=list(outputs),
        output_epoch=0,
    )


def test_runner_batches_multiple_prefills_and_emits_only_terminal_rows():
    backend = _NativeTorchBackend()
    model = _model(backend)
    runner = PagedModelRunner(model, _pool(model))
    chunks = (
        ScheduledChunk("partial", 2, True),
        ScheduledChunk("done", 3, True),
    )
    states = {
        "partial": _state("partial", (1, 2, 3, 4), (0,), computed=2),
        "done": _state("done", (5, 6, 7), (1,)),
    }

    result = runner.execute(chunks, states)

    assert backend.prefill_calls == model.config.num_hidden_layers
    assert set(result) == {"done"}
    assert len(result["done"]) == 1
    assert runner.prefill_execution_stats() == {
        "enabled": True,
        "capability_gap": None,
        "rows": 2,
        "model_calls": 1,
        "batched_groups": 1,
        "sequential_rows": 0,
        "backend": [],
    }


def test_runner_unifies_mixed_prefill_and_decode_with_numeric_parity(monkeypatch):
    mixed_backend = _NativeTorchBackend()
    split_backend = _NativeTorchBackend()
    mixed_model = _model(mixed_backend)
    split_model = _model(split_backend)
    mixed_pool = _pool(mixed_model)
    split_pool = _pool(split_model)
    decode_prompt = (11, 12, 13, 14)
    for model, pool in ((mixed_model, mixed_pool), (split_model, split_pool)):
        model.forward_tokens(
            torch.tensor(decode_prompt),
            torch.arange(len(decode_prompt)),
            pool,
            [0],
            seq_len=len(decode_prompt),
        )

    mixed_runner = PagedModelRunner(mixed_model, mixed_pool)
    split_runner = PagedModelRunner(split_model, split_pool)
    split_runner.set_batched_prefill_enabled(False)
    mixed_calls = 0
    split_calls = 0
    mixed_forward = mixed_model.forward_prefill_batch
    split_forward = split_model.forward_tokens

    def counted_mixed(*args, **kwargs):
        nonlocal mixed_calls
        mixed_calls += 1
        return mixed_forward(*args, **kwargs)

    def counted_split(*args, **kwargs):
        nonlocal split_calls
        split_calls += 1
        return split_forward(*args, **kwargs)

    monkeypatch.setattr(mixed_model, "forward_prefill_batch", counted_mixed)
    monkeypatch.setattr(split_model, "forward_tokens", counted_split)
    chunks = (
        ScheduledChunk("prefill", 3, True),
        ScheduledChunk("decode", 1, False, position=1),
    )

    def states():
        return {
            "prefill": _state("prefill", (1, 2, 3), (2,)),
            "decode": _state(
                "decode",
                decode_prompt,
                (0,),
                decode_pages=(1,),
                outputs=(9,),
            ),
        }

    mixed = mixed_runner.execute(chunks, states())
    split = split_runner.execute(chunks, states())

    assert mixed_calls == 1
    assert split_calls == 2
    assert {
        request_id: tuple(token.token_id for token in tokens)
        for request_id, tokens in mixed.items()
    } == {
        request_id: tuple(token.token_id for token in tokens)
        for request_id, tokens in split.items()
    }
    assert torch.allclose(mixed_pool.k, split_pool.k, atol=1e-6)
    assert torch.allclose(mixed_pool.v, split_pool.v, atol=1e-6)
    assert mixed_runner.prefill_execution_stats() == {
        "enabled": True,
        "capability_gap": None,
        "rows": 1,
        "model_calls": 1,
        "batched_groups": 1,
        "sequential_rows": 0,
        "backend": [],
    }


def test_mixed_batch_handles_multiple_rows_partial_prefill_and_cached_prefix():
    mixed_model = _model(_NativeTorchBackend())
    split_model = _model(_NativeTorchBackend())
    mixed_pool = _pool(mixed_model)
    split_pool = _pool(split_model)
    shared_prompt = (11, 12, 13, 14)
    second_decode_prompt = (21, 22, 23, 24)
    for model, pool in ((mixed_model, mixed_pool), (split_model, split_pool)):
        for prompt, page in ((shared_prompt, 4), (second_decode_prompt, 7)):
            model.forward_tokens(
                torch.tensor(prompt),
                torch.arange(len(prompt)),
                pool,
                [page],
                seq_len=len(prompt),
            )

    cached_k = mixed_pool.k[:, 4].clone()
    cached_v = mixed_pool.v[:, 4].clone()
    mixed_runner = PagedModelRunner(mixed_model, mixed_pool)
    split_runner = PagedModelRunner(split_model, split_pool)
    split_runner.set_batched_prefill_enabled(False)
    chunks = (
        ScheduledChunk("partial", 2, True),
        ScheduledChunk("cached", 1, True),
        ScheduledChunk("decode-a", 1, False, position=1),
        ScheduledChunk("decode-b", 1, False, position=1),
    )

    def states():
        return {
            "partial": _state(
                "partial", (31, 32, 33, 34), (6,), computed=2
            ),
            "cached": _state(
                "cached", shared_prompt, (4,), cached=len(shared_prompt)
            ),
            "decode-a": _state(
                "decode-a",
                shared_prompt,
                (4,),
                decode_pages=(5,),
                outputs=(9,),
            ),
            "decode-b": _state(
                "decode-b",
                second_decode_prompt,
                (7,),
                decode_pages=(8,),
                outputs=(10,),
            ),
        }

    mixed = mixed_runner.execute(chunks, states())
    split = split_runner.execute(chunks, states())

    assert set(mixed) == {"cached", "decode-a", "decode-b"}
    assert {
        request_id: tuple(token.token_id for token in tokens)
        for request_id, tokens in mixed.items()
    } == {
        request_id: tuple(token.token_id for token in tokens)
        for request_id, tokens in split.items()
    }
    assert torch.allclose(mixed_pool.k, split_pool.k, atol=1e-6)
    assert torch.allclose(mixed_pool.v, split_pool.v, atol=1e-6)
    assert torch.equal(mixed_pool.k[:, 4], cached_k)
    assert torch.equal(mixed_pool.v[:, 4], cached_v)
    assert mixed_runner.prefill_execution_stats()["model_calls"] == 1
    assert split_runner.prefill_execution_stats()["model_calls"] == 2


def test_mixed_step_falls_back_when_ragged_prefill_is_unsupported(monkeypatch):
    model = _model()
    pool = _pool(model)
    decode_prompt = (11, 12, 13, 14)
    model.forward_tokens(
        torch.tensor(decode_prompt),
        torch.arange(len(decode_prompt)),
        pool,
        [0],
        seq_len=len(decode_prompt),
    )
    runner = PagedModelRunner(model, pool)
    calls = 0
    forward = model.forward_tokens

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward_tokens", counted)
    result = runner.execute(
        (
            ScheduledChunk("prefill", 3, True),
            ScheduledChunk("decode", 1, False, position=1),
        ),
        {
            "prefill": _state("prefill", (1, 2, 3), (2,)),
            "decode": _state(
                "decode",
                decode_prompt,
                (0,),
                decode_pages=(1,),
                outputs=(9,),
            ),
        },
    )

    assert set(result) == {"prefill", "decode"}
    assert calls == 2
    assert "does not declare supports_batched_prefill" in runner._prefill_batch_gap


def test_unified_mixed_disable_keeps_pure_prefill_batching(monkeypatch):
    model = _model(_NativeTorchBackend())
    pool = _pool(model)
    decode_prompt = (11, 12, 13, 14)
    model.forward_tokens(
        torch.tensor(decode_prompt),
        torch.arange(len(decode_prompt)),
        pool,
        [0],
        seq_len=len(decode_prompt),
    )
    runner = PagedModelRunner(model, pool)
    runner.set_unified_mixed_enabled(False)
    ragged_calls = 0
    token_calls = 0
    ragged_forward = model.forward_prefill_batch
    token_forward = model.forward_tokens

    def counted_ragged(*args, **kwargs):
        nonlocal ragged_calls
        ragged_calls += 1
        return ragged_forward(*args, **kwargs)

    def counted_tokens(*args, **kwargs):
        nonlocal token_calls
        token_calls += 1
        return token_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward_prefill_batch", counted_ragged)
    monkeypatch.setattr(model, "forward_tokens", counted_tokens)
    mixed = runner.execute(
        (
            ScheduledChunk("prefill", 3, True),
            ScheduledChunk("decode", 1, False, position=1),
        ),
        {
            "prefill": _state("prefill", (1, 2, 3), (2,)),
            "decode": _state(
                "decode",
                decode_prompt,
                (0,),
                decode_pages=(1,),
                outputs=(9,),
            ),
        },
    )
    pure_prefill = runner.execute(
        (
            ScheduledChunk("a", 2, True),
            ScheduledChunk("b", 2, True),
        ),
        {
            "a": _state("a", (5, 6), (3,)),
            "b": _state("b", (7, 8), (4,)),
        },
    )

    assert set(mixed) == {"prefill", "decode"}
    assert set(pure_prefill) == {"a", "b"}
    assert token_calls == 2
    assert ragged_calls == 1
    assert runner.unified_mixed_enabled is False


def test_runner_falls_back_for_torch_backend_and_single_request():
    model = _model()
    runner = PagedModelRunner(model, _pool(model))
    assert "does not declare supports_batched_prefill" in runner._prefill_batch_gap
    states = {
        "a": _state("a", (1, 2), (0,)),
        "b": _state("b", (3, 4), (1,)),
    }
    result = runner.execute(
        (
            ScheduledChunk("a", 2, True),
            ScheduledChunk("b", 2, True),
        ),
        states,
    )
    assert set(result) == {"a", "b"}
    stats = runner.prefill_execution_stats(reset=True)
    assert stats["rows"] == stats["model_calls"] == 2
    assert stats["sequential_rows"] == 2
    assert runner.prefill_execution_stats()["rows"] == 0

    native = _NativeTorchBackend()
    native_model = _model(native)
    native_runner = PagedModelRunner(native_model, _pool(native_model))
    result = native_runner.execute(
        (ScheduledChunk("a", 2, True),),
        {"a": _state("a", (1, 2), (0,))},
    )
    assert set(result) == {"a"}
    assert native.prefill_calls == 0


def test_runner_passes_contiguous_host_metadata_to_opted_in_model(monkeypatch):
    model = _model()
    runner = PagedModelRunner(model, _pool(model))
    original = model.forward_tokens
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward_tokens", capture)
    result = runner.execute(
        (ScheduledChunk("single", 2, True),),
        {"single": _state("single", (1, 2), (0,))},
    )

    assert set(result) == {"single"}
    assert len(calls) == 1
    assert calls[0]["chunk_start"] == 0
    assert calls[0]["has_writable"] is True


def test_arbitrary_position_fallback_preserves_the_m12_write_contract():
    model = _model()
    pool = _pool(model)
    pages = [0, 1]
    model.forward_tokens(
        torch.tensor([1, 2, 3, 4, 5]),
        torch.arange(5),
        pool,
        pages,
        seq_len=5,
    )
    snapshot = pool.k.clone()

    model.forward_tokens(
        torch.tensor([8, 9]),
        torch.tensor([2, 4]),
        pool,
        pages,
        seq_len=5,
        write_from=3,
    )

    assert torch.equal(pool.k[:, 0, 2], snapshot[:, 0, 2])
    assert not torch.equal(pool.k[:, 1, 0], snapshot[:, 1, 0])


def test_host_prefill_metadata_avoids_tensor_scalar_reads(monkeypatch):
    model = _model()
    pool = _pool(model)

    def fail(name):
        def method(self, *args, **kwargs):
            raise AssertionError(f"host tensor scalar read through {name}")

        return method

    monkeypatch.setattr(torch.Tensor, "any", fail("any"))
    monkeypatch.setattr(torch.Tensor, "item", fail("item"))
    monkeypatch.setattr(torch.Tensor, "__bool__", fail("bool"))

    model.forward_tokens(
        torch.tensor([1, 2]),
        torch.arange(2),
        pool,
        [0],
        seq_len=2,
        write_from=0,
        chunk_start=0,
        has_writable=True,
    )
    snapshot = pool.k.clone()
    model.forward_tokens(
        torch.tensor([3, 4]),
        torch.arange(2),
        pool,
        [0],
        seq_len=2,
        write_from=1,
        chunk_start=0,
        has_writable=True,
    )

    assert torch.equal(pool.k[:, 0, 0], snapshot[:, 0, 0])
    assert not torch.equal(pool.k[:, 0, 1], snapshot[:, 0, 1])

    snapshot = pool.k.clone()
    model.forward_tokens(
        torch.tensor([5, 6]),
        torch.arange(2),
        pool,
        [0],
        seq_len=2,
        write_from=2,
        chunk_start=0,
        has_writable=False,
    )

    assert torch.equal(pool.k, snapshot)

    model.forward_tokens_with_aux(
        torch.tensor([7, 8]),
        torch.arange(2),
        pool,
        [0],
        seq_len=2,
        aux_layer_ids=(0,),
        write_from=2,
        chunk_start=0,
        has_writable=False,
    )
    assert torch.equal(pool.k, snapshot)


def _run_scheduler_preemption_case(*, batched: bool):
    backend = _NativeTorchBackend()
    model = _model(backend)
    cache = RadixKVCache(num_pages=4, page_size=PAGE)
    pool = PagedKVPool.for_cache(cache, model.config)
    runner = PagedModelRunner(model, pool, cache=cache)
    runner.set_batched_prefill_enabled(batched)
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=8,
        max_num_seqs=2,
        priority_age_s=0.0,
    )

    def execute_step():
        plan = scheduler.schedule()
        sampled = runner.execute(plan.scheduled, scheduler.states)
        if sampled:
            scheduler.update(token_ids(sampled))
        return plan

    scheduler.add_request(
        EngineRequest(
            "victim",
            tuple(range(1, 13)),
            max_new_tokens=1,
            priority=10,
        )
    )
    first = execute_step()
    assert [(chunk.request_id, chunk.num_tokens) for chunk in first.scheduled] == [("victim", 8)]
    first_allocation = scheduler.states["victim"].allocation
    assert first_allocation is not None
    first_pages = first_allocation.pages

    # Only one of four pages is free. The two-page high-priority prompt must
    # therefore release and evict the incomplete victim's three-page allocation.
    scheduler.add_request(
        EngineRequest(
            "urgent",
            tuple(range(21, 29)),
            max_new_tokens=1,
            priority=0,
        )
    )
    urgent = execute_step()
    assert [(chunk.request_id, chunk.num_tokens) for chunk in urgent.scheduled] == [("urgent", 8)]
    assert scheduler.states["victim"].allocation is None
    assert scheduler.states["victim"].computed_prompt == 0
    assert scheduler.waiting_ids == ("victim",)
    urgent_allocation = scheduler.states["urgent"].allocation
    assert urgent_allocation is not None
    assert set(urgent_allocation.pages) & set(first_pages)

    # The victim resumes from zero under a new allocation and immediately forms
    # a work-conserving prefill cohort with the newly admitted partner.
    scheduler.add_request(
        EngineRequest(
            "partner",
            tuple(range(41, 45)),
            max_new_tokens=1,
            priority=10,
        )
    )
    resumed = execute_step()
    assert [(chunk.request_id, chunk.num_tokens) for chunk in resumed.scheduled] == [
        ("victim", 4),
        ("partner", 4),
    ]
    resumed_allocation = scheduler.states["victim"].allocation
    assert resumed_allocation is not None
    assert resumed_allocation is not first_allocation

    completed = execute_step()
    assert [
        (chunk.request_id, chunk.num_tokens, chunk.is_prefill) for chunk in completed.scheduled
    ] == [
        ("victim", 8, True),
    ]
    outputs = {
        request_id: scheduler.output_tokens(request_id)
        for request_id in ("victim", "urgent", "partner")
    }
    stats = runner.prefill_execution_stats()
    for request_id in outputs:
        runner.release(request_id)
    return outputs, pool.k.clone(), pool.v.clone(), stats


def test_scheduler_preemption_resume_matches_sequential_prefill():
    sequential = _run_scheduler_preemption_case(batched=False)
    candidate = _run_scheduler_preemption_case(batched=True)

    assert candidate[0] == sequential[0]
    assert all(len(tokens) == 1 for tokens in candidate[0].values())
    assert torch.allclose(candidate[1], sequential[1], atol=1e-6)
    assert torch.allclose(candidate[2], sequential[2], atol=1e-6)
    assert sequential[3]["batched_groups"] == 0
    assert sequential[3]["model_calls"] == 5
    assert candidate[3]["batched_groups"] == 1
    assert candidate[3]["model_calls"] == 4
