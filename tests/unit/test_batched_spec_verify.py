"""Issue #215: target verification is one decode-shaped model invocation.

The production transform flattens every target position in every speculative
chunk into decode rows.  These tests deliberately exercise ``PagedModelRunner``
directly: ``SpeculativeRunner`` owns acceptance policy, while this layer must
return every target position, write the same logical KV as sequential scoring,
and route the flattened row count through the existing graph buckets.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

# Reuse the issue #224 tiny native model/state fixtures instead of carrying a
# second model definition in this contract suite.
from test_batched_prefill import PAGE, _model, _pool, _state

from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.model_runner import PagedModelRunner
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import EngineSampling
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.step_executor import SnapshotGraphBackend

TINY_MLA = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "moe_intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "kv_lora_rank": 8,
    "q_lora_rank": 8,
    "qk_nope_head_dim": 4,
    "qk_rope_head_dim": 4,
    "v_head_dim": 4,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
    "first_k_dense_replace": 1,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 128,
}


def _overlay_state(
    request_id: str,
    prompt: tuple[int, ...],
    pages: tuple[int, ...],
    outputs: tuple[int, ...],
    *,
    override: bool = True,
):
    state = _state(
        request_id,
        prompt,
        pages,
        cached=len(prompt),
    )
    state.outputs = list(outputs)
    state.outputs_override = override
    return state


def _prime_prompt(model, pool, prompt: tuple[int, ...], pages: tuple[int, ...]) -> None:
    model.forward_tokens(
        torch.tensor(prompt),
        torch.arange(len(prompt)),
        pool,
        list(pages),
        seq_len=len(prompt),
        write_from=0,
    )


def _score_sequential(runner, chunk: ScheduledChunk, state) -> tuple:
    """The pre-#215 oracle: one runner call for every target position."""
    target = []
    for offset in range(chunk.num_tokens):
        one = ScheduledChunk(
            chunk.request_id,
            num_tokens=1,
            is_prefill=False,
            position=chunk.position + offset,
        )
        target.append(runner.execute((one,), {chunk.request_id: state})[chunk.request_id][0])
    return tuple(target)


def _token_ids(records: tuple) -> tuple[int, ...]:
    return tuple(record.token_id for record in records)


def test_sampling_history_copy_is_limited_to_speculative_overlays():
    outputs = [11, 12, 13]
    ordinary = SimpleNamespace(outputs=outputs, outputs_override=False)
    overlay = SimpleNamespace(outputs=outputs, outputs_override=True)

    assert PagedModelRunner._sampling_outputs(ordinary, 2) is outputs
    assert PagedModelRunner._sampling_outputs(overlay, 2) == (11, 12)


def _mla_model():
    from kairyu.models.config import parse_model_config
    from kairyu.models.llama import DenseDecoder

    torch.manual_seed(215)
    return DenseDecoder(parse_model_config(TINY_MLA)).eval()


def _logical_kv(pool, pages: tuple[int, ...], logical_length: int):
    """Read only the logical prefix; rejected speculative suffixes may stay stale."""
    keys = torch.stack(
        [
            pool.k[:, pages[position // PAGE], position % PAGE]
            for position in range(logical_length)
        ],
        dim=1,
    )
    values = torch.stack(
        [
            pool.v[:, pages[position // PAGE], position % PAGE]
            for position in range(logical_length)
        ],
        dim=1,
    )
    return keys, values


def test_two_requests_score_every_position_in_one_tensor_forward(monkeypatch):
    candidate_model = _model()
    reference_model = _model()
    candidate_pool = _pool(candidate_model)
    reference_pool = _pool(reference_model)
    candidate = PagedModelRunner(candidate_model, candidate_pool)
    reference = PagedModelRunner(reference_model, reference_pool)

    prompts = {
        "a": (1, 2, 3, 4),
        "b": (5, 6, 7),
    }
    pages = {
        "a": (0, 1),
        "b": (2, 3),
    }
    # outputs = committed token followed by the complete draft overlay.
    overlays = {
        "a": (11, 12, 13),
        "b": (21, 22),
    }
    chunks = (
        ScheduledChunk("a", num_tokens=3, is_prefill=False, position=1),
        ScheduledChunk("b", num_tokens=2, is_prefill=False, position=1),
    )

    for request_id in prompts:
        _prime_prompt(
            candidate_model,
            candidate_pool,
            prompts[request_id],
            pages[request_id],
        )
        _prime_prompt(
            reference_model,
            reference_pool,
            prompts[request_id],
            pages[request_id],
        )

    candidate_states = {
        request_id: _overlay_state(
            request_id,
            prompts[request_id],
            pages[request_id],
            overlays[request_id],
        )
        for request_id in prompts
    }
    reference_states = {
        request_id: _overlay_state(
            request_id,
            prompts[request_id],
            pages[request_id],
            overlays[request_id],
        )
        for request_id in prompts
    }

    tensor_rows: list[int] = []
    forward = candidate_model.forward_decode_tensors

    def counted_forward(token_ids, *args, **kwargs):
        tensor_rows.append(int(token_ids.shape[0]))
        return forward(token_ids, *args, **kwargs)

    monkeypatch.setattr(candidate_model, "forward_decode_tensors", counted_forward)

    actual = candidate.execute(chunks, candidate_states)
    expected = {
        chunk.request_id: _score_sequential(
            reference,
            chunk,
            reference_states[chunk.request_id],
        )
        for chunk in chunks
    }

    assert tensor_rows == [sum(chunk.num_tokens for chunk in chunks)]
    assert {
        request_id: _token_ids(records)
        for request_id, records in actual.items()
    } == {
        request_id: _token_ids(records)
        for request_id, records in expected.items()
    }
    assert {request_id: len(records) for request_id, records in actual.items()} == {
        "a": 3,
        "b": 2,
    }

    stats = candidate.verification_execution_stats()
    assert stats["capability_gap"] is None
    assert stats["requests"] == 2
    assert stats["positions"] == 5
    assert stats["model_calls"] == 1
    assert stats["batched_groups"] == 1
    assert stats["sequential_positions"] == 0

    for chunk in chunks:
        request_id = chunk.request_id
        # The final target is a bonus token: its own KV is not written.  The
        # valid KV prefix therefore ends at the final input/draft position.
        logical_length = (
            len(prompts[request_id])
            + chunk.position
            + chunk.num_tokens
            - 1
        )
        actual_kv = _logical_kv(
            candidate_pool,
            pages[request_id],
            logical_length,
        )
        expected_kv = _logical_kv(
            reference_pool,
            pages[request_id],
            logical_length,
        )
        assert torch.allclose(actual_kv[0], expected_kv[0], atol=1e-5)
        assert torch.allclose(actual_kv[1], expected_kv[1], atol=1e-5)


def test_list_decode_capability_keeps_verification_batched(monkeypatch):
    candidate_model = _model()
    reference_model = _model()
    for layer in candidate_model.model.layers:
        # Model/layer list batching remains valid even when a backend cannot
        # satisfy the stricter tensor/graph contract.
        layer.self_attn.backend.supports_graph_capture = False
    candidate_pool = _pool(candidate_model)
    reference_pool = _pool(reference_model)
    candidate = PagedModelRunner(candidate_model, candidate_pool)
    reference = PagedModelRunner(reference_model, reference_pool)
    assert candidate._tensor_decode_supported is False
    assert candidate._verification_batch_gap is None

    prompt = (1, 2, 3, 4)
    pages = (0, 1)
    chunk = ScheduledChunk("list-only", 3, False, 1)
    _prime_prompt(candidate_model, candidate_pool, prompt, pages)
    _prime_prompt(reference_model, reference_pool, prompt, pages)
    candidate_state = _overlay_state(
        "list-only",
        prompt,
        pages,
        (11, 12, 13),
    )
    reference_state = _overlay_state(
        "list-only",
        prompt,
        pages,
        (11, 12, 13),
    )

    batch_rows: list[int] = []
    forward_decode_batch = candidate_model.forward_decode_batch

    def counted_batch(token_ids, *args, **kwargs):
        batch_rows.append(int(token_ids.shape[0]))
        return forward_decode_batch(token_ids, *args, **kwargs)

    monkeypatch.setattr(
        candidate_model,
        "forward_decode_batch",
        counted_batch,
    )
    actual = candidate.execute((chunk,), {"list-only": candidate_state})
    expected = _score_sequential(reference, chunk, reference_state)

    assert batch_rows == [3]
    assert _token_ids(actual["list-only"]) == _token_ids(expected)
    stats = candidate.verification_execution_stats()
    assert stats["capability_gap"] is None
    assert stats["model_calls"] == 1
    assert stats["batched_groups"] == 1
    assert stats["sequential_positions"] == 0


def test_list_decode_capability_keeps_ordinary_decode_batched(monkeypatch):
    candidate_model = _model()
    reference_model = _model()
    for layer in candidate_model.model.layers:
        # Tensor/graph decode is unavailable, but the model, layers, and backend
        # still implement the established list-metadata batch contract.
        layer.self_attn.backend.supports_graph_capture = False
    candidate_pool = _pool(candidate_model)
    reference_pool = _pool(reference_model)
    candidate = PagedModelRunner(candidate_model, candidate_pool)
    reference = PagedModelRunner(reference_model, reference_pool)
    assert candidate._tensor_decode_supported is False
    assert candidate._decode_batch_gap is None

    prompts = {
        "list-a": (1, 2, 3, 4),
        "list-b": (5, 6, 7),
    }
    pages = {
        "list-a": (0, 1),
        "list-b": (2, 3),
    }
    chunks = (
        ScheduledChunk("list-a", num_tokens=1, is_prefill=False, position=1),
        ScheduledChunk("list-b", num_tokens=1, is_prefill=False, position=1),
    )
    for request_id in prompts:
        _prime_prompt(
            candidate_model,
            candidate_pool,
            prompts[request_id],
            pages[request_id],
        )
        _prime_prompt(
            reference_model,
            reference_pool,
            prompts[request_id],
            pages[request_id],
        )
    candidate_states = {
        request_id: _overlay_state(
            request_id,
            prompts[request_id],
            pages[request_id],
            (11,),
            override=False,
        )
        for request_id in prompts
    }
    reference_states = {
        request_id: _overlay_state(
            request_id,
            prompts[request_id],
            pages[request_id],
            (11,),
            override=False,
        )
        for request_id in prompts
    }

    batch_rows: list[int] = []
    forward_decode_batch = candidate_model.forward_decode_batch

    def counted_batch(token_ids, *args, **kwargs):
        batch_rows.append(int(token_ids.shape[0]))
        return forward_decode_batch(token_ids, *args, **kwargs)

    def forbidden_sequential(*_args, **_kwargs):
        raise AssertionError("valid list decode was unnecessarily serialized")

    monkeypatch.setattr(candidate_model, "forward_decode_batch", counted_batch)
    monkeypatch.setattr(candidate_model, "forward_tokens", forbidden_sequential)

    actual = candidate.execute(chunks, candidate_states)
    expected = {
        chunk.request_id: _score_sequential(
            reference,
            chunk,
            reference_states[chunk.request_id],
        )
        for chunk in chunks
    }

    assert batch_rows == [2]
    assert {
        request_id: _token_ids(records)
        for request_id, records in actual.items()
    } == {
        request_id: _token_ids(records)
        for request_id, records in expected.items()
    }


def test_mla_ordinary_decode_falls_back_per_request_before_batch_call(monkeypatch):
    candidate_model = _mla_model()
    reference_model = _mla_model()
    candidate_pool = _pool(candidate_model)
    reference_pool = _pool(reference_model)
    candidate = PagedModelRunner(candidate_model, candidate_pool)
    reference = PagedModelRunner(reference_model, reference_pool)
    assert candidate._tensor_decode_supported is False
    assert "MlaAttention" in candidate._decode_batch_gap

    prompts = {
        "mla-a": (1, 2, 3, 4),
        "mla-b": (5, 6, 7),
    }
    pages = {
        "mla-a": (0, 1),
        "mla-b": (2, 3),
    }
    chunks = (
        ScheduledChunk("mla-a", num_tokens=1, is_prefill=False, position=1),
        ScheduledChunk("mla-b", num_tokens=1, is_prefill=False, position=1),
    )
    for request_id in prompts:
        _prime_prompt(
            candidate_model,
            candidate_pool,
            prompts[request_id],
            pages[request_id],
        )
        _prime_prompt(
            reference_model,
            reference_pool,
            prompts[request_id],
            pages[request_id],
        )
    candidate_states = {
        request_id: _overlay_state(
            request_id,
            prompts[request_id],
            pages[request_id],
            (11,),
            override=False,
        )
        for request_id in prompts
    }
    reference_states = {
        request_id: _overlay_state(
            request_id,
            prompts[request_id],
            pages[request_id],
            (11,),
            override=False,
        )
        for request_id in prompts
    }

    position_calls: list[int] = []
    forward_tokens = candidate_model.forward_tokens

    def counted_forward(token_ids, *args, **kwargs):
        position_calls.append(int(token_ids.shape[0]))
        return forward_tokens(token_ids, *args, **kwargs)

    def forbidden_batch(*_args, **_kwargs):
        raise AssertionError("MLA ordinary decode entered an unsupported batch path")

    monkeypatch.setattr(candidate_model, "forward_tokens", counted_forward)
    monkeypatch.setattr(candidate_model, "forward_decode_batch", forbidden_batch)
    monkeypatch.setattr(candidate_model, "forward_decode_tensors", forbidden_batch)

    actual = candidate.execute(chunks, candidate_states)
    expected = {
        chunk.request_id: _score_sequential(
            reference,
            chunk,
            reference_states[chunk.request_id],
        )
        for chunk in chunks
    }

    assert position_calls == [1, 1]
    assert {
        request_id: _token_ids(records)
        for request_id, records in actual.items()
    } == {
        request_id: _token_ids(records)
        for request_id, records in expected.items()
    }
    assert torch.equal(candidate_pool.k, reference_pool.k)
    assert torch.equal(candidate_pool.v, reference_pool.v)


def test_mla_verification_falls_back_per_position_and_keeps_tp_packet_width(
    monkeypatch,
):
    """DeepSeek-V3's MLA has no tensor-decode contract.

    ``DenseDecoder.forward_decode_batch`` still exists, so method sniffing at
    model level reproduces the P1 late failure.  The construction-time
    capability gate must retain the established one-position execution while
    preserving the multi-token public result and TP packet layout.
    """

    candidate_model = _mla_model()
    reference_model = _mla_model()
    candidate_pool = _pool(candidate_model)
    reference_pool = _pool(reference_model)
    candidate = PagedModelRunner(candidate_model, candidate_pool)
    reference = PagedModelRunner(reference_model, reference_pool)
    assert candidate._tensor_decode_supported is False
    assert "MlaAttention" in candidate._verification_batch_gap

    prompts = {
        "mla-a": (1, 2, 3, 4),
        "mla-b": (5, 6, 7),
    }
    pages = {
        "mla-a": (0, 1),
        "mla-b": (2, 3),
    }
    overlays = {
        "mla-a": (11, 12, 13),
        "mla-b": (21, 22),
    }
    chunks = (
        ScheduledChunk("mla-a", num_tokens=3, is_prefill=False, position=1),
        ScheduledChunk("mla-b", num_tokens=2, is_prefill=False, position=1),
    )
    for request_id in prompts:
        _prime_prompt(
            candidate_model,
            candidate_pool,
            prompts[request_id],
            pages[request_id],
        )
        _prime_prompt(
            reference_model,
            reference_pool,
            prompts[request_id],
            pages[request_id],
        )

    candidate_states = {
        request_id: _overlay_state(
            request_id,
            prompts[request_id],
            pages[request_id],
            overlays[request_id],
        )
        for request_id in prompts
    }
    reference_states = {
        request_id: _overlay_state(
            request_id,
            prompts[request_id],
            pages[request_id],
            overlays[request_id],
        )
        for request_id in prompts
    }

    position_calls: list[int] = []
    forward_tokens = candidate_model.forward_tokens

    def counted_forward(token_ids, *args, **kwargs):
        position_calls.append(int(token_ids.shape[0]))
        return forward_tokens(token_ids, *args, **kwargs)

    def forbidden_batch(*_args, **_kwargs):
        raise AssertionError("MLA verification entered an unsupported batch path")

    monkeypatch.setattr(candidate_model, "forward_tokens", counted_forward)
    monkeypatch.setattr(
        candidate_model,
        "forward_decode_batch",
        forbidden_batch,
    )
    monkeypatch.setattr(
        candidate_model,
        "forward_decode_tensors",
        forbidden_batch,
    )

    actual = candidate.execute(chunks, candidate_states)
    expected = {
        chunk.request_id: _score_sequential(
            reference,
            chunk,
            reference_states[chunk.request_id],
        )
        for chunk in chunks
    }

    assert position_calls == [1] * 5
    assert {
        request_id: _token_ids(records)
        for request_id, records in actual.items()
    } == {
        request_id: _token_ids(records)
        for request_id, records in expected.items()
    }
    assert {request_id: len(records) for request_id, records in actual.items()} == {
        "mla-a": 3,
        "mla-b": 2,
    }

    stats = candidate.verification_execution_stats()
    assert stats["enabled"] is True
    assert "MlaAttention" in stats["capability_gap"]
    assert stats["requests"] == 2
    assert stats["positions"] == 5
    assert stats["model_calls"] == 5
    assert stats["batched_groups"] == 0
    assert stats["sequential_positions"] == 5

    packet = candidate.make_sampling_token_packet(
        chunks,
        candidate_states,
        sampled=actual,
    )
    expected_packet = (
        *_token_ids(actual["mla-a"]),
        *_token_ids(actual["mla-b"]),
    )
    assert tuple(packet.tolist()) == expected_packet
    assert packet.numel() == 5
    follower_buffer = candidate.make_sampling_token_packet(
        chunks,
        candidate_states,
    )
    assert follower_buffer.tolist() == [-1] * 5


def test_partial_rejection_correction_overwrites_first_stale_slot():
    clean_model = _model()
    candidate_model = _model()
    clean_pool = _pool(clean_model)
    candidate_pool = _pool(candidate_model)
    clean = PagedModelRunner(clean_model, clean_pool)
    candidate = PagedModelRunner(candidate_model, candidate_pool)

    request_id = "r"
    prompt = (1, 2, 3, 4)
    pages = (0, 1)
    committed = 17
    first_position = 1
    _prime_prompt(clean_model, clean_pool, prompt, pages)
    _prime_prompt(candidate_model, candidate_pool, prompt, pages)

    # Build a clean autoregressive reference.  draft[0] is chosen to match its
    # target and draft[1] to reject, so the candidate accepts exactly one draft
    # before returning the correction/bonus token.
    target0 = _score_sequential(
        clean,
        ScheduledChunk(request_id, 1, False, first_position),
        _overlay_state(
            request_id,
            prompt,
            pages,
            (committed,),
            override=False,
        ),
    )[0].token_id
    target1 = _score_sequential(
        clean,
        ScheduledChunk(request_id, 1, False, first_position + 1),
        _overlay_state(
            request_id,
            prompt,
            pages,
            (committed, target0),
            override=False,
        ),
    )[0].token_id
    expected_next = _score_sequential(
        clean,
        ScheduledChunk(request_id, 1, False, first_position + 2),
        _overlay_state(
            request_id,
            prompt,
            pages,
            (committed, target0, target1),
            override=False,
        ),
    )[0].token_id

    rejected_draft = (target1 + 1) % candidate_model.config.vocab_size
    assert rejected_draft != target1
    trailing_draft = (rejected_draft + 1) % candidate_model.config.vocab_size
    verify = ScheduledChunk(
        request_id,
        num_tokens=4,
        is_prefill=False,
        position=first_position,
    )
    targets = candidate.execute(
        (verify,),
        {
            request_id: _overlay_state(
                request_id,
                prompt,
                pages,
                (committed, target0, rejected_draft, trailing_draft),
            )
        },
    )[request_id]

    assert _token_ids(targets[:2]) == (target0, target1)
    rejected_absolute = len(prompt) + first_position + 1
    rejected_page = pages[rejected_absolute // PAGE]
    rejected_slot = rejected_absolute % PAGE
    stale_k = candidate_pool.k[:, rejected_page, rejected_slot].clone()
    stale_v = candidate_pool.v[:, rejected_page, rejected_slot].clone()

    # Scheduler shortfall commits target0 + target1 and starts the next step at
    # the actual output length.  Its input KV lands exactly on draft[1]'s stale
    # slot; no rejected-suffix clearing pass is required.
    correction_position = first_position + 2
    corrected = candidate.execute(
        (
            ScheduledChunk(
                request_id,
                num_tokens=1,
                is_prefill=False,
                position=correction_position,
            ),
        ),
        {
            request_id: _overlay_state(
                request_id,
                prompt,
                pages,
                (committed, target0, target1),
                override=False,
            )
        },
    )[request_id][0].token_id

    assert corrected == expected_next
    assert not torch.equal(
        candidate_pool.k[:, rejected_page, rejected_slot],
        stale_k,
    )
    assert not torch.equal(
        candidate_pool.v[:, rejected_page, rejected_slot],
        stale_v,
    )

    valid_length = rejected_absolute + 1
    candidate_kv = _logical_kv(candidate_pool, pages, valid_length)
    clean_kv = _logical_kv(clean_pool, pages, valid_length)
    assert torch.allclose(candidate_kv[0], clean_kv[0], atol=1e-5)
    assert torch.allclose(candidate_kv[1], clean_kv[1], atol=1e-5)


def test_batched_verification_preserves_greedy_logprob_records():
    candidate_model = _model()
    reference_model = _model()
    candidate_pool = _pool(candidate_model)
    reference_pool = _pool(reference_model)
    candidate = PagedModelRunner(
        candidate_model, candidate_pool, sampler=Sampler()
    )
    reference = PagedModelRunner(
        reference_model, reference_pool, sampler=Sampler()
    )
    prompt = (1, 2, 3, 4)
    pages = (0, 1)
    overlay = (11, 12, 13)
    chunk = ScheduledChunk("logprob", 3, False, 1)
    for model, pool in (
        (candidate_model, candidate_pool),
        (reference_model, reference_pool),
    ):
        _prime_prompt(model, pool, prompt, pages)
    candidate_state = _overlay_state(
        "logprob", prompt, pages, overlay
    )
    reference_state = _overlay_state(
        "logprob", prompt, pages, overlay
    )
    sampling = EngineSampling(temperature=0.0, logprobs=3)
    candidate_state.request.sampling = sampling
    reference_state.request.sampling = sampling

    actual = candidate.execute(
        (chunk,), {"logprob": candidate_state}
    )["logprob"]
    expected = _score_sequential(reference, chunk, reference_state)

    assert _token_ids(actual) == _token_ids(expected)
    for candidate_token, reference_token in zip(
        actual, expected, strict=True
    ):
        assert candidate_token.logprob is not None
        assert reference_token.logprob is not None
        assert abs(candidate_token.logprob - reference_token.logprob) < 1e-6
        assert candidate_token.top_logprobs is not None
        assert reference_token.top_logprobs is not None
        assert tuple(index for index, _ in candidate_token.top_logprobs) == tuple(
            index for index, _ in reference_token.top_logprobs
        )
        for (_, candidate_value), (_, reference_value) in zip(
            candidate_token.top_logprobs,
            reference_token.top_logprobs,
            strict=True,
        ):
            assert abs(candidate_value - reference_value) < 1e-6


def test_snapshot_graph_buckets_use_flattened_verification_row_count():
    model = _model()
    cache = RadixKVCache(num_pages=12, page_size=PAGE)
    pool = PagedKVPool.for_cache(cache, model.config)
    allocated = tuple(cache.allocate_private_page() for _ in range(4))
    pages = {
        "a": allocated[:2],
        "b": allocated[2:],
    }
    prompts = {
        "a": (1, 2, 3, 4),
        "b": (5, 6, 7),
    }
    backend = SnapshotGraphBackend()
    runner = PagedModelRunner(
        model,
        pool,
        cache=cache,
        graph_backend=backend,
        graph_max_batch=8,
        graph_max_pages=2,
    )
    for request_id in prompts:
        _prime_prompt(model, pool, prompts[request_id], pages[request_id])

    chunks = (
        ScheduledChunk("a", 3, False, 1),
        ScheduledChunk("b", 2, False, 1),
    )

    def execute(overlays):
        return runner.execute(
            chunks,
            {
                request_id: _overlay_state(
                    request_id,
                    prompts[request_id],
                    pages[request_id],
                    overlays[request_id],
                )
                for request_id in prompts
            },
        )

    first = execute({"a": (11, 12, 13), "b": (21, 22)})
    second = execute({"a": (27, 28, 29), "b": (31, 32)})

    assert {request_id: len(records) for request_id, records in first.items()} == {
        "a": 3,
        "b": 2,
    }
    assert {request_id: len(records) for request_id, records in second.items()} == {
        "a": 3,
        "b": 2,
    }
    assert backend.captures == 1
    assert backend.replays == 2
    assert set(runner._graph._captured) == {8}

    _, static = runner._graph._captured[8]
    assert static.token_ids[:5].tolist() == [27, 28, 29, 31, 32]
    assert static.positions[:5].tolist() == [4, 5, 6, 3, 4]
    assert static.seq_lens[:5].tolist() == [5, 6, 7, 4, 5]
    assert static.page_tables[:5].tolist() == [
        list(pages["a"]),
        list(pages["a"]),
        list(pages["a"]),
        list(pages["b"]),
        list(pages["b"]),
    ]
    assert torch.all(
        static.page_tables[5:] == runner._graph_scratch_page
    )
    assert static.seq_lens[5:].tolist() == [1, 1, 1]

    # Three flattened rows select bucket 4, independently of request count.
    third = runner.execute(
        (ScheduledChunk("a", 3, False, 1),),
        {
            "a": _overlay_state(
                "a",
                prompts["a"],
                pages["a"],
                (41, 42, 43),
            )
        },
    )
    assert len(third["a"]) == 3
    assert backend.captures == 2
    assert backend.replays == 3
    assert set(runner._graph._captured) == {4, 8}

    stats = runner.verification_execution_stats()
    assert stats["requests"] == 5
    assert stats["positions"] == 13
    assert stats["model_calls"] == 3
    assert stats["batched_groups"] == 3
    assert stats["graph_groups"] == 3
    assert stats["graph"]["graph_executions"] == 3
    assert stats["graph"]["captured_buckets"] == (4, 8)
