"""The P-D construction seam (m18 D3 / G2 stage 5.3 entry).

`PDCoordinator` existed only in tests, so no deployment could reach a
`KVHandoff` — which is why `CudaStreamProvider` had no production caller. These
pin the placement decision and the assembled engine.
"""

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

# head_dim 64: on a CUDA profile the selector picks FlashInfer, whose MMA tiles
# reject a 16-wide head outright, so a smaller fixture would only ever exercise
# the torch fallback (same lesson as tests/gpu/test_tp_placement.py)
TINY = dict(
    vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    torch.manual_seed(71)
    path = tmp_path_factory.mktemp("pd-factory")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY)).to(
        torch.float32
    ).eval().save_pretrained(path, safe_serialization=True)
    # written locally, not fetched: a unit test must not reach the Hub
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0", "truncation": None, "padding": None,
                "added_tokens": [], "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None, "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": {f"<{index}>": index for index in range(256)},
                    "unk_token": "<0>",
                },
            }
        )
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<0>"})
    )
    return str(path)


def _pool(device="cpu"):
    from kairyu.engine.core.kv_pool import PagedKVPool

    return PagedKVPool(
        num_layers=1, num_pages=4, page_size=4, num_kv_heads=1, head_dim=4,
        device=device,
    )


class _Inner:
    def transfer(self, tokens, first_token, pages=()):
        return ("allocation", tokens, first_token, pages)


def test_a_host_pool_gets_the_plain_handoff():
    """A stream window around a host copy buys nothing, and CudaStreamProvider
    requires CUDA."""
    from kairyu.engine.core.pd_factory import build_kv_handoff

    inner = _Inner()
    assert build_kv_handoff(inner, _pool()) is inner


def test_forcing_the_side_stream_without_cuda_is_an_error():
    """Rather than silently degrading to the plain handoff."""
    from kairyu.engine.core.pd_factory import build_kv_handoff

    with pytest.raises(ValueError, match="needs a CUDA KV pool"):
        build_kv_handoff(_Inner(), _pool(), force_side_stream=True)


def test_the_side_stream_can_be_declined_explicitly():
    from kairyu.engine.core.pd_factory import build_kv_handoff

    inner = _Inner()
    assert build_kv_handoff(inner, _pool(), force_side_stream=False) is inner


def test_the_coordinator_assembles_and_generates(model_dir):
    from kairyu.engine.core.pd_factory import build_pd_coordinator
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest

    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16
    )
    coordinator.add_request(
        EngineRequest("a", tuple(range(9)), max_new_tokens=4, sampling=EngineSampling())
    )
    outputs = coordinator.run_to_completion()
    assert len(outputs["a"]) == 4
    assert not coordinator.failed_requests


def test_the_deferred_handoff_records_an_event_instead_of_blocking():
    """The ordering contract, on the recording provider."""
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    provider = CpuNoopStream()
    handoff = StreamCopyKVHandoff(_Inner(), provider, defer=True)
    handoff.transfer((1, 2), 0)

    assert provider.events == ["begin", "record"], provider.events
    assert handoff.pending_event is not None
    handoff.wait_for_pending()
    assert provider.events == ["begin", "record", "wait"]
    assert handoff.pending_event is None


def test_the_default_handoff_still_blocks_before_returning():
    """m6 D4: the commit point must not run ahead of the copy unless the caller
    explicitly takes that responsibility."""
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    provider = CpuNoopStream()
    handoff = StreamCopyKVHandoff(_Inner(), provider)
    handoff.transfer((1, 2), 0)

    assert provider.events == ["begin", "synchronize"]
    assert handoff.pending_event is None


def test_a_raising_deferred_transfer_still_closes_the_window():
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    class _Boom:
        def transfer(self, tokens, first_token, pages=()):
            raise RuntimeError("transfer failed")

    provider = CpuNoopStream()
    with pytest.raises(RuntimeError, match="transfer failed"):
        StreamCopyKVHandoff(_Boom(), provider, defer=True).transfer((1,), 0)
    assert provider.events == ["begin", "record"]


def test_pd_separation_builds_a_coordinator_backed_loop(model_dir):
    """m2 §2.4 reserved `pd_separation` as a config surface and never wired it;
    a deployment could only reach `PDCoordinator` from Python."""
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd_loop import PDLoopAdapter
    from kairyu.engine.kairyu_backend import build_engine_loop

    loop, cache, scheduler = build_engine_loop(
        model_path=model_dir, pd_separation=True, num_pages=64, page_size=16
    )
    coordinator = loop.pd_coordinator
    assert coordinator is not None
    # the loop drives the coordinator through the adapter, and reports the cache
    # and scheduler it ACTUALLY drives -- not a third, unrelated pair
    assert isinstance(scheduler, PDLoopAdapter)
    assert loop._scheduler is scheduler and loop._runner is scheduler
    assert scheduler.coordinator is coordinator
    assert cache is coordinator.decode_cache
    if torch.cuda.is_available():
        assert isinstance(coordinator._handoff, StreamCopyKVHandoff)


def _run(loop, request_id, prompt, max_tokens):
    """Drive the loop like the backend pump does, returning the last update."""
    from kairyu.sampling_params import SamplingParams

    loop.submit(request_id, prompt, SamplingParams(max_tokens=max_tokens, temperature=0.0))
    last = None
    for _ in range(200):
        if not loop.has_work():
            break
        for updated_id, update in loop.step():
            if updated_id == request_id:
                last = update
    assert last is not None, "the loop produced no update at all"
    return last


def test_a_request_flows_prefill_handoff_then_decode(model_dir):
    """The wiring test the construction assertions could not do: submit through
    the P-D loop and require actual output.

    Handing `EngineLoop` the bare coordinator cannot serve even one request --
    `_drain_ops` adds submissions to the scheduler it was given, so they land in
    the DECODE scheduler with no prompt KV and `PDCoordinator.add_request` is
    never called.
    """
    from kairyu.engine.kairyu_backend import build_engine_loop

    loop, _cache, scheduler = build_engine_loop(
        model_path=model_dir, pd_separation=True, num_pages=64, page_size=16
    )
    coordinator = loop.pd_coordinator

    update = _run(loop, "pd1", "<1> <2> <3> <4>", max_tokens=5)

    assert update.finished and update.outputs
    assert coordinator.failed_requests == ()
    # the request really went through prefill: the coordinator's clone finished
    # there, and decode adopted the transferred KV
    assert coordinator.internal_id_for("pd1") is None  # reclaimed on finish
    assert scheduler.states == {}  # both halves reclaimed (E2)
    assert coordinator.prefill_scheduler.states == {}


def test_pd_output_matches_the_single_engine_path(model_dir):
    """Greedy equivalence is what proves the KV BYTES moved.

    The two halves own two pools. An accounting-only handoff marks the
    destination pages computed without copying anything into them, so decode
    attends over a zero-initialised pool and diverges from the reference.
    """
    from kairyu.engine.kairyu_backend import build_engine_loop

    prompt = "<7> <11> <13> <17> <19>"
    combined, _c, _s = build_engine_loop(
        model_path=model_dir, num_pages=64, page_size=16
    )
    reference = _run(combined, "ref", prompt, max_tokens=6)

    separated, _c2, _s2 = build_engine_loop(
        model_path=model_dir, pd_separation=True, num_pages=64, page_size=16
    )
    served = _run(separated, "pd", prompt, max_tokens=6)

    assert len(reference.outputs) == 6
    assert served.outputs == reference.outputs


def test_the_paged_handoff_copies_the_pages_it_is_given():
    """`LocalKVHandoff` is the accounting half; this one moves the bytes."""
    from kairyu.engine.core.pd_factory import PagedCopyKVHandoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    source, destination = _pool(), _pool()
    torch.manual_seed(5)
    source.k.copy_(torch.randn_like(source.k))
    source.v.copy_(torch.randn_like(source.v))
    cache = RadixKVCache(num_pages=4, page_size=4)
    handoff = PagedCopyKVHandoff(cache, source, destination)

    allocation = handoff.transfer(tuple(range(8)), first_token=3, pages=(2, 1))

    assert allocation.tokens == tuple(range(8))
    for target, origin in zip(allocation.pages, (2, 1), strict=True):
        assert torch.equal(destination.k[:, target], source.k[:, origin])
        assert torch.equal(destination.v[:, target], source.v[:, origin])


def test_the_paged_handoff_skips_pages_the_destination_already_has():
    """Receiver-side dedup (m6 D4): a destination radix hit is a prefix, so the
    source pages it covers must not be re-copied over live shared pages."""
    from kairyu.engine.core.pd_factory import PagedCopyKVHandoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    source, destination = _pool(), _pool()
    torch.manual_seed(6)
    source.k.copy_(torch.randn_like(source.k))
    cache = RadixKVCache(num_pages=4, page_size=4)
    handoff = PagedCopyKVHandoff(cache, source, destination)
    # first transfer publishes tokens 0..3 in the destination's radix tree
    handoff.transfer(tuple(range(4)), first_token=0, pages=(0,))

    allocation = handoff.transfer(tuple(range(8)), first_token=0, pages=(0, 3))

    assert len(allocation.cached_pages) == 1
    tail = allocation.pages[-1]
    assert torch.equal(destination.k[:, tail], source.k[:, 3])


def test_the_paged_handoff_rejects_incompatible_pools():
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pd_factory import PagedCopyKVHandoff
    from kairyu.engine.core.radix_kv import RadixKVCache

    wider = PagedKVPool(
        num_layers=1, num_pages=4, page_size=4, num_kv_heads=2, head_dim=4
    )
    with pytest.raises(ValueError, match="pool mismatch"):
        PagedCopyKVHandoff(RadixKVCache(num_pages=4, page_size=4), _pool(), wider)


def test_pd_separation_needs_a_model():
    from kairyu.engine.kairyu_backend import build_engine_loop

    with pytest.raises(ValueError, match="needs a model_path"):
        build_engine_loop(pd_separation=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tensor_parallel_size": 2}, "tensor_parallel_size"),
        ({"speculative": "ngram"}, "speculative"),
    ],
    ids=["tp", "speculative"],
)
def test_pd_separation_rejects_combinations_it_does_not_implement(
    model_dir, kwargs, match
):
    """Fail loudly rather than silently serving a different topology."""
    from kairyu.engine.kairyu_backend import build_engine_loop

    with pytest.raises(ValueError, match=match):
        build_engine_loop(model_path=model_dir, pd_separation=True, **kwargs)


def test_the_backend_accepts_the_option(model_dir):
    """`backend: kairyu` with `options: {pd_separation: true}` in a deployment
    YAML has to reach the constructor."""
    from kairyu.engine.registry import create_backend

    backend = create_backend(
        "kairyu", model_path=model_dir, pd_separation=True, num_pages=64, page_size=16
    )
    assert backend._loop.pd_coordinator is not None


async def test_the_backend_serves_a_generation_through_the_pair(model_dir):
    """The whole deployment path, not just the constructor: an API request has
    to come back with tokens."""
    from kairyu.engine.backend import GenerationRequest
    from kairyu.engine.registry import create_backend
    from kairyu.sampling_params import SamplingParams

    backend = create_backend(
        "kairyu", model_path=model_dir, pd_separation=True, num_pages=64, page_size=16
    )
    result = await backend.generate(
        GenerationRequest(
            request_id="served",
            prompt="<3> <5> <8>",
            sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
        )
    )

    assert result.finished
    assert len(result.completions[0].token_ids) == 4
    assert result.usage is not None and result.usage.prompt_tokens > 0
    # E2: nothing left behind in either half
    assert backend._scheduler.states == {}
    assert backend._loop.pd_coordinator.prefill_scheduler.states == {}
    await backend.shutdown()
