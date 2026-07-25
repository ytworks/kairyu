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


@pytest.fixture(scope="module")
def _served(model_dir):
    return model_dir


def test_pd_separation_builds_a_coordinator_backed_loop(model_dir):
    """m2 §2.4 reserved `pd_separation` as a config surface and never wired it;
    a deployment could only reach `PDCoordinator` from Python."""
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.kairyu_backend import build_engine_loop

    loop, _cache, _scheduler = build_engine_loop(
        model_path=model_dir, pd_separation=True, num_pages=64, page_size=16
    )
    coordinator = loop.pd_coordinator
    assert coordinator is not None
    # the serving loop observes the DECODE scheduler: requests enter at prefill
    # and finish under decode
    assert loop._scheduler is coordinator.decode_scheduler
    if torch.cuda.is_available():
        assert isinstance(coordinator._handoff, StreamCopyKVHandoff)


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
