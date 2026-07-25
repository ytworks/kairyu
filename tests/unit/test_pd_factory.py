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


def _require_selected_backend() -> None:
    """`build_pd_coordinator` places both halves with the PRODUCTION probe, so on
    a GPU host it selects FlashInfer — an optional `[gpu]` extra. Not having it
    installed is an environment gap, like `transformers` above, not a failure of
    this seam."""
    from kairyu.engine.core.attention_selector import select_backend_name
    from kairyu.engine.core.hw_profile import probe

    if select_backend_name(probe()) == "flashinfer":
        pytest.importorskip("flashinfer")


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

    _require_selected_backend()
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


def test_every_deferred_copy_is_settled_not_just_the_last():
    """A prefill step transfers every prompt that completed in it, so several
    copies can be in flight at once; keeping one slot drops the earlier ones."""
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    provider = CpuNoopStream()
    handoff = StreamCopyKVHandoff(_Inner(), provider, defer=True)
    handoff.transfer((1,), 0)
    handoff.transfer((2,), 0)

    assert len(handoff.pending_events) == 2
    handoff.gate_pending()
    assert provider.events.count("wait") == 2, provider.events
    assert handoff.pending_events == ()


def test_gating_settles_the_copies_without_a_host_wait():
    """`gate_pending` expresses the dependency on the caller's stream — that is
    what leaves the producer's already-queued step running."""
    from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff

    provider = CpuNoopStream()
    handoff = StreamCopyKVHandoff(_Inner(), provider, defer=True)
    handoff.transfer((1, 2), 0)
    handoff.gate_pending()

    assert provider.events == ["begin", "record", "wait"]
    # never the blocking form's stream-wide stop
    assert "synchronize" not in provider.events


def test_deferring_is_opt_in_because_it_is_a_consumer_contract():
    from kairyu.engine.core.pd_factory import build_cpu_kv_handoff

    assert build_cpu_kv_handoff(_Inner()).defers is False
    assert build_cpu_kv_handoff(_Inner(), defer=True).defers is True


def test_a_host_pool_stays_plain_even_when_deferring_is_asked_for():
    from kairyu.engine.core.pd_factory import build_kv_handoff

    inner = _Inner()
    assert build_kv_handoff(inner, _pool(), defer=True) is inner


def test_the_production_coordinator_enables_and_settles_the_deferred_copy(model_dir):
    """The finding this closes: nothing enabled or waited on the deferred path.

    `build_pd_coordinator` is that caller — it turns defer on wherever a side
    stream exists, and `PDCoordinator` is what holds the prefill-side lease until
    the copy's completion event releases it.
    """
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    _require_selected_backend()
    coordinator = build_pd_coordinator(model_path=model_dir, num_pages=64, page_size=16)
    handoff = coordinator._handoff

    if torch.cuda.is_available():
        assert isinstance(handoff, StreamCopyKVHandoff)
        assert handoff.defers, "the one caller that settles the copy did not enable it"
        assert coordinator._gate_pending is not None
    else:
        # a host pool has no side stream to defer onto; the plain handoff blocks
        assert not isinstance(handoff, StreamCopyKVHandoff)
        assert coordinator._gate_pending is None


def test_the_deferred_copy_can_be_declined(model_dir):
    from kairyu.engine.core.handoff_stream import StreamCopyKVHandoff
    from kairyu.engine.core.pd_factory import build_pd_coordinator

    _require_selected_backend()
    coordinator = build_pd_coordinator(
        model_path=model_dir, num_pages=64, page_size=16, defer_handoff=False
    )
    handoff = coordinator._handoff
    assert not isinstance(handoff, StreamCopyKVHandoff) or not handoff.defers
    assert coordinator._gate_pending is None


def test_the_deferred_and_blocking_copies_generate_the_same_tokens(model_dir):
    """The end-to-end gate: on a GPU host this runs the real deferred handoff
    through `PDCoordinator`, and overlap must not change the answer."""
    from kairyu.engine.core.pd_factory import build_pd_coordinator
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest

    _require_selected_backend()

    def _generate(defer_handoff: bool):
        coordinator = build_pd_coordinator(
            model_path=model_dir, num_pages=64, page_size=16,
            defer_handoff=defer_handoff,
        )
        coordinator.add_request(
            EngineRequest(
                "a", tuple(range(9)), max_new_tokens=4, sampling=EngineSampling()
            )
        )
        outputs = coordinator.run_to_completion()
        assert not coordinator.failed_requests
        return outputs

    assert _generate(True) == _generate(False)
