"""OverlapEngineCore with a REAL runner (m2 §2.2 future tokens).

`overlap.py` states the contract: decode chunks carry an explicit position "so
the runner never needs previously-committed token values from the host". The toy
runner honours it by ignoring outputs entirely; `PagedModelRunner` did not — it
read `state.outputs[position - 1]`, which under overlap is one short because the
snapshot for step N+1 is taken before step N commits. Every real-model overlap
run raised `IndexError: tuple index out of range`.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

TINY = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    torch.manual_seed(71)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY))
    path = tmp_path_factory.mktemp("overlap-llama")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _generate(
    model_dir: str, core_cls, prompt, max_new: int, sampling=None, **scheduler_kwargs
):
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    model, config, _ = load_model(model_dir)
    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(
        cache, max_num_batched_tokens=6, page_size=4, **scheduler_kwargs
    )
    runner = PagedModelRunner(model, PagedKVPool.for_cache(cache, config), sampler=Sampler())
    core = core_cls(scheduler, runner)
    for index, tokens in enumerate(prompt):
        core.add_request(
            EngineRequest(
                f"r{index}", tuple(tokens), max_new_tokens=max_new,
                sampling=sampling or EngineSampling(),
            )
        )
    return core.run_to_completion(), runner


def test_overlap_matches_eager_on_a_real_runner(llama_dir):
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.overlap import OverlapEngineCore

    prompt = [list(range(11))]
    eager, _ = _generate(llama_dir, EngineCore, prompt, 8)
    overlapped, _ = _generate(llama_dir, OverlapEngineCore, prompt, 8)
    assert overlapped["r0"] == eager["r0"]
    assert len(eager["r0"]) == 8


def test_overlap_matches_eager_with_several_requests(llama_dir):
    """Batched decode reads the same path, once per row."""
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.overlap import OverlapEngineCore

    prompt = [list(range(9)), list(range(3, 14)), list(range(7, 15))]
    eager, _ = _generate(llama_dir, EngineCore, prompt, 6)
    overlapped, _ = _generate(llama_dir, OverlapEngineCore, prompt, 6)
    assert overlapped == eager
    assert all(len(v) == 6 for v in eager.values())


def test_committed_outputs_win_over_the_in_flight_buffer(llama_dir):
    """A speculative rollback replaces already-sampled tokens; the buffer must
    not shadow the committed value."""
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.models.loader import load_model

    model, config, _ = load_model(llama_dir)
    cache = RadixKVCache(num_pages=64, page_size=4)
    runner = PagedModelRunner(model, PagedKVPool.for_cache(cache, config))

    class _State:
        class request:  # noqa: N801 - mirrors the runner's duck-typed access
            request_id = "a"
            sampling_identity = "a"  # no P-D clone here: same id
            prompt_token_ids = (1, 2, 3)

        outputs = (41,)

    runner._remember("a", 0, __import__(
        "kairyu.engine.core.sampling_types", fromlist=["SampledToken"]
    ).SampledToken(99))
    # position 1 needs the token at index 0: committed 41, in-flight 99
    assert runner._previous_token(_State(), 1) == 41


def test_a_missing_token_is_an_error_not_a_silent_wrong_input(llama_dir):
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.models.loader import load_model

    model, config, _ = load_model(llama_dir)
    cache = RadixKVCache(num_pages=64, page_size=4)
    runner = PagedModelRunner(model, PagedKVPool.for_cache(cache, config))

    class _State:
        class request:  # noqa: N801
            request_id = "gone"
            sampling_identity = "gone"  # no P-D clone here: same id
            prompt_token_ids = (1, 2)

        outputs = ()

    with pytest.raises(RuntimeError, match="no token for gone at position 2"):
        runner._previous_token(_State(), 3)


def test_the_in_flight_buffer_does_not_grow_with_the_request(llama_dir):
    """A decode reads exactly position-1, so only the newest token is needed.

    EngineLoop calls release() on finish, but a bare EngineCore does not — an
    unbounded per-request history would leak for the length of every request."""
    from kairyu.engine.core.engine_core import EngineCore

    _outputs, runner = _generate(llama_dir, EngineCore, [list(range(9))], 16)
    assert list(runner._future_tokens) == ["r0"]
    # bounded by the pipeline's reach, not by the request length
    # bounded by what is still uncommitted, not by the request length
    assert len(runner._future_tokens["r0"]) <= 2


def test_release_drops_the_buffer(llama_dir):
    from kairyu.engine.core.engine_core import EngineCore

    _outputs, runner = _generate(llama_dir, EngineCore, [list(range(9))], 4)
    runner.release("r0")
    assert runner._future_tokens == {}


@pytest.mark.parametrize(
    "penalty",
    [
        {"presence_penalty": 2.0},
        {"frequency_penalty": 1.5},
        {"repetition_penalty": 1.8},
    ],
    ids=["presence", "frequency", "repetition"],
)
def test_overlap_matches_eager_under_penalties(llama_dir, penalty):
    """The half the first version missed.

    `_decode_inputs` was corrected but `_sample` still handed the sampler the
    stale `state.outputs`, so under overlap the penalties were computed over a
    history missing the in-flight token — a different token set penalised, and a
    different argmax. Pure-greedy tests cannot see it, because greedy ignores the
    history entirely.
    """
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.overlap import OverlapEngineCore
    from kairyu.engine.core.sampling_types import EngineSampling

    sampling = EngineSampling(**penalty)
    prompt = [list(range(11)), list(range(4, 17))]
    eager, _ = _generate(llama_dir, EngineCore, prompt, 10, sampling=sampling)
    overlapped, _ = _generate(llama_dir, OverlapEngineCore, prompt, 10, sampling=sampling)

    assert overlapped == eager
    assert all(len(tokens) == 10 for tokens in eager.values())


def test_effective_outputs_fills_the_gap_the_snapshot_left(llama_dir):
    """Directly: a snapshot two tokens behind must still see both."""
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import SampledToken
    from kairyu.models.loader import load_model

    model, config, _ = load_model(llama_dir)
    cache = RadixKVCache(num_pages=64, page_size=4)
    runner = PagedModelRunner(model, PagedKVPool.for_cache(cache, config))

    class _State:
        class request:  # noqa: N801
            request_id = "a"
            sampling_identity = "a"  # no P-D clone here: same id
            prompt_token_ids = (1, 2, 3)

        outputs = (11,)

    runner._remember("a", 1, SampledToken(22))
    runner._remember("a", 2, SampledToken(33))
    # sampling position 3 must see the committed 11 plus both in-flight tokens
    assert runner._effective_outputs(_State(), 3) == (11, 22, 33)
    # and a committed-only history is returned untouched
    assert runner._effective_outputs(_State(), 1) == (11,)


def test_a_gap_in_the_history_is_an_error(llama_dir):
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.models.loader import load_model

    model, config, _ = load_model(llama_dir)
    cache = RadixKVCache(num_pages=64, page_size=4)
    runner = PagedModelRunner(model, PagedKVPool.for_cache(cache, config))

    class _State:
        class request:  # noqa: N801
            request_id = "a"
            sampling_identity = "a"  # no P-D clone here: same id
            prompt_token_ids = (1,)

        outputs = ()

    with pytest.raises(RuntimeError, match="no token for a at position 0"):
        runner._effective_outputs(_State(), 2)


def test_the_sampler_history_actually_changes_the_pick(llama_dir):
    """The reviewer's repro, pinned directly.

    The end-to-end penalty tests above pass even against the stale history —
    with a real model the penalised token rarely happens to be the argmax, so
    they cannot demonstrate the defect. This does: presence_penalty 2.0 over
    logits [0.0, 1.0, 0.9] picks token 1 with an empty history and token 2 once
    the in-flight token 1 is included.
    """
    import torch

    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling

    logits = torch.tensor([0.0, 1.0, 0.9])
    sampling = EngineSampling(presence_penalty=2.0)

    sampler = Sampler()
    stale = sampler.sample("a", sampling, 1, logits, prompt=(5,), outputs=())
    sampler.release("a")
    with_in_flight = sampler.sample("a", sampling, 1, logits, prompt=(5,), outputs=(1,))

    assert stale.token_id == 1
    assert with_in_flight.token_id == 2, "the histories must diverge for this to test anything"


def test_the_runner_hands_the_sampler_the_in_flight_history(llama_dir):
    """And that the runner supplies the second of those two histories."""
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import SampledToken
    from kairyu.models.loader import load_model

    model, config, _ = load_model(llama_dir)
    cache = RadixKVCache(num_pages=64, page_size=4)

    seen = {}

    class _Recording:
        def sample(self, request_id, sampling, position, logits, **kwargs):
            seen["outputs"] = kwargs["outputs"]
            return SampledToken(0)

        def release(self, request_id):
            return None

    runner = PagedModelRunner(
        model, PagedKVPool.for_cache(cache, config), sampler=_Recording()
    )

    class _State:
        class request:  # noqa: N801
            request_id = "a"
            sampling_identity = "a"  # no P-D clone here: same id
            prompt_token_ids = (5,)
            sampling = None
            eos_token_id = None

        outputs = ()

    runner._remember("a", 0, SampledToken(1))  # step N sampled 1, not yet committed
    runner._sample(_State(), torch.tensor([0.0, 1.0, 0.9]), position=1)

    # the stale snapshot would have handed over (); the pick differs on that
    assert seen["outputs"] == (1,)


@pytest.mark.parametrize("depth", [1, 2, 4, 10])
def test_any_pipeline_depth_matches_eager(llama_dir, depth):
    """Review [P2] on #136: a fixed in-flight cap does not track
    `OverlapEngineCore(pipeline_depth=...)`, which accepts any depth >= 1. At a
    depth deeper than the cap, the position a later step still needs had already
    been trimmed."""
    from functools import partial

    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.overlap import OverlapEngineCore

    prompt = [list(range(11)), list(range(6, 19))]
    eager, _ = _generate(llama_dir, EngineCore, prompt, 12)
    overlapped, _ = _generate(
        llama_dir, partial(OverlapEngineCore, pipeline_depth=depth), prompt, 12
    )
    assert overlapped == eager


def test_the_buffer_holds_only_uncommitted_positions(llama_dir):
    """Trimming follows what the scheduler committed, so it cannot drop a
    position a deeper pipeline still needs, and it cannot grow with the request."""
    from kairyu.engine.core.engine_core import EngineCore

    _outputs, runner = _generate(llama_dir, EngineCore, [list(range(9))], 24)
    pending = runner._future_tokens["r0"]
    # eager commits every step, so at most the one just sampled remains
    assert len(pending) <= 2, pending
