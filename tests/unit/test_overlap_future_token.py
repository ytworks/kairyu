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


def _generate(model_dir: str, core_cls, prompt, max_new: int, **scheduler_kwargs):
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
                sampling=EngineSampling(),
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
    assert len(runner._future_tokens["r0"]) == 1


def test_release_drops_the_buffer(llama_dir):
    from kairyu.engine.core.engine_core import EngineCore

    _outputs, runner = _generate(llama_dir, EngineCore, [list(range(9))], 4)
    runner.release("r0")
    assert runner._future_tokens == {}
