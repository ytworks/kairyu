"""The decode input slots are persistent device tensors, patched in place.

Review [P2] on #143: the device-input mechanism was reached only from
`_execute_decode_batch` (`len(decodes) >= 2`). Single-request generation — and
the tail of every workload, once the running set drops to one — still built
`torch.tensor([input_token], device=...)` from the host on every step. These
tests pin the mechanism itself and pin that BOTH decode paths use it.

Aliasing is what makes the assertions falsifiable: a view handed out at step N
reads step N+1's values only if the slot was written in place. A path that
allocates a fresh tensor per step cannot make that true, however the allocator
happens to reuse addresses.
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
    path = tmp_path_factory.mktemp("decode-slots-llama")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _runner(model_dir, **kwargs):
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.models.loader import load_model

    model, config, _ = load_model(model_dir)
    cache = RadixKVCache(num_pages=64, page_size=4)
    return PagedModelRunner(model, PagedKVPool.for_cache(cache, config), **kwargs)


def _generate(model_dir, core_cls, prompts, max_new: int):
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    model, config, _ = load_model(model_dir)
    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(cache, max_num_batched_tokens=6, page_size=4)
    runner = PagedModelRunner(model, PagedKVPool.for_cache(cache, config), sampler=Sampler())
    core = core_cls(scheduler, runner)
    for index, tokens in enumerate(prompts):
        core.add_request(
            EngineRequest(
                f"r{index}", tuple(tokens), max_new_tokens=max_new,
                sampling=EngineSampling(),
            )
        )
    return core.run_to_completion(), runner


def test_the_slots_are_written_in_place_not_rebuilt(llama_dir):
    """The mechanism, directly: step N's view reads step N+1's values."""
    runner = _runner(llama_dir)

    tokens, positions = runner._decode_input_slots([5], [3])
    assert tokens.tolist() == [5]
    assert positions.tolist() == [3]
    address = tokens.data_ptr()

    later_tokens, later_positions = runner._decode_input_slots([7], [4])
    assert later_tokens.data_ptr() == address
    # the earlier view now reads the new step: same memory, patched in place
    assert tokens.tolist() == [7]
    assert positions.tolist() == [4]
    assert later_tokens.dtype == torch.long and later_positions.dtype == torch.long


def test_the_slots_are_allocated_once_across_batch_sizes(llama_dir):
    """A shrinking batch must reuse the slots, not reallocate per step."""
    runner = _runner(llama_dir)

    runner._decode_input_slots([1, 2, 3], [0, 1, 2])
    address = runner._decode_slots.data_ptr()
    tokens, positions = runner._decode_input_slots([9], [5])

    assert runner._decode_slots.data_ptr() == address
    assert tokens.tolist() == [9] and positions.tolist() == [5]
    # a larger batch than the capacity grows once, and the values still land
    wide = list(range(20))
    grown, _ = runner._decode_input_slots(wide, wide)
    assert grown.tolist() == wide


def _capture_decode_input(runner, method: str):
    """Record the tensor the model was handed, without disturbing the forward."""
    seen = {}
    original = getattr(runner._model, method)

    def recording(token_ids, positions, *args, **kwargs):
        seen.setdefault("token_ids", token_ids)
        seen.setdefault("positions", positions)
        return original(token_ids, positions, *args, **kwargs)

    setattr(runner._model, method, recording)
    return seen


class _Fake:
    """Duck-typed stand-in for whatever the runner reads off the state."""

    def __init__(self, **fields) -> None:
        self.__dict__.update(fields)


def _decode_state(request_id: str, prompt, outputs, pages):
    """The exact slice of state the runner's access contract names."""
    return _Fake(
        request=_Fake(
            request_id=request_id,
            prompt_token_ids=tuple(prompt),
            sampling=None,
            eos_token_id=None,
        ),
        allocation=_Fake(pages=tuple(pages), num_cached_tokens=0),
        decode_pages=(),
        outputs=tuple(outputs),
    )


def test_the_single_request_decode_reads_the_shared_slots(llama_dir):
    """[P2]: `_execute_decode` used to rebuild its input from the host each step.

    The captured tensor is proven to BE the slot by writing the next step into
    the slots and reading the capture back.
    """
    from kairyu.engine.core.scheduler import ScheduledChunk

    runner = _runner(llama_dir)
    seen = _capture_decode_input(runner, "forward_tokens")

    state = _decode_state("a", (1, 2, 3), (41,), pages=[0, 1])
    chunk = ScheduledChunk(request_id="a", num_tokens=1, is_prefill=False, position=1)
    runner._execute_decode(chunk, state, {})

    assert seen["token_ids"].tolist() == [41]
    runner._decode_input_slots([123], [77])
    assert seen["token_ids"].tolist() == [123], "the single decode built its own tensor"
    assert seen["positions"].tolist() == [77]


def test_the_batched_decode_reads_the_same_slots(llama_dir):
    """Both paths share one buffer, so neither is a fallback for the other."""
    from kairyu.engine.core.scheduler import ScheduledChunk

    runner = _runner(llama_dir)
    seen = _capture_decode_input(runner, "forward_decode_tensors")

    states = {
        "a": _decode_state("a", (1, 2, 3), (41,), pages=[0, 1]),
        "b": _decode_state("b", (4, 5, 6), (42,), pages=[2, 3]),
    }
    chunks = [
        ScheduledChunk(request_id="a", num_tokens=1, is_prefill=False, position=1),
        ScheduledChunk(request_id="b", num_tokens=1, is_prefill=False, position=1),
    ]
    runner._execute_decode_batch(chunks, states, {})

    assert seen["token_ids"].tolist() == [41, 42]
    runner._decode_input_slots([123, 124], [77, 78])
    assert seen["token_ids"].tolist() == [123, 124], "the batched decode built its own tensor"
    assert seen["positions"].tolist() == [77, 78]


def test_every_decode_step_of_a_single_request_run_reuses_one_slot(llama_dir):
    """End to end: a one-request generation must not allocate per step."""
    from kairyu.engine.core.engine_core import EngineCore

    outputs, runner = _generate(llama_dir, EngineCore, [list(range(9))], 6)

    assert len(outputs["r0"]) == 6
    assert runner._decode_slots is not None, "the single-decode path never used the slots"
    assert runner._decode_slots.numel() >= 1


def test_the_slots_do_not_change_what_is_generated(llama_dir):
    """Overlap and eager still agree, single-request and batched."""
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.overlap import OverlapEngineCore

    for prompts in ([list(range(11))], [list(range(9)), list(range(3, 14))]):
        eager, _ = _generate(llama_dir, EngineCore, prompts, 6)
        overlapped, _ = _generate(llama_dir, OverlapEngineCore, prompts, 6)
        assert overlapped == eager
        assert all(len(tokens) == 6 for tokens in eager.values())
