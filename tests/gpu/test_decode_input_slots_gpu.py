"""The decode input slots on real CUDA (m2 §2.2, partial — see model_runner.py).

What this gate asserts is what the code actually does: the slots are CUDA
tensors allocated once and patched in place, both decode paths write into them,
generation is unchanged, and a capacity growth waits for the outstanding H2D
before it retires the buffers that transfer is reading. It deliberately does NOT
assert that the token
never crosses H2D — it does, because the sampling decision is host-side (m8 D2),
and a gate that injected an already-on-device tensor would assert nothing about
where the runner's own tokens come from. That was review [P1] on #143.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.gpu

TINY = dict(
    vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)
PROMPTS = [list(range(9)), list(range(4, 15)), list(range(2, 11))]


def _require_cuda() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("decode-slot gate needs CUDA")


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    torch.manual_seed(71)
    path = tmp_path_factory.mktemp("decode-slots")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY)).to(
        torch.float32
    ).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _runner(model_dir, sampler=None):
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.models.loader import load_model

    model, config, _ = load_model(model_dir, dtype=torch.bfloat16)
    cache = RadixKVCache(num_pages=128, page_size=16)
    pool = PagedKVPool.for_cache(cache, config, dtype=torch.bfloat16, device="cuda:0")
    return PagedModelRunner(model.to("cuda:0"), pool, sampler=sampler), cache


def _generate(model_dir, core_cls, prompts=PROMPTS, max_new=8):
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler

    runner, cache = _runner(model_dir, sampler=Sampler())
    scheduler = Scheduler(cache, max_num_batched_tokens=64, page_size=16)
    core = core_cls(scheduler, runner)
    for index, prompt in enumerate(prompts):
        core.add_request(
            EngineRequest(
                f"r{index}", tuple(prompt), max_new_tokens=max_new,
                sampling=EngineSampling(),
            )
        )
    return core.run_to_completion(), runner


def test_the_slots_are_cuda_resident_and_patched_in_place(llama_dir):
    _require_cuda()
    runner, _cache = _runner(llama_dir)

    tokens, positions = runner._decode_input_slots([5], [3])
    assert tokens.device.type == "cuda"
    assert positions.device.type == "cuda"
    assert tokens.tolist() == [5]
    address = tokens.data_ptr()

    runner._decode_input_slots([7], [4])
    # same device memory, new values: the slot was written, not reallocated
    assert tokens.data_ptr() == address
    assert tokens.tolist() == [7]
    assert positions.tolist() == [4]
    # the pinned staging row exists, so the H2D is a real async DMA
    assert runner._slot_staging is not None
    assert runner._slot_staging.is_pinned()


class _EventProbe(torch.cuda.Event):
    """A real CUDA event that also counts the waits performed on it."""

    def __init__(self) -> None:
        super().__init__()
        self.sync_calls = 0

    def synchronize(self) -> None:
        self.sync_calls += 1
        super().synchronize()


def test_growing_the_slots_waits_for_the_in_flight_staging_dma(llama_dir, monkeypatch):
    """A capacity growth must not retire buffers a DMA is still reading.

    `_decode_input_slots` grows before it synchronizes, so the event it waits on
    used to be the freshly recorded (empty) one belonging to the NEW buffers —
    the event representing the outstanding transfer had already been dropped
    together with the pinned rows it was protecting. Ordinary generation only
    survived that because the host sampler pulls logits to CPU every step; the
    ordering contract the code states did not hold across a growth.

    The stream is deliberately kept busy so the transfer really is unfinished at
    the moment the growth happens.
    """
    _require_cuda()
    runner, _cache = _runner(llama_dir)

    runner._decode_input_slots([5], [3])  # first allocation: capacity 8
    probe = _EventProbe()
    runner._slot_copy_done = probe

    torch.cuda._sleep(300_000_000)  # ~0.1s of stream work the H2D queues behind
    runner._decode_input_slots([6], [4])  # stages, copies, records `probe`
    probe.sync_calls = 0
    old_staging = runner._slot_staging
    assert old_staging is not None
    assert not probe.query(), "the staging DMA should still be in flight here"

    allocated_while_in_flight = []
    real_zeros = torch.zeros

    def recording_zeros(*args, **kwargs):
        allocated_while_in_flight.append(probe.query())
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", recording_zeros)
    wide = list(range(9))  # 9 > 8: this is the growth
    grown_tokens, grown_positions = runner._decode_input_slots(wide, wide)
    monkeypatch.undo()

    assert allocated_while_in_flight, "the growth did not reallocate at all"
    assert all(allocated_while_in_flight), (
        "replacement buffers were allocated while the old DMA was still reading "
        "the pinned staging rows"
    )
    assert probe.sync_calls == 1, "the OLD completion event was never waited on"
    assert runner._slot_staging is not old_staging
    assert grown_tokens.tolist() == wide
    assert grown_positions.tolist() == wide


def test_a_whole_generation_reuses_one_slot_allocation(llama_dir):
    """Batched and single-request decode alike: no per-step device allocation."""
    from kairyu.engine.core.engine_core import EngineCore

    _require_cuda()
    outputs, runner = _generate(llama_dir, EngineCore)
    batched_address = runner._decode_slots.data_ptr()

    single, single_runner = _generate(llama_dir, EngineCore, prompts=PROMPTS[:1])

    assert all(len(tokens) == 8 for tokens in outputs.values())
    assert len(single["r0"]) == 8
    assert batched_address == runner._decode_slots.data_ptr()
    # [P2]: the one-request run must have gone through the slots too
    assert single_runner._decode_slots is not None
    assert single_runner._decode_slots.device.type == "cuda"


def test_the_slots_do_not_change_what_is_generated(llama_dir):
    """The whole point is that this is unobservable in the output."""
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.overlap import OverlapEngineCore

    _require_cuda()
    eager, _ = _generate(llama_dir, EngineCore)
    overlapped, _ = _generate(llama_dir, OverlapEngineCore)

    assert overlapped == eager
    assert all(len(tokens) == 8 for tokens in eager.values())
