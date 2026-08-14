"""The decode input slots on real CUDA (m2 §2.2, partial — see model_runner.py).

What this gate asserts is what the code actually does: the slots are CUDA
tensors allocated once and patched in place, both decode paths write into them,
generation is unchanged, and in-flight pinned staging rows are never reused or
waited on. Device-sampled token IDs bypass H2D and patch the same slots D2D.
"""

import pytest
import torch

from evals.profiling import profile_scope

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

    model, config, _ = load_model(
        model_dir, dtype=torch.bfloat16, target_device="cuda:0"
    )
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


def test_in_flight_staging_grows_the_pool_without_waiting(llama_dir):
    """An unfinished H2D gets another staging row, never Event.synchronize()."""
    _require_cuda()
    runner, _cache = _runner(llama_dir)

    runner._decode_input_slots([5], [3])  # first allocation: capacity 8
    old_staging = runner._slot_staging
    assert old_staging is not None
    probe = _EventProbe()
    torch.cuda._sleep(300_000_000)  # ~0.1s of stream work the H2D queues behind
    probe.record()
    runner._slot_staging_pool = [(old_staging, probe)]
    assert not probe.query(), "the staging DMA should still be in flight here"

    wide = list(range(9))
    grown_tokens, grown_positions = runner._decode_input_slots(wide, wide)

    assert probe.sync_calls == 0
    assert runner._slot_staging is not old_staging
    assert len(runner._slot_staging_pool) == 2
    assert grown_tokens.tolist() == wide
    assert grown_positions.tolist() == wide


def test_device_token_batch_uses_one_slot_update_kernel(llama_dir):
    """Thirty-two future-token views use one CUDA kernel and no device allocation."""
    _require_cuda()
    runner, _cache = _runner(llama_dir)
    packet = torch.arange(4 * 33, dtype=torch.long, device="cuda:0").reshape(
        4, 33
    )
    sources = [packet[index % 4, index + 1] for index in range(32)]
    expected = [(index % 4) * 33 + index + 1 for index in range(32)]
    positions = list(range(len(sources)))

    # Initialize the cat kernel and staging pool outside the measured range.
    runner._decode_input_slots(sources, positions)
    torch.cuda.synchronize()
    with profile_scope(
        enabled=True,
        activities=("cpu", "cuda"),
        acc_events=True,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        tokens, decoded_positions = runner._decode_input_slots(
            sources, positions
        )
    torch.cuda.synchronize()

    averages = {event.key: event for event in prof.key_averages()}
    assert averages["aten::stack"].count == 1
    # CPU staging plus the one position H2D are the only explicit copy_ calls.
    assert averages["aten::copy_"].count == 2
    cuda_events = [
        event
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    kernels = [
        event
        for event in cuda_events
        if not event.name.startswith(("Memcpy", "Memset"))
    ]
    assert len(kernels) == 1, [event.name for event in cuda_events]
    assert (
        sum(max(event.self_device_memory_usage, 0) for event in prof.events()) == 0
    )
    assert tokens.tolist() == expected
    assert decoded_positions.tolist() == positions


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
