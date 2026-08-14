"""GPU coverage for attention-DP's non-blocking fast status protocol."""

from types import SimpleNamespace

import pytest
import torch

from kairyu.engine.core import worker as worker_module
from kairyu.engine.core.sampling_types import EngineSampling
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.step_input import RequestSnapshot, StepDelta
from verification.l1.performance.profiling import profile_scope

pytestmark = pytest.mark.gpu


def test_attention_dp_fast_status_does_not_synchronize_cuda_event() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("attention-DP fast status profiler gate needs CUDA")

    device = torch.device("cuda")
    chunk = ScheduledChunk("r", 1, False, 0)
    state = RequestSnapshot(
        request_id="r",
        prompt_token_ids=(1,),
        computed_prompt=1,
        outputs=(),
        in_flight=0,
        page_ids=(0,),
        decode_page_ids=(),
        eos_token_id=None,
        max_new_tokens=2,
        num_cached_tokens=0,
        sampling=EngineSampling(),
    )
    gathered = torch.tensor(
        (worker_module._EP_ATTENTION_DP_PACKET_OK, 7),
        dtype=torch.int64,
        device=device,
    )
    runner = SimpleNamespace(_output_copy_stream=torch.cuda.Stream())
    reductions: list[tuple[float, ...]] = []

    def all_reduce(values):
        reductions.append(values)
        return (0.0,)

    control_comm = SimpleNamespace(
        rank=0,
        world_size=1,
        all_reduce=all_reduce,
        all_gather=lambda _reply: pytest.fail(
            "successful fast status must not gather Python objects"
        ),
    )
    reply = worker_module._EPAttentionDPReply(rank=0, sampled={})

    # Warm allocations outside the steady measurement.
    warm = worker_module._ep_attention_dp_deferred_packet_output(
        gathered,
        chunks=(chunk,),
        states={"r": state},
        owners=(("r", 0),),
        world_size=1,
        local_runner=runner,
    )
    assert warm["r"] == (worker_module.SampledToken(7),)
    reductions.clear()
    torch.cuda.synchronize()

    # Keep the producer stream busy so an accidental event resolution is both
    # observable as cudaEventSynchronize and expensive enough not to race away.
    torch.cuda._sleep(20_000_000)
    with profile_scope(
        enabled=True,
        activities=("cpu", "cuda"),
        acc_events=True,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        output = worker_module._ep_attention_dp_deferred_packet_output(
            gathered,
            chunks=(chunk,),
            states={"r": state},
            owners=(("r", 0),),
            world_size=1,
            local_runner=runner,
        )
        worker_module._ep_attention_dp_fast_status(control_comm, reply)

    keys = {event.key for event in prof.key_averages()}
    assert not any("cudaEventSynchronize" in key for key in keys), sorted(keys)
    assert reductions == [(0.0,)]
    # Public token materialization remains deferred and resolves outside the
    # measured status path.
    assert output["r"] == (worker_module.SampledToken(7),)


def test_attention_dp_packet_status_does_not_synchronize_the_cuda_stream() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("attention-DP packet profiler gate needs CUDA")

    device = torch.device("cuda")
    chunk = ScheduledChunk("r", 1, False, 0)
    state = RequestSnapshot(
        request_id="r",
        prompt_token_ids=(1,),
        computed_prompt=1,
        outputs=(),
        in_flight=0,
        page_ids=(0,),
        decode_page_ids=(),
        eos_token_id=None,
        max_new_tokens=2,
        num_cached_tokens=0,
        sampling=EngineSampling(),
    )
    payload = worker_module._EPAttentionDPStep(
        StepDelta(
            chunks=(chunk,),
            new=(state,),
            updates=(),
            dropped=(),
        ),
        owners=(("r", 0),),
    )
    model_comm = SimpleNamespace(rank=0, world_size=4)
    runner = SimpleNamespace(_device=device)
    token = torch.empty((), dtype=torch.int64, device=device)
    token.fill_(7)

    def build_packet():
        sampled = {
            "r": (
                SimpleNamespace(
                    sample=SimpleNamespace(token_id=token),
                ),
            )
        }
        return worker_module._ep_attention_dp_local_token_packet(
            model_comm,
            runner,
            payload,
            {"r": state},
            sampled,
            dummy_ids=(),
            failed=False,
        )

    # Initialize the allocator and kernels outside the measured steady path.
    build_packet()
    torch.cuda.synchronize()
    with profile_scope(
        enabled=True,
        activities=("cpu", "cuda"),
        acc_events=True,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        packet = build_packet()
        torch.cuda.synchronize()

    keys = {event.key for event in prof.key_averages()}
    assert not any("cudaStreamSynchronize" in key for key in keys), sorted(keys)
    assert packet.cpu().tolist() == [
        worker_module._EP_ATTENTION_DP_PACKET_OK,
        7,
    ]
