"""Real-CUDA coverage for the grammar-free batched sampler (#326)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kairyu.bench.profiling import profile_scope
from kairyu.engine.core.model_runner import PagedModelRunner, _PendingDeviceToken
from kairyu.engine.core.sampler import DeviceSample, DeviceSamplingInput, Sampler
from kairyu.engine.core.sampling_types import EngineSampling, SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("batched sampling requires CUDA")


def _sampling_rows(batch_size: int) -> tuple[DeviceSamplingInput, ...]:
    top_ks = (32, 48, 64)
    return tuple(
        DeviceSamplingInput(
            request_id=f"row-{index}",
            sampling=EngineSampling(
                temperature=0.7 + 0.01 * index,
                top_k=top_ks[index % len(top_ks)],
                top_p=0.9 + 0.01 * (index % 3),
                seed=326 + index,
            ),
            position=7 + index,
        )
        for index in range(batch_size)
    )


def test_batched_sampler_matches_scalar_and_uses_one_bounded_topk() -> None:
    batch_size = 16
    vocab_size = 151_936
    generator = torch.Generator(device="cuda").manual_seed(326)
    logits = torch.randn(
        (batch_size, vocab_size),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    rows = _sampling_rows(batch_size)
    sampler = Sampler()

    sampler.sample_batch_device(rows, logits)
    torch.cuda.synchronize()
    with profile_scope(
        enabled=True,
        activities=("cpu", "cuda"),
        acc_events=False,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as trace:
        batched = sampler.sample_batch_device(rows, logits)

    scalar = tuple(
        Sampler().sample_device(
            row.request_id,
            row.sampling,
            row.position,
            logits[index],
        )
        for index, row in enumerate(rows)
    )
    torch.cuda.synchronize()

    assert [int(sample.token_id.cpu()) for sample in batched] == [
        int(sample.token_id.cpu()) for sample in scalar
    ]
    events = {event.key: event.count for event in trace.key_averages()}
    assert events.get("aten::topk") == 1
    assert events.get("aten::sort", 0) == 0
    assert events.get("aten::_local_scalar_dense", 0) == 0
    assert not any("cudaStreamSynchronize" in key for key in events)


def test_batched_sampler_matches_scalar_for_mixed_processors() -> None:
    logits = torch.randn((5, 257), device="cuda")
    rows = (
        DeviceSamplingInput("report", EngineSampling(logprobs=3), 4),
        DeviceSamplingInput(
            "bounded",
            EngineSampling(temperature=0.73, top_k=31, top_p=0.87, seed=17),
            5,
        ),
        DeviceSamplingInput(
            "nucleus",
            EngineSampling(temperature=0.91, top_p=0.83, seed=23),
            6,
        ),
        DeviceSamplingInput(
            "forced",
            EngineSampling(temperature=0.8, forced_token_ids=(7,), seed=29),
            0,
        ),
        DeviceSamplingInput(
            "penalty",
            EngineSampling(
                temperature=0.67,
                top_k=47,
                top_p=0.92,
                repetition_penalty=1.15,
                presence_penalty=0.2,
                frequency_penalty=0.1,
                seed=31,
            ),
            3,
            prompt=(1, 2, 1),
            outputs=(2, 3),
            eos_token_id=4,
            stop_token_ids=(5,),
            min_tokens=4,
        ),
    )

    batched = Sampler().sample_batch_device(rows, logits)
    scalar = tuple(
        Sampler().sample_device(
            row.request_id,
            row.sampling,
            row.position,
            logits[index],
            prompt=row.prompt,
            outputs=row.outputs,
            eos_token_id=row.eos_token_id,
            stop_token_ids=row.stop_token_ids,
            min_tokens=row.min_tokens,
        )
        for index, row in enumerate(rows)
    )
    torch.cuda.synchronize()

    for batch_sample, row_sample in zip(batched, scalar, strict=True):
        assert int(batch_sample.token_id.cpu()) == int(row_sample.token_id.cpu())
        if row_sample.logprob is None:
            assert batch_sample.logprob is None
        else:
            assert batch_sample.logprob is not None
            assert float(batch_sample.logprob.cpu()) == pytest.approx(
                float(row_sample.logprob.cpu())
            )
        if row_sample.top_indices is None:
            assert batch_sample.top_indices is None
            assert batch_sample.top_logprobs is None
        else:
            assert batch_sample.top_indices is not None
            assert batch_sample.top_logprobs is not None
            assert torch.equal(batch_sample.top_indices, row_sample.top_indices)
            assert torch.allclose(
                batch_sample.top_logprobs,
                row_sample.top_logprobs,
            )


class _RoutingSampler:
    def __init__(self) -> None:
        self.batch_ids: tuple[str, ...] = ()
        self.batch_outputs = ()
        self.scalar_ids: list[str] = []

    def can_argmax_logits(self, *_args, **_kwargs) -> bool:
        return False

    def sample_batch_device(self, rows, logits):
        self.batch_ids = tuple(row.request_id for row in rows)
        self.batch_outputs = tuple(row.outputs for row in rows)
        return tuple(
            DeviceSample(torch.tensor(10 + index, device=logits.device))
            for index in range(len(rows))
        )

    def sample(self, request_id, *_args, **_kwargs) -> SampledToken:
        self.scalar_ids.append(request_id)
        return SampledToken(99)


def _state(request_id: str, sampling: EngineSampling):
    return SimpleNamespace(
        request=SimpleNamespace(
            request_id=request_id,
            sampling_identity=request_id,
            sampling=sampling,
            eos_token_id=1,
            prompt_token_ids=(1,),
            stop_token_ids=(),
            min_tokens=0,
        ),
        outputs=[2],
        output_epoch=0,
    )


def test_mixed_grammar_batch_only_routes_grammar_row_to_scalar_sampler() -> None:
    sampler = _RoutingSampler()
    runner = object.__new__(PagedModelRunner)
    runner._sampler = sampler
    runner._sampling_owner = True
    runner._future_tokens = {}
    runner._future_device_tokens = {}
    chunks = [
        ScheduledChunk("a", 1, False, 1),
        ScheduledChunk("b", 1, False, 1),
        ScheduledChunk("c", 1, False, 1),
    ]
    states = {
        "a": _state("a", EngineSampling(temperature=0.8, top_k=8)),
        "b": _state("b", EngineSampling(json_mode=True)),
        "c": _state("c", EngineSampling(logprobs=0)),
    }
    logits = torch.zeros((3, 32), device="cuda")

    records = runner._sample_rows(chunks, states, logits)

    assert sampler.batch_ids == ("a", "c")
    assert sampler.batch_outputs[0] is states["a"].outputs
    assert sampler.batch_outputs[1] is states["c"].outputs
    assert sampler.scalar_ids == ["b"]
    assert isinstance(records[0], _PendingDeviceToken)
    assert isinstance(records[1], SampledToken)
    assert isinstance(records[2], _PendingDeviceToken)
    assert int(records[0].sample.token_id.cpu()) == 10
    assert records[1].token_id == 99
    assert int(records[2].sample.token_id.cpu()) == 11
