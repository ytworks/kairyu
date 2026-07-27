"""CUDA parity for incremental sampler penalty state (issue #216)."""

from __future__ import annotations

import pytest
import torch

from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import EngineSampling

pytestmark = pytest.mark.gpu


def _legacy_device_penalties(
    logits: torch.Tensor,
    sampling: EngineSampling,
    prompt: tuple[int, ...],
    outputs: tuple[int, ...],
    pending: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    work = logits.clone()
    device = logits.device
    vocab = logits.shape[-1]
    seen = torch.zeros(vocab, dtype=torch.bool, device=device)
    host_seen = tuple(sorted(set(prompt) | set(outputs)))
    if host_seen:
        seen.scatter_(
            0,
            torch.as_tensor(host_seen, dtype=torch.long, device=device),
            True,
        )
    for token in pending:
        seen.scatter_(0, token.to(dtype=torch.long).view(1), True)
    work.copy_(
        torch.where(
            seen,
            torch.where(
                work > 0,
                work / sampling.repetition_penalty,
                work * sampling.repetition_penalty,
            ),
            work,
        )
    )

    counts = torch.zeros(vocab, dtype=torch.float32, device=device)
    if outputs:
        indices = torch.as_tensor(
            outputs,
            dtype=torch.long,
            device=device,
        )
        counts.scatter_add_(
            0,
            indices,
            torch.ones_like(indices, dtype=torch.float32),
        )
    for token in pending:
        index = token.to(dtype=torch.long).view(1)
        counts.scatter_add_(0, index, torch.ones(1, device=device))
    work.sub_(sampling.frequency_penalty * counts)
    work.sub_(
        sampling.presence_penalty * (counts > 0).to(work.dtype)
    )
    return work


def test_cuda_incremental_pending_and_rollback_match_legacy() -> None:
    if not torch.cuda.is_available():
        pytest.skip("incremental sampler CUDA gate needs a GPU")
    sampling = EngineSampling(
        repetition_penalty=1.3,
        presence_penalty=-0.4,
        frequency_penalty=0.8,
    )
    prompt = (1, 2, 1, 11)
    sampler = Sampler()
    state = sampler._state_for("cuda-request", sampling)
    logits = torch.randn(128, device="cuda")
    first = torch.tensor(7, device="cuda")
    rejected = torch.tensor(9, device="cuda")

    histories = (
        ((), (first, rejected)),
        ((7,), (rejected,)),
        ((7, 13), ()),
        ((7,), ()),
    )
    for outputs, pending in histories:
        expected = _legacy_device_penalties(
            logits,
            sampling,
            prompt,
            outputs,
            pending,
        )
        actual = logits.clone()
        sampler._apply_incremental_penalties(
            state,
            actual,
            sampling,
            prompt,
            outputs,
            pending,
        )
        assert torch.equal(actual, expected)

    penalties = next(iter(state.penalty_states.values()))
    counts_storage = penalties.output_counts.data_ptr()
    sampler._apply_incremental_penalties(
        state,
        logits.clone(),
        sampling,
        prompt,
        (7,),
    )
    assert penalties.output_counts.data_ptr() == counts_storage


def test_cuda_duplicate_pending_overlap_matches_legacy() -> None:
    if not torch.cuda.is_available():
        pytest.skip("incremental sampler CUDA gate needs a GPU")
    sampling = EngineSampling(
        repetition_penalty=1.2,
        presence_penalty=0.4,
        frequency_penalty=0.7,
    )
    prompt = (1, 2)
    outputs = (3, 3, 4)
    pending = (
        torch.tensor(3, device="cuda"),
        torch.tensor(3, device="cuda"),
    )
    logits = torch.randn(
        16,
        generator=torch.Generator(device="cuda").manual_seed(11),
        device="cuda",
    )
    expected = _legacy_device_penalties(
        logits,
        sampling,
        prompt,
        outputs,
        pending,
    )
    sampler = Sampler()
    state = sampler._state_for("cuda-duplicate", sampling)
    actual = logits.clone()

    sampler._apply_incremental_penalties(
        state,
        actual,
        sampling,
        prompt,
        outputs,
        pending,
        history_epoch=0,
    )

    assert torch.equal(actual, expected)


def test_cuda_handoff_reclaims_the_old_device_row() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("cross-device sampler handoff needs two GPUs")
    sampling = EngineSampling(frequency_penalty=0.5)
    source = Sampler()
    destination = Sampler()

    source.sample_device(
        "handoff",
        sampling,
        0,
        torch.zeros(64, device="cuda:0"),
    )
    state = source._states["handoff"]
    assert next(iter(state.penalty_states))[1] == 0
    assert source.hand_over("handoff", destination)

    destination.sample_device(
        "handoff",
        sampling,
        1,
        torch.zeros(64, device="cuda:1"),
        outputs=(3,),
    )
    assert len(state.penalty_states) == 1
    assert next(iter(state.penalty_states))[1] == 1
