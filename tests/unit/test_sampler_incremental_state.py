"""Incremental sampler-penalty state parity and lifecycle (issue #216)."""

from __future__ import annotations

import itertools

import pytest
import torch

from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import EngineSampling, mix_seed


def _legacy_penalties(
    logits: torch.Tensor,
    sampling: EngineSampling,
    prompt: tuple[int, ...],
    outputs: tuple[int, ...],
) -> torch.Tensor:
    """The exact pre-#216 full-history implementation used as an oracle."""
    work = logits.detach().to(device="cpu", dtype=torch.float32).clone()
    if sampling.repetition_penalty != 1.0:
        seen = torch.tensor(
            sorted(set(prompt) | set(outputs)),
            dtype=torch.long,
        )
        if seen.numel():
            values = work[seen]
            work[seen] = torch.where(
                values > 0,
                values / sampling.repetition_penalty,
                values * sampling.repetition_penalty,
            )
    if outputs and (
        sampling.presence_penalty != 0.0
        or sampling.frequency_penalty != 0.0
    ):
        counts = torch.bincount(
            torch.tensor(outputs, dtype=torch.long),
            minlength=work.shape[-1],
        ).to(work.dtype)
        work -= sampling.frequency_penalty * counts
        work -= sampling.presence_penalty * (counts > 0).to(work.dtype)
    return work


_PENALTY_COMBINATIONS = tuple(
    EngineSampling(
        temperature=temperature,
        seed=123,
        repetition_penalty=repetition,
        presence_penalty=presence,
        frequency_penalty=frequency,
    )
    for temperature, repetition, presence, frequency in itertools.product(
        (0.0, 0.8),
        (0.5, 1.0, 1.7),
        (-1.25, 0.0, 0.75),
        (-0.8, 0.0, 1.1),
    )
    if (
        repetition != 1.0
        or presence != 0.0
        or frequency != 0.0
    )
)


@pytest.mark.parametrize("sampling", _PENALTY_COMBINATIONS)
def test_incremental_penalties_match_full_history_logits_and_samples(
    sampling: EngineSampling,
) -> None:
    prompt = (1, 2, 1, 7)
    histories = (
        (),
        (3,),
        (3, 3),
        (3, 3, 5),
        (3, 3, 5, 1),
        # Exceptional rollback: discard the last two committed values.
        (3, 3),
        (3, 3, 8),
        (3, 3, 8, 8),
    )
    sampler = Sampler()
    state = sampler._state_for("request", sampling)

    for position, outputs in enumerate(histories):
        logits = torch.randn(
            32,
            generator=torch.Generator().manual_seed(10_000 + position),
        )
        expected = _legacy_penalties(logits, sampling, prompt, outputs)
        actual = logits.clone()
        sampler._apply_incremental_penalties(
            state,
            actual,
            sampling,
            prompt,
            outputs,
        )
        assert torch.equal(actual, expected)

        sampled = sampler.sample(
            "request",
            sampling,
            position,
            logits,
            prompt=prompt,
            outputs=outputs,
        )
        if sampling.temperature == 0.0:
            expected_token = int(torch.argmax(expected).item())
        else:
            expected_token = Sampler._sample_scaled(
                expected,
                sampling,
                state.base_seed,
                position,
            )
        assert sampled.token_id == expected_token
        assert mix_seed(state.base_seed, position) == mix_seed(123, position)


def test_pending_overlay_commit_and_rollback_update_only_live_tokens() -> None:
    sampling = EngineSampling(
        repetition_penalty=1.4,
        presence_penalty=0.5,
        frequency_penalty=0.75,
    )
    sampler = Sampler()
    state = sampler._state_for("overlap", sampling)
    logits = torch.zeros(16)
    first = torch.tensor(4)
    rejected = torch.tensor(6)

    sampler._apply_incremental_penalties(
        state,
        logits.clone(),
        sampling,
        (1, 2),
        (),
        (first, rejected),
    )
    penalties = next(iter(state.penalty_states.values()))
    assert penalties.output_counts.tolist() == pytest.approx(
        [0, 0, 0, 0, 1, 0, 1] + [0] * 9
    )

    # The first pending token commits; the second is rejected and replaced by
    # authoritative token 9. Neither device scalar is read back to compare it.
    sampler._apply_incremental_penalties(
        state,
        logits.clone(),
        sampling,
        (1, 2),
        (4, 9),
    )
    expected = [0.0] * 16
    expected[4] = 1.0
    expected[9] = 1.0
    assert penalties.output_counts.tolist() == pytest.approx(expected)
    assert penalties.pending == ()

    # A restart rolls committed history back to its accepted prefix.
    sampler._apply_incremental_penalties(
        state,
        logits.clone(),
        sampling,
        (1, 2),
        (4,),
    )
    expected[9] = 0.0
    assert penalties.output_counts.tolist() == pytest.approx(expected)


def test_duplicate_pending_overlap_matches_one_full_count_operation() -> None:
    sampling = EngineSampling(
        repetition_penalty=1.2,
        presence_penalty=0.4,
        frequency_penalty=0.7,
    )
    prompt = (1, 2)
    outputs = (3, 3, 4)
    pending = (torch.tensor(3), torch.tensor(3))
    logits = torch.randn(16, generator=torch.Generator().manual_seed(11))
    expected = _legacy_penalties(
        logits,
        sampling,
        prompt,
        outputs + (3, 3),
    )
    sampler = Sampler()
    state = sampler._state_for("duplicate-pending", sampling)
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


def test_versioned_history_commits_pending_without_recounting() -> None:
    sampling = EngineSampling(
        repetition_penalty=1.4,
        presence_penalty=0.5,
        frequency_penalty=0.75,
    )
    sampler = Sampler()
    state = sampler._state_for("versioned", sampling)
    logits = torch.zeros(16)
    outputs: list[int] = []
    first = torch.tensor(4)
    second = torch.tensor(6)

    sampler._apply_incremental_penalties(
        state,
        logits.clone(),
        sampling,
        (1, 2),
        outputs,
        (first,),
        history_epoch=0,
    )
    penalties = next(iter(state.penalty_states.values()))
    outputs.append(4)
    sampler._apply_incremental_penalties(
        state,
        logits.clone(),
        sampling,
        (1, 2),
        outputs,
        (second,),
        history_epoch=0,
    )

    expected = [0.0] * 16
    expected[4] = 1.0
    expected[6] = 1.0
    assert penalties.output_counts.tolist() == pytest.approx(expected)
    assert penalties.committed == [4]
    assert penalties.pending == (second,)


def test_history_correction_is_exact_with_comparison_or_epoch() -> None:
    sampling = EngineSampling(
        repetition_penalty=1.2,
        presence_penalty=0.3,
        frequency_penalty=0.7,
    )
    logits = torch.zeros(16)

    for versioned in (False, True):
        sampler = Sampler()
        state = sampler._state_for(f"correction-{versioned}", sampling)
        outputs = [4, 9]
        kwargs = {"history_epoch": 0} if versioned else {}
        sampler._apply_incremental_penalties(
            state,
            logits.clone(),
            sampling,
            (1, 2),
            outputs,
            **kwargs,
        )
        outputs[1] = 8
        if versioned:
            kwargs["history_epoch"] = 1
        sampler._apply_incremental_penalties(
            state,
            logits.clone(),
            sampling,
            (1, 2),
            outputs,
            **kwargs,
        )

        penalties = next(iter(state.penalty_states.values()))
        expected = [0.0] * 16
        expected[4] = 1.0
        expected[8] = 1.0
        assert penalties.output_counts.tolist() == pytest.approx(expected)
        assert penalties.committed == [4, 8]


def test_handoff_moves_incremental_state_and_syncs_carried_token() -> None:
    sampling = EngineSampling(
        repetition_penalty=1.2,
        frequency_penalty=0.5,
    )
    source = Sampler()
    destination = Sampler()
    logits = torch.zeros(16)

    source.sample(
        "request",
        sampling,
        0,
        logits,
        prompt=(1, 2, 1),
        outputs=(),
    )
    source_state = source._states["request"]
    penalties = next(iter(source_state.penalty_states.values()))

    assert source.hand_over("request", destination)
    assert "request" not in source._states
    assert destination._states["request"] is source_state
    destination.sample(
        "request",
        sampling,
        1,
        logits,
        prompt=(1, 2, 1),
        outputs=(7,),
    )
    assert penalties.output_counts[7].item() == 1.0
    assert penalties.repetition_seen[1].item()
    assert penalties.repetition_seen[2].item()
    assert penalties.repetition_seen[7].item()

    destination.release("request")
    assert "request" not in destination._states


def test_steady_history_reuses_dense_penalty_storage() -> None:
    sampling = EngineSampling(
        repetition_penalty=1.1,
        presence_penalty=0.2,
        frequency_penalty=0.3,
    )
    prompt = tuple(range(128))
    outputs = tuple(index % 512 for index in range(32_768))
    sampler = Sampler()
    logits = torch.zeros(1024)

    sampler.sample(
        "long",
        sampling,
        len(outputs),
        logits,
        prompt=prompt,
        outputs=outputs,
    )
    penalties = next(
        iter(sampler._states["long"].penalty_states.values())
    )
    counts_storage = penalties.output_counts.data_ptr()
    seen_storage = penalties.repetition_seen.data_ptr()

    for position in range(64):
        sampler.sample(
            "long",
            sampling,
            len(outputs) + position,
            logits,
            prompt=prompt,
            outputs=outputs,
        )

    assert penalties.output_counts.data_ptr() == counts_storage
    assert penalties.repetition_seen.data_ptr() == seen_storage
    assert len(penalties.committed) == len(outputs)
