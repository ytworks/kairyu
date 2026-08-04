"""Sampler behaviors pinned by design m8 D2 (order, determinism, grammar)."""

import math

import pytest
import torch

from kairyu.engine.core.sampler import Sampler, mix_seed, stable_request_seed
from kairyu.engine.core.sampling_types import EngineSampling
from kairyu.kernels import sampling_gpu
from kairyu.kernels.sampling_gpu import stateless_gumbel_argmax

VOCAB = 32


def _logits(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, generator=generator)


def test_temperature_zero_is_exact_argmax():
    sampler = Sampler()
    logits = _logits()
    token = sampler.sample("r", EngineSampling(temperature=0.0), 0, logits)
    assert token.token_id == int(torch.argmax(logits).item())


def test_only_plain_greedy_without_logprobs_can_argmax_before_host_copy():
    sampler = Sampler()

    assert sampler.can_argmax_logits("plain", EngineSampling())
    assert not sampler.can_argmax_logits(
        "logprobs", EngineSampling(logprobs=0)
    )
    assert not sampler.can_argmax_logits(
        "penalty", EngineSampling(repetition_penalty=1.1)
    )
    assert not sampler.can_argmax_logits(
        "stochastic", EngineSampling(temperature=0.5)
    )
    assert not sampler.can_argmax_logits(
        "grammar", EngineSampling(json_mode=True), eos_token_id=1
    )


def test_min_tokens_mask_blocks_fast_argmax_until_logical_position_is_reached():
    sampler = Sampler()
    sampling = EngineSampling()

    assert not sampler.can_argmax_logits(
        "held",
        sampling,
        eos_token_id=7,
        position=1,
        stop_token_ids=(8,),
        min_tokens=2,
    )
    assert sampler.can_argmax_logits(
        "released",
        sampling,
        eos_token_id=7,
        position=2,
        stop_token_ids=(8,),
        min_tokens=2,
    )


def test_min_tokens_masks_eos_and_stop_ids_before_selection_but_not_raw_logprobs():
    logits = torch.zeros(VOCAB)
    logits[3] = 5.0
    logits[7] = 10.0  # EOS would win without the processor.
    logits[8] = 9.0  # An explicit stop ID must be masked too.
    raw = torch.log_softmax(logits, dim=-1)
    sampler = Sampler()
    sampling = EngineSampling(logprobs=0)

    for position in (0, 1):
        token = sampler.sample(
            "held",
            sampling,
            position,
            logits,
            eos_token_id=7,
            stop_token_ids=(8, 7, 8),
            min_tokens=2,
        )
        assert token.token_id == 3
        assert token.logprob == pytest.approx(float(raw[3]))

    released = sampler.sample(
        "released",
        sampling,
        2,
        logits,
        eos_token_id=7,
        stop_token_ids=(8,),
        min_tokens=2,
    )
    assert released.token_id == 7


def test_seeded_sampling_never_selects_a_masked_stop_id():
    logits = torch.full((VOCAB,), -20.0)
    logits[3] = 0.0
    logits[7] = 100.0
    logits[8] = 90.0

    token = Sampler().sample(
        "sampled",
        EngineSampling(temperature=1.0, seed=3),
        0,
        logits,
        eos_token_id=7,
        stop_token_ids=(8,),
        min_tokens=1,
    )

    assert token.token_id not in (7, 8)


def test_same_seed_same_tokens():
    logits = _logits()
    sampling = EngineSampling(temperature=1.0, seed=7)
    a = Sampler().sample("r1", sampling, 3, logits)
    b = Sampler().sample("r2", sampling, 3, logits)  # explicit seed wins over request id
    assert a.token_id == b.token_id


def test_default_seed_is_stable_per_request_id():
    logits = _logits()
    sampling = EngineSampling(temperature=1.0)
    a = Sampler().sample("req-x", sampling, 0, logits)
    b = Sampler().sample("req-x", sampling, 0, logits)
    assert a.token_id == b.token_id


def test_different_positions_can_differ():
    logits = torch.zeros(VOCAB)  # uniform: position seeds decide
    sampling = EngineSampling(temperature=1.0, seed=1)
    sampler = Sampler()
    tokens = {sampler.sample("r", sampling, pos, logits).token_id for pos in range(16)}
    assert len(tokens) > 1


def test_seed_mixing_not_additive():
    # splitmix: (base=1,pos=2) must not collide with (base=2,pos=1)
    assert mix_seed(1, 2) != mix_seed(2, 1)
    assert stable_request_seed("a") != stable_request_seed("b")


def test_stateless_gumbel_cpu_has_fixed_golden_draws():
    log_weights = torch.tensor([-2.5, -1.0, 0.0, 0.5, -0.25, 1.0, -4.0, 0.25])
    seeds = (0, 1, 2, 3, 7, 71, 0x12345678, 0x89ABCDEF, 0xFFFFFFFF)

    actual = [
        int(stateless_gumbel_argmax(log_weights, seed).item()) for seed in seeds
    ]

    assert actual == [0, 3, 3, 5, 4, 5, 2, 5, 1]


@pytest.mark.parametrize(
    "sampling",
    [
        EngineSampling(temperature=0.7),
        EngineSampling(temperature=0.7, min_p=0.2),
        EngineSampling(temperature=0.7, top_k=5),
        EngineSampling(temperature=0.7, top_p=0.65),
        EngineSampling(temperature=1.3, min_p=0.05, top_k=7, top_p=0.8),
    ],
    ids=("plain", "min-p", "top-k", "top-p", "combined"),
)
def test_cpu_scaled_wrappers_share_one_draw_across_seed_and_filter_matrix(sampling):
    logits = torch.tensor(
        [1.7, -0.4, 0.2, 2.3, -1.8, 0.9, 1.1, -0.7, 0.5, 1.9, -2.2]
    )

    for base_seed in (0, 1, 71, (1 << 63) - 1):
        for position in (0, 1, 7, 31):
            tensor_token = Sampler._sample_scaled_device(
                logits, sampling, base_seed, position
            )
            scalar_token = Sampler._sample_scaled(
                logits, sampling, base_seed, position
            )

            assert tensor_token.device.type == "cpu"
            assert tensor_token.dtype == torch.int64
            assert scalar_token == int(tensor_token.item())


def test_cpu_public_sample_matches_the_canonical_gumbel_helper():
    logits = torch.tensor([-1.2, 0.3, 1.8, -0.7, 0.9, 1.1, -2.0, 0.1])
    sampling = EngineSampling(temperature=0.83, seed=353)

    for position in (0, 1, 4, 19):
        probabilities = torch.softmax(logits / sampling.temperature, dim=-1)
        expected = stateless_gumbel_argmax(
            torch.log(probabilities),
            mix_seed(sampling.seed, position),
        )
        actual = Sampler().sample("request", sampling, position, logits)

        assert actual.token_id == int(expected.item())


def test_cpu_scaled_sampling_falls_back_deterministically_for_degenerate_weights():
    sampling = EngineSampling(temperature=0.8, min_p=0.3, top_k=3, top_p=0.5)
    unfiltered = EngineSampling(temperature=0.8)
    degenerate = torch.full((8,), float("-inf"))
    only_one_legal = torch.tensor(
        [float("-inf"), float("-inf"), 3.0, float("-inf"), float("-inf")]
    )

    for position in (0, 5, 17):
        # With no filter, softmax(-inf, ...) leaves a NaN total. The CPU fast
        # path must reject it and retain the same deterministic fallback.
        assert Sampler._sample_scaled(degenerate, unfiltered, 71, position) == 0
        assert Sampler._sample_scaled(degenerate, sampling, 71, position) == 0
        assert int(
            Sampler._sample_scaled_device(degenerate, sampling, 71, position).item()
        ) == 0
        assert Sampler._sample_scaled(only_one_legal, sampling, 71, position) == 2
        assert int(
            Sampler._sample_scaled_device(only_one_legal, sampling, 71, position).item()
        ) == 2


def test_cpu_offset_cache_is_reused_bounded_and_never_mutated(monkeypatch):
    cache = sampling_gpu._cached_cpu_offsets
    cache.cache_clear()
    first = cache(17)
    pristine = first.clone()

    stateless_gumbel_argmax(torch.zeros(17), 353)

    assert cache(17) is first
    assert torch.equal(first, pristine)
    for size in (18, 19, 20, 21):
        cache(size)
    info = cache.cache_info()
    assert info.maxsize == 4
    assert info.currsize == 4

    calls: list[int] = []

    def recording_cache(size: int) -> torch.Tensor:
        calls.append(size)
        return cache(size)

    monkeypatch.setattr(sampling_gpu, "_cached_cpu_offsets", recording_cache)
    oversized = torch.zeros(sampling_gpu._MAX_CACHED_CPU_OFFSETS + 1)
    stateless_gumbel_argmax(oversized, 353)
    assert calls == []


def test_top_k_one_is_greedy():
    logits = _logits()
    token = Sampler().sample("r", EngineSampling(temperature=1.0, top_k=1), 0, logits)
    assert token.token_id == int(torch.argmax(logits).item())


def test_top_p_keeps_at_least_the_top_token():
    logits = _logits()
    token = Sampler().sample("r", EngineSampling(temperature=1.0, top_p=1e-9), 0, logits)
    assert token.token_id == int(torch.argmax(logits).item())


def test_min_p_filters_tail():
    logits = torch.tensor([10.0, 9.9] + [0.0] * (VOCAB - 2))
    sampling = EngineSampling(temperature=1.0, min_p=0.5, seed=5)
    sampler = Sampler()
    for pos in range(20):
        token = sampler.sample("r", sampling, pos, logits)
        assert token.token_id in (0, 1)


def test_repetition_penalty_covers_prompt_and_outputs():
    logits = torch.zeros(VOCAB)
    logits[3] = 5.0  # would win greedily
    logits[7] = 1.0  # runner-up: wins once 3 is penalized below it
    sampling = EngineSampling(temperature=0.0, repetition_penalty=100.0)
    token = Sampler().sample("r", sampling, 0, logits, prompt=(3,), outputs=())
    assert token.token_id == 7  # penalized via prompt membership
    token = Sampler().sample("r2", sampling, 0, logits, prompt=(), outputs=(3,))
    assert token.token_id == 7  # penalized via output membership


def test_presence_frequency_penalties_use_outputs_only():
    logits = torch.zeros(VOCAB)
    logits[3] = 1.0
    sampling = EngineSampling(temperature=0.0, presence_penalty=2.0)
    # in prompt only: NOT penalized
    token = Sampler().sample("r", sampling, 0, logits, prompt=(3,), outputs=())
    assert token.token_id == 3
    token = Sampler().sample("r2", sampling, 0, logits, prompt=(), outputs=(3,))
    assert token.token_id != 3


def test_logprobs_reported_from_raw_logits():
    logits = _logits()
    raw = torch.log_softmax(logits, dim=-1)
    sampling = EngineSampling(temperature=0.0, logprobs=2)
    token = Sampler().sample("r", sampling, 0, logits)
    assert token.logprob == pytest.approx(float(raw[token.token_id]))
    assert token.top_logprobs is not None and len(token.top_logprobs) == 2
    # temperature-independence: same raw logprob under different temperature
    hot = Sampler().sample("r2", EngineSampling(temperature=2.0, logprobs=0, seed=1), 0, logits)
    assert hot.logprob == pytest.approx(float(raw[hot.token_id]))


def test_cumulative_logprob_is_finite():
    logits = _logits()
    token = Sampler().sample("r", EngineSampling(logprobs=0), 0, logits)
    assert token.logprob is not None and math.isfinite(token.logprob)


class _FakeEnforcer:
    """Stands in for XGrammarEnforcer: only even token ids are legal."""

    def __init__(self) -> None:
        self.accepted: list[int] = []
        self.terminated_after = 2

    def mask_logits(self, logits):
        logits[1::2] = float("-inf")
        return logits

    def accept(self, token_id: int) -> bool:
        self.accepted.append(token_id)
        return token_id % 2 == 0

    def is_terminated(self) -> bool:
        return len(self.accepted) >= self.terminated_after


class _PermissiveEnforcer:
    """Stateful grammar stand-in whose mask permits the complete vocabulary."""

    def __init__(self) -> None:
        self.accepted: list[int] = []

    @staticmethod
    def mask_logits(logits):
        return logits

    def accept(self, token_id: int) -> bool:
        self.accepted.append(token_id)
        return True

    @staticmethod
    def is_terminated() -> bool:
        return False


def _sampler_with_fake_enforcer() -> tuple[Sampler, _FakeEnforcer]:
    sampler = Sampler(vocab_provider=lambda: [f"t{i}" for i in range(VOCAB)])
    fake = _FakeEnforcer()
    state = sampler._state_for("r", EngineSampling())
    state.enforcer = fake  # swap in the fake behind the same interface
    return sampler, fake


def test_permissive_structured_sampling_matches_unstructured_rng_trajectory():
    logits = torch.tensor(
        [0.1, 1.7, -0.4, 0.8, 2.1, -1.3, 0.5, 1.2, -0.9, 0.3]
    )
    sampling = EngineSampling(
        temperature=0.73,
        min_p=0.04,
        top_k=8,
        top_p=0.9,
        seed=353,
    )
    structured = Sampler()
    enforcer = _PermissiveEnforcer()
    structured._state_for("structured", sampling).enforcer = enforcer
    unstructured = Sampler()

    structured_tokens = [
        structured.sample("structured", sampling, position, logits).token_id
        for position in range(8)
    ]
    unstructured_tokens = [
        unstructured.sample("unstructured", sampling, position, logits).token_id
        for position in range(8)
    ]

    assert structured_tokens == unstructured_tokens
    assert enforcer.accepted == structured_tokens


def test_grammar_mask_applies_before_selection():
    sampler, _ = _sampler_with_fake_enforcer()
    logits = torch.zeros(VOCAB)
    logits[1] = 10.0  # odd id: illegal, must never be chosen
    token = sampler.sample("r", EngineSampling(temperature=0.0), 0, logits)
    assert token.token_id % 2 == 0


def test_grammar_accept_is_idempotent_per_position():
    # shared-instance TP ranks re-sample the same position: matcher advances once
    sampler, fake = _sampler_with_fake_enforcer()
    logits = torch.zeros(VOCAB)
    logits[4] = 1.0
    for _ in range(3):
        sampler.sample("r", EngineSampling(temperature=0.0), 0, logits)
    assert len(fake.accepted) == 1


def test_grammar_termination_flag_set():
    sampler, fake = _sampler_with_fake_enforcer()
    fake.terminated_after = 1
    logits = torch.zeros(VOCAB)
    logits[4] = 1.0
    token = sampler.sample("r", EngineSampling(temperature=0.0), 0, logits)
    assert token.grammar_terminated is True


def test_grammar_requires_vocab_provider():
    with pytest.raises(RuntimeError, match="vocab_provider"):
        Sampler().sample("r", EngineSampling(json_mode=True), 0, torch.zeros(VOCAB))


def test_release_drops_state():
    sampler = Sampler()
    sampler.sample("r", EngineSampling(), 0, torch.zeros(VOCAB))
    assert "r" in sampler._states
    sampler.release("r")
    assert "r" not in sampler._states
