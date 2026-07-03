"""Sampler behaviors pinned by design m8 D2 (order, determinism, grammar)."""

import math

import pytest
import torch

from kairyu.engine.core.sampler import Sampler, mix_seed, stable_request_seed
from kairyu.engine.core.sampling_types import EngineSampling

VOCAB = 32


def _logits(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, generator=generator)


def test_temperature_zero_is_exact_argmax():
    sampler = Sampler()
    logits = _logits()
    token = sampler.sample("r", EngineSampling(temperature=0.0), 0, logits)
    assert token.token_id == int(torch.argmax(logits).item())


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


def _sampler_with_fake_enforcer() -> tuple[Sampler, _FakeEnforcer]:
    sampler = Sampler(vocab_provider=lambda: [f"t{i}" for i in range(VOCAB)])
    fake = _FakeEnforcer()
    state = sampler._state_for("r", EngineSampling())
    state.enforcer = fake  # swap in the fake behind the same interface
    return sampler, fake


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
