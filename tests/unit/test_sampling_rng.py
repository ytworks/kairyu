"""Full-width stateless sampling RNG quality gates (issue #354)."""

from __future__ import annotations

import msgpack
import torch

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.core.engine_service import (
    sampling_params_from_wire,
    sampling_params_to_wire,
)
from kairyu.engine.core.sampling_types import mix_seed, stable_request_seed
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.kernels import sampling_gpu

_MASK64 = (1 << 64) - 1
_SIGN_BIT64 = 1 << 63
_GAMMA64 = 0x9E3779B97F4A7C15
_MIX1_64 = 0xBF58476D1CE4E5B9
_MIX2_64 = 0x94D049BB133111EB


def _python_random_word(index: int, seed: int) -> int:
    """Independent unsigned-64 oracle for the production tensor mixer."""
    value = (((index + 1) * _GAMMA64) & _MASK64) ^ (seed & _MASK64)
    value = ((value ^ (value >> 30)) * _MIX1_64) & _MASK64
    value = ((value ^ (value >> 27)) * _MIX2_64) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _as_signed_int64(value: int) -> int:
    value &= _MASK64
    return value if value < _SIGN_BIT64 else value - (1 << 64)


def test_mix_seed_retains_full_uint64_output() -> None:
    assert mix_seed(0, 0) == 0xE220A8397B1DCDAF
    assert mix_seed(0, 1) == 0x6E789E6AA1B965F4
    assert mix_seed(0, 2) == 0x06C45D188009454F
    assert mix_seed(0, 0) >= _SIGN_BIT64
    assert mix_seed(-1, 7) == mix_seed(_MASK64, 7)
    assert mix_seed(7 + (1 << 64), 11) == mix_seed(7, 11)
    assert stable_request_seed("issue-354") < _SIGN_BIT64


def test_n_choice_full_uint64_seed_round_trips_through_process_wire() -> None:
    request = GenerationRequest(
        "n-choice",
        "prompt",
        SamplingParams(n=2, seed=1, max_tokens=1),
    )
    backend = object.__new__(KairyuBackend)
    children = backend._sub_requests(request)
    derived_seed = mix_seed(1, 1)

    assert [child.sampling_params.seed for child in children] == [1, derived_seed]
    assert derived_seed >= _SIGN_BIT64

    payload = sampling_params_to_wire(children[1].sampling_params)
    packed = msgpack.packb(payload)
    unpacked = msgpack.unpackb(packed)
    rebuilt = sampling_params_from_wire(unpacked)
    assert rebuilt.seed == derived_seed


def test_tensor_counter_matches_independent_unsigned_splitmix_oracle() -> None:
    indices = torch.tensor([0, 1, 2, 3, 7, 31, 257, 151_935], dtype=torch.int64)

    for seed in (
        0,
        1,
        1 << 32,
        1 << 48,
        1 << 63,
        _MASK64,
        -1,
        (1 << 80) + 7,
    ):
        actual = [
            value & _MASK64
            for value in sampling_gpu._stateless_random_words(indices, seed).tolist()
        ]
        expected = [_python_random_word(index, seed) for index in indices.tolist()]
        assert actual == expected


def test_full_seed_bits_change_words_without_adjacent_shift_aliases() -> None:
    offsets = torch.arange(100_000, dtype=torch.int64)
    seed = 0x1234_5678_9ABC_DEF0
    baseline = sampling_gpu._stateless_random_words(offsets, seed)

    for bit in (32, 48, 62, 63):
        changed = sampling_gpu._stateless_random_words(offsets, seed ^ (1 << bit))
        assert not torch.any(changed == baseline)

    adjacent = sampling_gpu._stateless_random_words(offsets, seed + 1)
    assert not torch.equal(baseline[1:], adjacent[:-1])

    # These production position seeds collide under the removed low-32 mask.
    first_seed = mix_seed(353, 35_033)
    second_seed = mix_seed(353, 53_335)
    assert first_seed != second_seed
    assert (first_seed & 0xFFFFFFFF) == (second_seed & 0xFFFFFFFF)
    assert not torch.equal(
        sampling_gpu._stateless_random_words(offsets[:257], first_seed),
        sampling_gpu._stateless_random_words(offsets[:257], second_seed),
    )


def test_uniform_mapping_is_open_wide_finite_and_collision_free_at_qwen_width() -> None:
    endpoint_words = torch.tensor([0, sampling_gpu._UNIFORM_MASK, -1], dtype=torch.int64)
    endpoints = sampling_gpu._uniform_from_words(endpoint_words)
    noise = -torch.log(-torch.log1p(-endpoints))

    assert endpoints.dtype == torch.float64
    assert endpoints[0] == 2**-53
    assert endpoints[1] == 1.0 - 2**-53
    assert endpoints[2] == 1.0 - 2**-53
    assert torch.all((endpoints > 0.0) & (endpoints < 1.0))
    assert torch.all(torch.isfinite(noise))
    assert noise[0] > 30.0

    vocab_size = 151_936
    offsets = torch.arange(vocab_size, dtype=torch.int64)
    words = sampling_gpu._stateless_random_words(offsets, mix_seed(354, 0))
    mantissas = words & sampling_gpu._UNIFORM_MASK
    assert torch.unique(words).numel() == vocab_size
    assert torch.unique(mantissas).numel() == vocab_size


def test_adjacent_seed_correlations_and_three_category_frequencies() -> None:
    sample_count = 100_000
    offsets = torch.arange(sample_count, dtype=torch.int64)
    first = sampling_gpu._uniform_from_words(sampling_gpu._stateless_random_words(offsets, 354))
    second = sampling_gpu._uniform_from_words(sampling_gpu._stateless_random_words(offsets, 355))

    pairs = ((first[:-1], second[1:]), (first, second), (first[1:], second[:-1]))
    for left, right in pairs:
        correlation = torch.corrcoef(torch.stack((left, right)))[0, 1]
        assert abs(float(correlation)) < 0.01
        assert not torch.any(left == right)

    position_seeds = torch.tensor(
        [_as_signed_int64(mix_seed(354, position)) for position in range(sample_count)],
        dtype=torch.int64,
    )
    vocab_offsets = torch.arange(3, dtype=torch.int64)
    counters = (vocab_offsets + 1) * sampling_gpu._SPLITMIX_GAMMA
    words = sampling_gpu._splitmix64_finalizer(counters.unsqueeze(0) ^ position_seeds.unsqueeze(1))
    uniforms = sampling_gpu._uniform_from_words(words)
    noise = -torch.log(-torch.log1p(-uniforms))
    probabilities = torch.tensor((0.2, 0.3, 0.5), dtype=torch.float64)
    draws = torch.argmax(torch.log(probabilities) + noise, dim=1)
    frequencies = torch.bincount(draws, minlength=3).to(torch.float64) / sample_count

    assert torch.all(torch.abs(frequencies - probabilities) <= 0.005)
