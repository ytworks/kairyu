"""Stateless tensor sampling shared by CPU and CUDA paths.

The random draw is a pure function of ``(base seed, output position, vocab
index)``.  It therefore survives batch reordering, CUDA-graph replay, and
overlap retries without carrying a mutable generator offset on the host.  The
same tensor algorithm is used for CPU/structured replay and CUDA serving.
"""

from __future__ import annotations

from functools import lru_cache

import torch

_MAX_CACHED_CPU_OFFSETS = 1 << 20
_MASK64 = (1 << 64) - 1
_SIGN_BIT64 = 1 << 63
_SPLITMIX_GAMMA = -7046029254386353131
_SPLITMIX_MIX1 = -4658895280553007687
_SPLITMIX_MIX2 = -7723592293110705685
_UNIFORM_BITS = 52
_UNIFORM_MASK = (1 << _UNIFORM_BITS) - 1
_UNIFORM_SCALE = 1.0 / (1 << _UNIFORM_BITS)


@lru_cache(maxsize=4)
def _cached_cpu_offsets(size: int) -> torch.Tensor:
    """Reuse a bounded immutable vocabulary index vector on CPU."""
    return torch.arange(size, dtype=torch.int64, device="cpu")


def _signed_int64(value: int) -> int:
    """Return an unsigned Python integer's signed two's-complement value."""
    value &= _MASK64
    return value if value < _SIGN_BIT64 else value - (_MASK64 + 1)


def _logical_right_shift(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Logical uint64 shift implemented with supported signed-int64 ops."""
    return (values >> bits) & ((1 << (64 - bits)) - 1)


def _splitmix64_finalizer(values: torch.Tensor) -> torch.Tensor:
    """Avalanche signed-int64 words exactly as unsigned SplitMix64 words."""
    values = (values ^ _logical_right_shift(values, 30)) * _SPLITMIX_MIX1
    values = (values ^ _logical_right_shift(values, 27)) * _SPLITMIX_MIX2
    return values ^ _logical_right_shift(values, 31)


def _stateless_random_words(offsets: torch.Tensor, seed: int) -> torch.Tensor:
    """Return one full-width keyed counter word per vocabulary offset.

    Multiplication by the odd SplitMix increment permutes all uint64 counter
    values.  XOR keys that counter with every seed bit before the finalizer, so
    adjacent seeds cannot become shifted views of one another.
    """
    counters = (offsets + 1) * _SPLITMIX_GAMMA
    return _splitmix64_finalizer(counters ^ _signed_int64(seed))


def _uniform_from_words(words: torch.Tensor) -> torch.Tensor:
    """Map 52 random bits to a strictly open float64 uniform interval."""
    mantissas = words & _UNIFORM_MASK
    return (mantissas.to(torch.float64) + 0.5) * _UNIFORM_SCALE


def stateless_gumbel_argmax(log_weights: torch.Tensor, seed: int) -> torch.Tensor:
    """Return one categorical draw from unnormalized log weights.

    A full-width keyed counter supplies one 52-bit open-interval uniform per
    vocabulary index.  The flipped Gumbel transform puts the winning tail at
    ``u -> 0``.  Everything is ordinary tensor work on
    ``log_weights.device``, so neither CPU nor CUDA needs a mutable Generator
    or a compiled extension; CUDA also keeps the result off the host-visible
    path.
    """
    if log_weights.ndim != 1:
        raise ValueError(f"log_weights must be 1D, got {tuple(log_weights.shape)}")

    size = log_weights.numel()
    if log_weights.device.type == "cpu" and size <= _MAX_CACHED_CPU_OFFSETS:
        offsets = _cached_cpu_offsets(size)
    else:
        offsets = torch.arange(size, dtype=torch.int64, device=log_weights.device)
    values = _stateless_random_words(offsets, seed)
    uniform = _uniform_from_words(values)
    noise = -torch.log(-torch.log1p(-uniform))
    return torch.argmax(log_weights.to(torch.float64) + noise).to(torch.int64)
