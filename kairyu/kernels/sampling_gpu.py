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


@lru_cache(maxsize=4)
def _cached_cpu_offsets(size: int) -> torch.Tensor:
    """Reuse a bounded immutable vocabulary index vector on CPU."""
    return torch.arange(size, dtype=torch.int64, device="cpu")


def stateless_gumbel_argmax(log_weights: torch.Tensor, seed: int) -> torch.Tensor:
    """Return one categorical draw from unnormalized log weights.

    A small integer mixer supplies one uniform per vocabulary index.  The
    flipped Gumbel transform puts the winning tail at ``u -> 0``, where fp32
    retains much finer resolution than at ``u -> 1``.  Everything is ordinary
    tensor work on ``log_weights.device``, so neither CPU nor CUDA needs a
    mutable Generator or a compiled extension; CUDA also keeps the result off
    the host-visible path.
    """
    if log_weights.ndim != 1:
        raise ValueError(f"log_weights must be 1D, got {tuple(log_weights.shape)}")

    size = log_weights.numel()
    if log_weights.device.type == "cpu" and size <= _MAX_CACHED_CPU_OFFSETS:
        offsets = _cached_cpu_offsets(size)
    else:
        offsets = torch.arange(size, dtype=torch.int64, device=log_weights.device)
    # Thomas Wang-style 32-bit integer mix.  int64 intermediates make the wrap
    # explicit and identical across CUDA architectures.
    values = (offsets + (seed & 0xFFFFFFFF)) & 0xFFFFFFFF
    values = ((values ^ (values >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    values = ((values ^ (values >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    values = values ^ (values >> 16)
    uniform = ((values & 0xFFFFFF).to(torch.float32) + 0.5) * (1.0 / (1 << 24))
    noise = -torch.log(-torch.log1p(-uniform))
    return torch.argmax(log_weights.to(torch.float32) + noise).to(torch.int64)
