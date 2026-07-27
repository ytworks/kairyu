"""Stateless device sampling for the overlap decode path.

The random draw is a pure function of ``(base seed, output position, vocab
index)``.  It therefore survives batch reordering, CUDA-graph replay, and
overlap retries without carrying a mutable generator offset on the host.
"""

from __future__ import annotations

import torch


def stateless_gumbel_argmax(log_weights: torch.Tensor, seed: int) -> torch.Tensor:
    """Return one categorical draw from unnormalized log weights on CUDA.

    A small integer mixer supplies one uniform per vocabulary index.  The
    flipped Gumbel transform puts the winning tail at ``u -> 0``, where fp32
    retains much finer resolution than at ``u -> 1``.  Everything is ordinary
    device tensor work, so this path does not need a mutable CUDA Generator,
    a compiled extension, or a host-visible result.
    """
    if log_weights.device.type != "cuda":
        raise ValueError("stateless_gumbel_argmax requires CUDA log weights")
    if log_weights.ndim != 1:
        raise ValueError(f"log_weights must be 1D, got {tuple(log_weights.shape)}")

    offsets = torch.arange(
        log_weights.numel(), dtype=torch.int64, device=log_weights.device
    )
    # Thomas Wang-style 32-bit integer mix.  int64 intermediates make the wrap
    # explicit and identical across CUDA architectures.
    values = (offsets + (seed & 0xFFFFFFFF)) & 0xFFFFFFFF
    values = ((values ^ (values >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    values = ((values ^ (values >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    values = values ^ (values >> 16)
    uniform = ((values & 0xFFFFFF).to(torch.float32) + 0.5) * (1.0 / (1 << 24))
    noise = -torch.log(-torch.log1p(-uniform))
    return torch.argmax(log_weights.to(torch.float32) + noise).to(torch.int64)
