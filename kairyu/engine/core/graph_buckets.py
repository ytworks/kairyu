"""Decode-batch bucket policy for graph capture (m17 D2) — pure functions."""

from __future__ import annotations

_SMALL = (1, 2, 4, 8, 16, 24, 32)
_STEP = 8


def decode_buckets(max_batch: int) -> tuple[int, ...]:
    """vLLM-style capture sizes: small powers, then +8 steps, capped."""
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1")
    buckets = [size for size in _SMALL if size <= max_batch]
    next_size = _SMALL[-1] + _STEP
    while next_size <= max_batch:
        buckets.append(next_size)
        next_size += _STEP
    if buckets[-1] != max_batch:
        buckets.append(max_batch)
    return tuple(buckets)


def bucket_for(batch_size: int, buckets: tuple[int, ...]) -> int | None:
    """Smallest bucket >= batch_size; None -> caller falls back to eager."""
    for bucket in buckets:
        if bucket >= batch_size:
            return bucket
    return None
