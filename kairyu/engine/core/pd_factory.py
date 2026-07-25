"""Build the P-D handoff a deployment should use for its placement (m18 D3).

Two decisions live here. *What* is copied: the two engines own separate pools,
so the inner handoff is ``LocalCopyKVHandoff`` — the accounting-only
``LocalKVHandoff`` would publish the destination's untouched pages as computed
and decode from KV that was never written. *Where* the copy runs: the stream
seam had no production caller because nothing chose between the plain handoff
and the side-stream one. This is that choice, in one place, keyed off where the
KV actually lives:

- host pools get the plain handoff — `CudaStreamProvider` requires CUDA, and
  wrapping a host copy in a stream window buys nothing;
- device pools get `StreamCopyKVHandoff` over a `CudaStreamProvider` bound to
  the pool's own device, so the extraction copy runs on its own stream.

What this does NOT do, and the docstrings say so at the seam: overlap the copy
with the next forward. `StreamCopyKVHandoff` blocks the host before returning, so
the consumer would need a completion event instead of a host-wide wait.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff
from kairyu.engine.core.kv_pool import PagedKVPool

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from kairyu.engine.core.hw_profile import HardwareProfile


def build_kv_handoff(inner, pool: PagedKVPool, *, force_side_stream: bool | None = None):
    """Wrap ``inner`` for the device ``pool`` lives on.

    ``force_side_stream`` overrides the placement decision — True demands the
    side stream (and raises without CUDA rather than silently degrading), False
    keeps the plain handoff. Deployment leaves it None.
    """
    on_device = pool.k.device.type == "cuda"
    wanted = on_device if force_side_stream is None else force_side_stream
    if not wanted:
        return inner
    if not on_device and force_side_stream:
        raise ValueError(
            "force_side_stream=True needs a CUDA KV pool; this one is on "
            f"{pool.k.device}"
        )
    from kairyu.engine.core.handoff_stream import CudaStreamProvider

    return StreamCopyKVHandoff(inner, CudaStreamProvider(device=pool.k.device))


def build_cpu_kv_handoff(inner):
    """The host-side equivalent, for tests that want the recording provider."""
    return StreamCopyKVHandoff(inner, CpuNoopStream())


def build_pd_coordinator(
    *,
    model_path: str,
    num_pages: int = 4096,
    page_size: int = 16,
    max_num_batched_tokens: int = 2048,
    tokenizer=None,
    max_transfer_retries: int = 1,
    force_side_stream: bool | None = None,
    profile: HardwareProfile | None = None,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    attention_backend=None,
):
    """Assemble a prefill/decode pair from a checkpoint (G2 stage 5.3 entry).

    `PDCoordinator` had no production constructor at all — it existed only in
    tests, so no deployment could reach a `KVHandoff`, let alone select a stream
    provider for one. This is the missing seam: two engines over one checkpoint,
    with the handoff chosen by where their KV lives.

    Placement defaults to the same probe the single-process path uses, so a GPU
    host gets bf16 on-device pools and therefore the side-stream handoff.
    `profile` / `device` / `dtype` / `attention_backend` override each half of
    that decision: unit tests inject a CPU placement instead of depending on
    whatever hardware and optional kernels the machine happens to have.

    The two halves own separate pools, so the handoff must move BYTES —
    `LocalCopyKVHandoff`, not the accounting-only `LocalKVHandoff`.
    """
    import torch

    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.pd import LocalCopyKVHandoff, PDCoordinator
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.scheduler import Scheduler
    from kairyu.engine.tokenizer import resolve_tokenizer
    from kairyu.models.loader import load_model

    profile = probe() if profile is None else profile
    gpu = profile.arch == "cuda"
    if device is None:
        device = "cuda:0" if gpu else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if gpu else torch.float32
    if attention_backend is None:
        attention_backend = select_backend(profile)
    resolved = resolve_tokenizer(tokenizer if tokenizer is not None else model_path)

    def half():
        model, config, _generation = load_model(
            model_path, dtype=dtype, attention_backend=attention_backend
        )
        cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
        scheduler = Scheduler(
            cache, max_num_batched_tokens=max_num_batched_tokens, page_size=page_size
        )
        pool = PagedKVPool.for_cache(cache, config, dtype=dtype, device=device)
        runner = PagedModelRunner(
            model.to(device), pool, sampler=Sampler(vocab_provider=resolved.vocab),
            cache=cache,
        )
        return cache, scheduler, runner, pool

    _prefill_cache, prefill_scheduler, prefill_runner, prefill_pool = half()
    decode_cache, decode_scheduler, decode_runner, decode_pool = half()

    handoff = build_kv_handoff(
        LocalCopyKVHandoff(decode_cache, prefill_pool, decode_pool),
        prefill_pool,
        force_side_stream=force_side_stream,
    )
    return PDCoordinator(
        prefill_scheduler=prefill_scheduler,
        prefill_runner=prefill_runner,
        decode_scheduler=decode_scheduler,
        decode_runner=decode_runner,
        handoff=handoff,
        max_transfer_retries=max_transfer_retries,
    )
