"""Build the P-D handoff a deployment should use for its placement (m18 D3).

The stream seam had no production caller because nothing chose between the plain
handoff and the side-stream one. This is that choice, in one place, keyed off
where the KV actually lives:

- host pools get the plain handoff — `CudaStreamProvider` requires CUDA, and
  wrapping a host copy in a stream window buys nothing;
- device pools get `StreamCopyKVHandoff` over a `CudaStreamProvider` bound to
  the pool's own device, so the extraction copy runs on its own stream.

The device path also DEFERS (m18 D3): the copy's completion event is recorded
instead of blocking the host, so the producer's next step is queued alongside it.
That is only offered where the consumer honours it — `build_pd_coordinator`
returns a `PDCoordinator`, which gates every prefill-side release on the event
and therefore keeps the source pages leased for the copy's whole lifetime.
`build_kv_handoff` defaults to the blocking form for any other caller.
"""

from __future__ import annotations

from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff
from kairyu.engine.core.kv_pool import PagedKVPool


def build_kv_handoff(
    inner,
    pool: PagedKVPool,
    *,
    force_side_stream: bool | None = None,
    defer: bool = False,
):
    """Wrap ``inner`` for the device ``pool`` lives on.

    ``force_side_stream`` overrides the placement decision — True demands the
    side stream (and raises without CUDA rather than silently degrading), False
    keeps the plain handoff. Deployment leaves it None.

    ``defer`` is opt-in and defaults off because it is a CONSUMER contract, not a
    performance knob: the returned handoff comes back while the copy still holds
    the source pages, so its caller must gate before reading the destination or
    releasing the source. ``build_pd_coordinator`` turns it on because
    ``PDCoordinator`` does exactly that.
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

    return StreamCopyKVHandoff(
        inner, CudaStreamProvider(device=pool.k.device), defer=defer
    )


def build_cpu_kv_handoff(inner, *, defer: bool = False):
    """The host-side equivalent, for tests that want the recording provider."""
    return StreamCopyKVHandoff(inner, CpuNoopStream(), defer=defer)


def build_pd_coordinator(
    *,
    model_path: str,
    num_pages: int = 4096,
    page_size: int = 16,
    max_num_batched_tokens: int = 2048,
    tokenizer=None,
    max_transfer_retries: int = 1,
    force_side_stream: bool | None = None,
    defer_handoff: bool = True,
):
    """Assemble a prefill/decode pair from a checkpoint (G2 stage 5.3 entry).

    `PDCoordinator` had no production constructor at all — it existed only in
    tests, so no deployment could reach a `KVHandoff`, let alone select a stream
    provider for one. This is the missing seam: two engines over one checkpoint,
    with the handoff chosen by where their KV lives.

    Both halves are placed by the same probe the single-process path uses, so a
    GPU host gets bf16 on-device pools and therefore the side-stream handoff.

    `defer_handoff` defaults ON here, and only here: this is the one caller that
    both enables the deferred copy and settles it. `PDCoordinator` keeps the
    prefill-side allocation leased and gates every release on the copy's
    completion event, so the producer's next step overlaps the copy without the
    source pages ever being reusable under it. Pass False to fall back to the
    blocking copy (bisecting a suspected ordering bug against it, for instance).
    """
    import torch

    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.pd import LocalKVHandoff, PDCoordinator
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.scheduler import Scheduler
    from kairyu.engine.tokenizer import resolve_tokenizer
    from kairyu.models.loader import load_model

    profile = probe()
    gpu = profile.arch == "cuda"
    device = "cuda:0" if gpu else "cpu"
    dtype = torch.bfloat16 if gpu else torch.float32
    resolved = resolve_tokenizer(tokenizer if tokenizer is not None else model_path)

    def half():
        model, config, _generation = load_model(
            model_path, dtype=dtype, attention_backend=select_backend(profile)
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
    decode_cache, decode_scheduler, decode_runner, _decode_pool = half()

    handoff = build_kv_handoff(
        LocalKVHandoff(decode_cache),
        prefill_pool,
        force_side_stream=force_side_stream,
        defer=defer_handoff,
    )
    return PDCoordinator(
        prefill_scheduler=prefill_scheduler,
        prefill_runner=prefill_runner,
        decode_scheduler=decode_scheduler,
        decode_runner=decode_runner,
        handoff=handoff,
        max_transfer_retries=max_transfer_retries,
    )
