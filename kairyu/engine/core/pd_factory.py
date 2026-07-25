"""Build the P-D handoff a deployment should use for its placement (m18 D3).

The stream seam had no production caller because nothing chose between the plain
handoff and the side-stream one. This is that choice, in one place, keyed off
where the KV actually lives:

- host pools get the plain handoff — `CudaStreamProvider` requires CUDA, and
  wrapping a host copy in a stream window buys nothing;
- device pools get `StreamCopyKVHandoff` over a `CudaStreamProvider` bound to
  the pool's own device, so the extraction copy runs on its own stream.

What this does NOT do, and the docstrings say so at the seam: overlap the copy
with the next forward. `StreamCopyKVHandoff` blocks the host before returning
(production leaves `defer=False`), so the consumer would need a completion event
instead of a host-wide wait.
"""

from __future__ import annotations

from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff
from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.kv_serde import pool_fingerprint
from kairyu.engine.core.pd import KVHandoffError
from kairyu.engine.core.radix_kv import KVAllocation, KVCacheFull, RadixKVCache


class PagedCopyKVHandoff:
    """Same-process handoff that moves the KV BYTES between two paged pools.

    ``LocalKVHandoff`` is the accounting half only — it allocates in the
    destination cache and marks it computed, ignoring the source pages. That is
    correct when both halves share one pool (the CPU protocol tests), and wrong
    for a real prefill/decode pair: the destination pool is a second tensor,
    zero-initialised, so decode would attend over empty KV. This copies each
    non-cached page, all layers at once, source pool -> destination pool.

    Ordering is the caller's, and it is copy-before-commit (m6 D4):
    ``PDCoordinator`` transfers between ``execute()`` and ``update()``, while
    every source page is still locked — after ``commit_and_release`` the tail
    page may already be reallocated.
    """

    def __init__(
        self, dest_kv: RadixKVCache, src_pool: PagedKVPool, dest_pool: PagedKVPool
    ) -> None:
        if pool_fingerprint(src_pool) != pool_fingerprint(dest_pool):
            raise ValueError(
                f"pool mismatch: source {pool_fingerprint(src_pool)} cannot be "
                f"copied into destination {pool_fingerprint(dest_pool)}"
            )
        self._dest = dest_kv
        self._src_pool = src_pool
        self._dest_pool = dest_pool

    def transfer(
        self, tokens: tuple[int, ...], first_token: int, pages: tuple[int, ...] = ()
    ) -> KVAllocation:
        if not pages:
            raise KVHandoffError("paged handoff needs the source page ids")
        try:
            allocation = self._dest.allocate(tuple(tokens))
        except KVCacheFull as error:
            raise KVHandoffError(f"destination cache full: {error}") from error
        try:
            # receiver-side dedup (m6 D4): a destination radix hit is a PREFIX,
            # so the leading source pages it covers are already there
            targets = tuple(
                page
                for page in tuple(allocation.new_full_pages) + (allocation.tail_page,)
                if page is not None
            )
            sources = tuple(pages)[len(allocation.cached_pages) :]
            if len(sources) != len(targets):
                raise KVHandoffError(
                    f"{len(sources)} source pages for {len(targets)} destination "
                    "slots (source and destination page sizes disagree)"
                )
            for source, target in zip(sources, targets, strict=True):
                self._dest_pool.k[:, target].copy_(self._src_pool.k[:, source])
                if self._dest_pool.v_head_dim:  # MLA keeps the latent in k
                    self._dest_pool.v[:, target].copy_(self._src_pool.v[:, source])
            self._dest.mark_computed(allocation)
            return allocation
        except Exception:
            self._dest.release_preempted(allocation)
            raise


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
):
    """Assemble a prefill/decode pair from a checkpoint (G2 stage 5.3 entry).

    `PDCoordinator` had no production constructor at all — it existed only in
    tests, so no deployment could reach a `KVHandoff`, let alone select a stream
    provider for one. This is the missing seam: two engines over one checkpoint,
    with the handoff chosen by where their KV lives.

    Both halves are placed by the same probe the single-process path uses, so a
    GPU host gets bf16 on-device pools and therefore the side-stream handoff.

    The two halves own two pools, so the handoff has to move BYTES: an
    accounting-only ``LocalKVHandoff`` here would leave decode attending over a
    zero-initialised pool.
    """
    import torch

    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.pd import PDCoordinator
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
    decode_cache, decode_scheduler, decode_runner, decode_pool = half()

    handoff = build_kv_handoff(
        PagedCopyKVHandoff(decode_cache, prefill_pool, decode_pool),
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
