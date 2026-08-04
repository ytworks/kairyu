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
- device pools get `StreamCopyKVHandoff` over a `CudaStreamProvider`; a
  cross-device pair owns one source-copy stream and one destination-completion
  stream and requires peer access when deferred.

The device path also DEFERS (m18 D3): the copy's completion event is recorded
instead of blocking the host, so the producer's next step is queued alongside it.
That is only offered where the consumer honours it — `build_pd_coordinator`
returns a `PDCoordinator`, which holds the whole settlement (release, commit and
decode-side adoption) until the event physically completes, keeping source pages
leased for the copy's whole lifetime. `build_kv_handoff` defaults to the blocking
form for any other caller.

Prefill and decode may name distinct role devices. Automatic attention selection
probes each role and constructs separate stateful backends; an explicitly shared
backend is accepted cross-device only when it declares itself device-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kairyu.engine.core.handoff_stream import CpuNoopStream, StreamCopyKVHandoff
from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.models.generation import GenerationConfigMode

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from kairyu.engine.core.hw_profile import HardwareProfile


def build_kv_handoff(
    inner,
    pool: PagedKVPool,
    *,
    force_side_stream: bool | None = None,
    defer: bool = False,
    transfer_device: object | None = None,
    gate_devices: tuple[object, ...] = (),
):
    """Wrap ``inner`` for the device ``pool`` lives on.

    ``force_side_stream`` overrides the placement decision — True demands the
    side stream (and raises without CUDA rather than silently degrading), False
    keeps the plain handoff. Deployment leaves it None.

    ``defer`` is opt-in and defaults off because it is a CONSUMER contract: the
    returned handoff comes back while the copy still holds source pages, so its
    caller must poll and finalize physical completion before reading or
    releasing. Cross-device deferred transfer additionally requires CUDA peer
    access. ``build_pd_coordinator`` implements that ownership contract.
    """
    on_device = pool.k.device.type == "cuda"
    wanted = on_device if force_side_stream is None else force_side_stream
    if not wanted:
        return inner
    if not on_device and force_side_stream:
        raise ValueError(
            f"force_side_stream=True needs a CUDA KV pool; this one is on {pool.k.device}"
        )
    from kairyu.engine.core.handoff_stream import CudaStreamProvider

    return StreamCopyKVHandoff(
        inner,
        CudaStreamProvider(
            device=pool.k.device,
            transfer_device=transfer_device,
            gate_devices=gate_devices,
            require_peer_access=defer,
        ),
        defer=defer,
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
    max_num_seqs: int = 256,
    priority_age_s: float | None = 60.0,
    tokenizer=None,
    max_transfer_retries: int = 1,
    force_side_stream: bool | None = None,
    defer_handoff: bool = True,
    profile: HardwareProfile | None = None,
    device: str | None = None,
    prefill_device: str | None = None,
    decode_device: str | None = None,
    dtype: torch.dtype | None = None,
    attention_backend=None,
    prefill_attention_backend=None,
    decode_attention_backend=None,
    generation_config: GenerationConfigMode = "auto",
):
    """Assemble a prefill/decode pair from a checkpoint (G2 stage 5.3 entry).

    `PDCoordinator` had no production constructor at all — it existed only in
    tests, so no deployment could reach a `KVHandoff`, let alone select a stream
    provider for one. This is the missing seam: two engines over one checkpoint,
    with the handoff chosen by where their KV lives.

    Placement defaults to the same probe the single-process path uses. Explicit
    `prefill_device`/`decode_device` values may select distinct CUDA devices;
    each role is probed and receives its own stateful attention backend.
    `profile` / `device` / `dtype` / role-specific backends remain injectable for
    controlled placements and tests.

    The two halves own separate pools, so the handoff must move BYTES —
    `LocalCopyKVHandoff`, not the accounting-only `LocalKVHandoff`.

    `defer_handoff` defaults ON here, and only here: this is the one caller that
    both enables the deferred copy and settles it. `PDCoordinator` keeps the
    prefill-side allocation leased and settles a step's transfers only after the
    NEXT eligible forward has been queued, then publishes/adopts only after a
    physical completion poll. Pass False for the safe blocking/non-overlap
    control.
    """
    import torch

    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.attention_selector import (
        AttentionBackendDecision,
        compose_backend_decisions,
    )
    from kairyu.engine.core.hw_profile import HardwareProfile, probe
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.pd import LocalCopyKVHandoff, PDCoordinator
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.scheduler import Scheduler
    from kairyu.engine.tokenizer import grammar_vocabulary, resolve_tokenizer
    from kairyu.models.loader import load_model

    profile_supplied = profile is not None
    profile = probe() if profile is None else profile
    gpu = profile.arch == "cuda"
    if device is None:
        device = "cuda:0" if gpu else "cpu"
    prefill_device = device if prefill_device is None else prefill_device
    decode_device = device if decode_device is None else decode_device

    def canonical_device(value: object) -> torch.device:
        selected = torch.device(value)
        if selected.type not in {"cpu", "cuda"}:
            raise ValueError(
                f"P-D role devices must be CPU or CUDA, got unsupported device {selected}"
            )
        if selected.type == "cuda" and selected.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return selected

    prefill_device_key = canonical_device(prefill_device)
    decode_device_key = canonical_device(decode_device)
    if prefill_device_key.type != decode_device_key.type:
        raise ValueError(
            "prefill_device and decode_device must both be CUDA or both be CPU; "
            f"got {prefill_device_key} and {decode_device_key}"
        )
    prefill_device = prefill_device_key
    decode_device = decode_device_key
    if dtype is None:
        any_gpu_role = prefill_device_key.type == "cuda" or decode_device_key.type == "cuda"
        dtype = torch.bfloat16 if any_gpu_role else torch.float32

    def role_profile(role_device: torch.device) -> HardwareProfile:
        if role_device.type == "cpu":
            return HardwareProfile(arch="cpu")
        if profile_supplied and prefill_device_key == decode_device_key:
            return profile
        return probe(role_device)

    if attention_backend is not None and (
        prefill_attention_backend is not None or decode_attention_backend is not None
    ):
        raise ValueError(
            "attention_backend cannot be combined with role-specific "
            "prefill_attention_backend/decode_attention_backend"
        )
    if attention_backend is not None:
        if prefill_device_key != decode_device_key:
            raise ValueError(
                "cross-device P-D cannot share one stateful attention_backend; "
                "pass role-specific backends or leave them unset for automatic selection"
            )
        prefill_attention_backend = decode_attention_backend = attention_backend
    elif (
        prefill_attention_backend is None
        and decode_attention_backend is None
        and prefill_device_key == decode_device_key
    ):
        shared = select_backend(role_profile(prefill_device_key), device=prefill_device_key)
        prefill_attention_backend = decode_attention_backend = shared
    else:
        if prefill_attention_backend is None:
            prefill_attention_backend = select_backend(
                role_profile(prefill_device_key), device=prefill_device_key
            )
        if decode_attention_backend is None:
            decode_attention_backend = select_backend(
                role_profile(decode_device_key), device=decode_device_key
            )

    if (
        prefill_device_key != decode_device_key
        and prefill_attention_backend is decode_attention_backend
        and not getattr(prefill_attention_backend, "device_agnostic", False)
    ):
        raise ValueError("cross-device P-D cannot share one stateful attention backend instance")

    def validate_backend_device(name: str, backend, role_device: torch.device) -> None:
        declared = getattr(backend, "device", None)
        if declared is None:
            return
        backend_device = canonical_device(declared)
        if backend_device != role_device:
            raise ValueError(
                f"{name} is bound to {backend_device}, not its role device {role_device}"
            )

    validate_backend_device(
        "prefill_attention_backend",
        prefill_attention_backend,
        prefill_device_key,
    )
    validate_backend_device(
        "decode_attention_backend",
        decode_attention_backend,
        decode_device_key,
    )
    resolved = resolve_tokenizer(tokenizer if tokenizer is not None else model_path)

    def half(role_device: torch.device, role_attention_backend):
        model, config, _generation = load_model(
            model_path,
            dtype=dtype,
            attention_backend=role_attention_backend,
            target_device=role_device,
            generation_config=generation_config,
        )
        cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
        scheduler = Scheduler(
            cache,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            page_size=page_size,
            priority_age_s=priority_age_s,
        )
        pool = PagedKVPool.for_cache(cache, config, dtype=dtype, device=role_device)
        grammar_vocab = grammar_vocabulary(resolved, model_vocab_size=config.vocab_size)
        runner = PagedModelRunner(
            model.to(role_device),
            pool,
            sampler=Sampler(vocab_provider=lambda: grammar_vocab),
            cache=cache,
        )
        return cache, scheduler, runner, pool

    _prefill_cache, prefill_scheduler, prefill_runner, prefill_pool = half(
        prefill_device, prefill_attention_backend
    )
    decode_cache, decode_scheduler, decode_runner, decode_pool = half(
        decode_device, decode_attention_backend
    )

    handoff = build_kv_handoff(
        LocalCopyKVHandoff(
            decode_cache,
            prefill_pool,
            decode_pool,
        ),
        prefill_pool,
        force_side_stream=force_side_stream,
        defer=defer_handoff,
        transfer_device=decode_pool.k.device,
        gate_devices=(prefill_pool.k.device, decode_pool.k.device),
    )
    coordinator = PDCoordinator(
        prefill_scheduler=prefill_scheduler,
        prefill_runner=prefill_runner,
        decode_scheduler=decode_scheduler,
        decode_runner=decode_runner,
        handoff=handoff,
        max_transfer_retries=max_transfer_retries,
    )
    prefill_decision = getattr(prefill_attention_backend, "selection_decision", None)
    decode_decision = getattr(decode_attention_backend, "selection_decision", None)
    coordinator.attention_backend_decision = (
        compose_backend_decisions(prefill_decision, decode_decision)
        if isinstance(prefill_decision, AttentionBackendDecision)
        and isinstance(decode_decision, AttentionBackendDecision)
        else None
    )
    return coordinator
