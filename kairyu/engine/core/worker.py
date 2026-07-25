"""SPMD TP execution: driver-side runner + worker main (m16 D4).

Rank 0 owns the scheduler/EngineCore and broadcasts the frozen ``StepInput``
(m16 A1 snapshot); every rank executes the SAME step on its shard and samples
identically from identical full logits (m5 D1 agreement invariant — logits
are bitwise-deterministic through gloo/CPU collectives). Workers read rank
0's committed tokens from the NEXT snapshot's ``outputs``. Shutdown is a
``None`` broadcast (A11).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from kairyu.engine.core.step_input import StateSync, StepDelta

_SHUTDOWN = None

#: Ranks load their shard between the rendezvous and the handshake, so the
#: CI-tuned 120s default would fire on a cold multi-GB read long before anything
#: is actually deadlocked. This covers ONLY startup.
_STARTUP_TIMEOUT_S = 1800.0
#: Every collective once the group is serving. `init_process_group(timeout=)` is
#: the timeout for EVERY operation on that group, not just the rendezvous, so
#: giving the startup allowance to the default group would let one wedged rank
#: hold an in-flight generation for half an hour. The step loop therefore runs on
#: a SECOND group created with this bound, while the startup handshake — the one
#: collective that legitimately has to absorb load skew — keeps the long one.
_SERVE_OP_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class ReleaseRequest:
    request_id: str


def _config_fingerprint(model_dir: str) -> str:
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16]


def make_handshake(model_dir: str, num_pages: int, page_size: int) -> dict:
    """Rank 0 broadcasts this before the step loop; workers validate (A11)."""
    return {
        "num_pages": num_pages,
        "page_size": page_size,
        "config": _config_fingerprint(model_dir),
    }


def validate_handshake(handshake: dict, model_dir: str, num_pages: int, page_size: int) -> None:
    expected = make_handshake(model_dir, num_pages, page_size)
    if handshake != expected:
        raise RuntimeError(
            f"TP worker mismatch: driver={handshake} worker={expected} — "
            "pool sizing/config must be identical on every rank"
        )


class DistTPModelRunner:
    """Driver-side ModelRunner: snapshot → broadcast → local sharded execute.

    Drops in where ``TPModelRunner`` sits: the driver's own rank-0 shard runs
    inside this call, so ``execute`` returns real sampled tokens.
    """

    def __init__(self, comm, local_runner) -> None:
        self._comm = comm
        self._local = local_runner
        # delta-broadcast state (F4): only new/finished requests + committed
        # tokens cross the wire each step, not a full pickled snapshot of every
        # active request's (growing) prompt/outputs
        self._sync = StateSync()

    def execute(self, scheduled, states) -> dict:
        chunks = tuple(scheduled)
        delta = self._sync.diff(chunks, states)
        self._comm.broadcast(delta, src=0)
        view = self._sync.apply(delta)  # reconstructs snapshot_step()'s states exactly
        return self._local.execute(chunks, view)

    def release(self, request_id: str) -> None:
        try:
            self._comm.broadcast(ReleaseRequest(request_id), src=0)
        finally:
            release = getattr(self._local, "release", None)
            if release is not None:
                release(request_id)

    def shutdown(self) -> None:
        self._comm.broadcast(_SHUTDOWN, src=0)


def worker_step_loop(comm, local_runner) -> int:
    """Non-zero-rank main loop: execute broadcast steps until shutdown.

    Returns the number of steps executed (spawn tests assert on it).
    """
    steps = 0
    sync = StateSync()
    while True:
        payload = comm.broadcast(_SHUTDOWN, src=0)
        if payload is _SHUTDOWN or payload is None:
            return steps
        if isinstance(payload, ReleaseRequest):
            release = getattr(local_runner, "release", None)
            if release is not None:
                release(payload.request_id)
            continue
        assert isinstance(payload, StepDelta)
        view = sync.apply(payload)  # same delta -> same reconstructed states
        local_runner.execute(payload.chunks, view)
        steps += 1


@dataclass(frozen=True)
class TPPlacement:
    """Where one TP rank computes: the multi-process twin of the single-process
    ``probe()`` block in ``kairyu_backend.build_engine_loop``.

    That block never runs for ``model_path`` + ``tp > 1`` — ``build_engine_loop``
    returns into ``_build_dist_tp_loop`` before it — so without this the spawned
    ranks silently kept the CPU/fp32 defaults of ``DenseDecoder`` and
    ``PagedKVPool`` on a machine full of GPUs.
    """

    device: str
    dtype: object  # torch.dtype; annotated loosely to keep this import-light
    backend: str  # torch.distributed backend matching the device


def tp_placement(tp: int, rank: int, force_cpu: bool = False) -> TPPlacement:
    """Rank-local placement: one GPU per rank, else CPU (m8 D5 probe rules).

    ``force_cpu`` is for the CPU parity tests, which compare TP output against a
    single-process fp32 host reference and so must not follow the probe onto a
    GPU. ``KAIRYU_TP_FORCE_CPU`` is the same switch for callers that cannot pass
    the argument — notably `build_engine_loop`, whose spawned ranks read it from
    the inherited environment, so rank 0 and the workers cannot end up on
    different backends and deadlock the first collective. Deployment sets neither.
    """
    import os

    import torch

    from kairyu.engine.core.hw_profile import probe

    force_cpu = force_cpu or bool(os.environ.get("KAIRYU_TP_FORCE_CPU"))
    profile = probe()
    if force_cpu or profile.arch != "cuda":
        return TPPlacement("cpu", torch.float32, "gloo")
    if profile.device_count < tp:
        # one rank per device: overcommitting would put two shards on one GPU
        # and silently halve the memory each expects
        raise RuntimeError(
            f"tensor_parallel_size={tp} needs {tp} CUDA devices; "
            f"found {profile.device_count}"
        )
    # gloo would move every RowParallelLinear all_reduce through host memory
    return TPPlacement(f"cuda:{rank}", torch.bfloat16, "nccl")


def serving_group(backend: str):
    """A second process group carrying the OPERATIONAL collective timeout.

    Every rank must call this at the same point — it is itself a collective — so
    it is created right after the rendezvous, before the slow shard load, while
    the ranks are still in lockstep.
    """
    from datetime import timedelta

    import torch.distributed as dist

    return dist.new_group(
        timeout=timedelta(seconds=_SERVE_OP_TIMEOUT_S), backend=backend
    )


def build_tp_runner(
    model_dir: str,
    tp: int,
    rank: int,
    comm,
    num_pages: int,
    page_size: int,
    vocab: list[str],
    placement: TPPlacement | None = None,
):
    """The per-rank sharded PagedModelRunner (pool sized from the tp_view config).

    ``placement`` defaults to CPU/fp32 so the CPU-only callers (tests, gloo
    parity targets) are unchanged.
    """
    import torch

    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.sampler import Sampler
    from kairyu.models.parallel import build_tp_model

    if placement is None:
        placement = TPPlacement("cpu", torch.float32, "gloo")
    model, local_config, full_config = build_tp_model(
        model_dir,
        tp,
        rank,
        comm,
        dtype=placement.dtype,
        device=placement.device,
        # keyed off the PLACEMENT, not the raw probe: a CPU-placed rank on a GPU
        # box would otherwise get the flashinfer kernel and hand it fp32 tensors
        attention_backend=select_backend(probe() if placement.device != "cpu" else None),
    )
    pool = PagedKVPool(
        num_layers=local_config.num_hidden_layers,
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=local_config.kv_cache_num_heads,
        head_dim=local_config.kv_cache_head_dim,
        dtype=placement.dtype,
        device=placement.device,
    )
    vocab_table = list(vocab)
    runner = PagedModelRunner(
        model,
        pool,
        sampler=Sampler(vocab_provider=lambda: vocab_table),
    )
    return runner, full_config


def _tp_worker_entry(
    spawn_index: int, world_size: int, init_file: str,
    model_dir: str, num_pages: int, page_size: int, vocab: list[str],
    force_cpu: bool = False,
) -> None:
    """Spawned worker (rank = spawn_index + 1; rank 0 is the driver process).

    Module-level and side-effect-free at import (m16 A6) so torch spawn can
    pickle it. Joins the group, validates the handshake, runs the step loop
    until rank 0 broadcasts shutdown, then tears the group down."""
    import torch

    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

    rank = spawn_index + 1
    torch.set_num_threads(1)
    placement = tp_placement(world_size, rank, force_cpu)
    if placement.backend == "nccl":
        # must precede init_process_group: NCCL binds the rank to the current
        # device, and object collectives stage their buffers on it
        torch.cuda.set_device(rank)
    init_distributed(
        rank,
        world_size,
        f"file://{init_file}",
        backend=placement.backend,
        timeout_s=_STARTUP_TIMEOUT_S,
    )
    startup_comm = TorchDistCommunicator(device=placement.device)
    # created before the slow shard load, while every rank is still in lockstep
    comm = TorchDistCommunicator(
        group=serving_group(placement.backend), device=placement.device
    )
    runner, _ = build_tp_runner(
        model_dir, world_size, rank, comm, num_pages, page_size, vocab, placement
    )
    # the handshake is the collective that absorbs load skew, so it — and only it
    # — runs on the long-timeout startup group
    handshake = startup_comm.broadcast(None, src=0)
    validate_handshake(handshake, model_dir, num_pages, page_size)
    try:
        worker_step_loop(comm, runner)
    finally:
        import torch.distributed as dist

        dist.destroy_process_group()  # tears down the serving subgroup too


class DistTPLauncher:
    """Owns the spawned worker processes + the rank-0 DistTPModelRunner.

    Wires real multi-process TP into a single-process serve path: rank 0 lives in
    THIS process, ranks 1..tp-1 are spawned workers. ``shutdown()`` broadcasts the
    terminating None (worker_step_loop returns), joins the workers, and destroys
    the rank-0 group — so ``kairyu serve --tp N`` starts and stops cleanly."""

    def __init__(
        self,
        model_dir: str,
        tp: int,
        num_pages: int,
        page_size: int,
        vocab: list[str],
        force_cpu: bool = False,
    ) -> None:
        import tempfile

        import torch
        import torch.multiprocessing as mp

        from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

        # a fresh, not-yet-created path is the file:// rendezvous point
        self._init_file = tempfile.mktemp(prefix="kairyu-tp-")  # noqa: S306
        placement = tp_placement(tp, 0, force_cpu)
        # force_cpu travels to the workers: rank 0 on host memory while the
        # spawned ranks probed their way onto GPUs would deadlock the first
        # all_reduce on mismatched backends
        self._ctx = mp.spawn(
            _tp_worker_entry,
            args=(tp, self._init_file, model_dir, num_pages, page_size, vocab, force_cpu),
            nprocs=tp - 1,
            join=False,
        )
        if placement.backend == "nccl":
            torch.cuda.set_device(0)
        init_distributed(
            0,
            tp,
            f"file://{self._init_file}",
            backend=placement.backend,
            timeout_s=_STARTUP_TIMEOUT_S,
        )
        startup_comm = TorchDistCommunicator(device=placement.device)
        # created before the slow shard load, while every rank is still in lockstep
        self._comm = TorchDistCommunicator(
            group=serving_group(placement.backend), device=placement.device
        )
        runner, self.full_config = build_tp_runner(
            model_dir, tp, 0, self._comm, num_pages, page_size, vocab, placement
        )
        # the one collective that legitimately absorbs load skew
        startup_comm.broadcast(make_handshake(model_dir, num_pages, page_size), src=0)
        self.runner = DistTPModelRunner(self._comm, runner)

    def dead_ranks(self) -> tuple[int, ...]:
        """Spawned ranks that are no longer running (rank 0 is this process).

        A dead rank leaves the group unable to complete a single collective, but
        rank 0 stays up and keeps answering health checks — on hardware this
        presented as a served model that accepted requests and never returned a
        token. Cheap enough for `/readyz`: `is_alive()` is a waitpid, no IPC.
        """
        return tuple(
            index + 1
            for index, process in enumerate(self._ctx.processes)
            if not process.is_alive()
        )

    def shutdown(self) -> None:
        import contextlib
        import os

        import torch.distributed as dist

        self.runner.shutdown()  # broadcasts None -> workers leave worker_step_loop
        # BEFORE the join, not after: NCCL's destroy_process_group waits for every
        # rank to reach it, so joining first deadlocks rank 0 against workers that
        # are already sitting in their own destroy. gloo never blocks here, which
        # is why the CPU parity gates could not see this.
        if dist.is_initialized():
            # no argument: torch tears down the serving subgroup along with the
            # default one, and destroying them individually is registration-order
            # dependent across backends
            dist.destroy_process_group()
        self._ctx.join()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._init_file)
