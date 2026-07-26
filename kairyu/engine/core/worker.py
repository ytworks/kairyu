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
#: Model collectives once the group is serving. `init_process_group(timeout=)`
#: bounds every operation on that group, so a wedged rank must not hold an
#: in-flight generation for the startup allowance.
_SERVE_OP_TIMEOUT_S = 120.0
#: Non-zero ranks intentionally sit inside the next control broadcast while the
#: server is idle. A short collective timeout therefore kills a healthy TP group
#: after exactly that much idle time. Keep the control receive effectively
#: process-lifetime while model work retains the fail-fast bound above.
_CONTROL_IDLE_TIMEOUT_S = 365 * 24 * 60 * 60.0


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
        self._fatal_error: Exception | None = None
        # delta-broadcast state (F4): only new/finished requests + committed
        # tokens cross the wire each step, not a full pickled snapshot of every
        # active request's (growing) prompt/outputs
        self._sync = StateSync()

    def execute(self, scheduled, states) -> dict:
        if self._fatal_error is not None:
            raise RuntimeError(
                "tensor-parallel runner is unavailable after a fatal step failure"
            ) from self._fatal_error
        chunks = tuple(scheduled)
        delta = self._sync.diff(chunks, states)
        try:
            self._comm.broadcast(delta, src=0)
            view = self._sync.apply(delta)  # reconstructs snapshot_step() exactly
            return self._local.execute(chunks, view)
        except Exception as error:
            # Once one TP rank misses a step, retrying on the same groups can only
            # diverge their collective sequences further. Surface a fatal health
            # state and require process replacement.
            self._fatal_error = error
            raise

    @property
    def fatal_error(self) -> Exception | None:
        return self._fatal_error

    def release(self, request_id: str) -> None:
        try:
            if self._fatal_error is None:
                self._comm.broadcast(ReleaseRequest(request_id), src=0)
        except Exception as error:
            self._fatal_error = error
            raise
        finally:
            release = getattr(self._local, "release", None)
            if release is not None:
                release(request_id)

    def shutdown(self) -> None:
        if self._fatal_error is None:
            self._comm.broadcast(_SHUTDOWN, src=0)

    def invalidate_graphs(self) -> None:
        invalidate = getattr(self._local, "invalidate_graphs", None)
        if invalidate is not None:
            invalidate()


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


class _DeferredComm:
    """Forwards to whichever communicator is bound to it.

    The model's `RowParallelLinear` wrappers capture a communicator at build
    time, but the serving groups must not exist until every failure-prone startup
    step has succeeded — otherwise a rank-local failure leaves an UNaborted
    subgroup whose teardown waits on peers that will never arrive (review [P1] on
    #129). Building against this proxy and binding afterwards keeps both: the
    load happens on the startup group, the model runs on the serving tensor one.
    """

    def __init__(self, target) -> None:
        self._target = target

    def bind(self, target) -> None:
        self._target = target

    @property
    def group(self):
        return self._target.group

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_target"), name)


@dataclass(frozen=True)
class ServingGroups:
    """Operational groups with control and model collectives kept disjoint.

    ``broadcast_object_list`` is not one NCCL operation: it broadcasts metadata
    and payload tensors, then receivers copy the payload back to the host before
    deserializing it.  A source rank can therefore enqueue the following model
    all-reduce while peers are still completing the object broadcast.  Keeping
    the Python control protocol on gloo makes that hand-off blocking and leaves
    the NCCL group with tensor collectives only.
    """

    control: object
    model: object


def serving_group(backend: str, *, timeout_s: float = _SERVE_OP_TIMEOUT_S):
    """One process group carrying the OPERATIONAL collective timeout.

    Every rank must call this at the same point — it is itself a collective — so
    it is created right after the rendezvous, before the slow shard load, while
    the ranks are still in lockstep.
    """
    from datetime import timedelta

    import torch.distributed as dist

    return dist.new_group(
        timeout=timedelta(seconds=timeout_s), backend=backend
    )


def serving_groups(
    model_backend: str,
    *,
    control_timeout_s: float = _CONTROL_IDLE_TIMEOUT_S,
    model_timeout_s: float = _SERVE_OP_TIMEOUT_S,
) -> ServingGroups:
    """Create control/model groups in the same order on every rank.

    The control timeout must cover the server's idle lifetime because workers
    wait *inside* its receive. The model group has no pending operation while
    idle, so it keeps the short fail-fast bound.
    """
    return ServingGroups(
        control=serving_group("gloo", timeout_s=control_timeout_s),
        model=serving_group(model_backend, timeout_s=model_timeout_s),
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
    graph_scratch_page: int | None = None,
    graph_max_batch: int = 0,
    graph_max_pages: int = 0,
    graph_warmup_iters: int = 3,
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
    graph_options = {}
    if graph_scratch_page is not None:
        from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend

        graph_options = {
            "graph_backend": CudaGraphBackend(warmup_iters=graph_warmup_iters),
            "graph_max_batch": graph_max_batch,
            "graph_max_pages": graph_max_pages,
            "graph_scratch_page": graph_scratch_page,
        }
    runner = PagedModelRunner(
        model,
        pool,
        sampler=Sampler(vocab_provider=lambda: vocab_table),
        **graph_options,
    )
    return runner, full_config


def _tp_worker_entry(
    spawn_index: int, world_size: int, init_file: str,
    model_dir: str, num_pages: int, page_size: int, vocab: list[str],
    force_cpu: bool = False,
    graph_scratch_page: int | None = None,
    graph_max_batch: int = 0,
    graph_max_pages: int = 0,
    graph_warmup_iters: int = 3,
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
    comm = _DeferredComm(startup_comm)
    runner, _ = build_tp_runner(
        model_dir,
        world_size,
        rank,
        comm,
        num_pages,
        page_size,
        vocab,
        placement,
        graph_scratch_page,
        graph_max_batch,
        graph_max_pages,
        graph_warmup_iters,
    )
    # the handshake is the collective that absorbs load skew, so it — and only it
    # — runs on the long-timeout startup group
    handshake = startup_comm.broadcast(None, src=0)
    validate_handshake(handshake, model_dir, num_pages, page_size)
    groups = serving_groups(placement.backend)
    comm.bind(
        TorchDistCommunicator(group=groups.model, device=placement.device)
    )
    control_comm = TorchDistCommunicator(group=groups.control)
    try:
        worker_step_loop(control_comm, runner)
    finally:
        import torch.distributed as dist

        invalidate = getattr(runner, "invalidate_graphs", None)
        if invalidate is not None:
            invalidate()
        if placement.backend == "nccl":
            # Captured NCCL collectives retain graph-owned communicator work
            # until the graph objects are released and their stream is drained.
            # Drain it, then rendezvous on the model communicator so no rank
            # destroys that communicator while a peer is still releasing its
            # captured work.
            torch.cuda.synchronize()
            comm.barrier()
        dist.destroy_process_group(comm.group)
        dist.destroy_process_group(control_comm.group)
        dist.destroy_process_group()


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
        graph_scratch_page: int | None = None,
        graph_max_batch: int = 0,
        graph_max_pages: int = 0,
        graph_warmup_iters: int = 3,
    ) -> None:
        import tempfile

        import torch
        import torch.multiprocessing as mp

        from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

        # a fresh, not-yet-created path is the file:// rendezvous point
        self._init_file = tempfile.mktemp(prefix="kairyu-tp-")  # noqa: S306
        placement = tp_placement(tp, 0, force_cpu)
        self._placement_backend = placement.backend
        if graph_scratch_page is not None and placement.backend != "nccl":
            raise ValueError("CUDA graph decode needs CUDA/NCCL TP placement")
        # force_cpu travels to the workers: rank 0 on host memory while the
        # spawned ranks probed their way onto GPUs would deadlock the first
        # all_reduce on mismatched backends
        self._ctx = mp.spawn(
            _tp_worker_entry,
            args=(
                tp,
                self._init_file,
                model_dir,
                num_pages,
                page_size,
                vocab,
                force_cpu,
                graph_scratch_page,
                graph_max_batch,
                graph_max_pages,
                graph_warmup_iters,
            ),
            nprocs=tp - 1,
            join=False,
        )
        # Everything past the spawn can raise — an indivisible TP degree, a
        # missing tensor, not enough GPUs. Without this the workers and the
        # process group outlive the failed constructor: the caller sees the real
        # error, then a "destroy_process_group() was not called" warning, and the
        # next launcher in the same process cannot rendezvous at all.
        try:
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
            # the model is built against the startup group; nothing that can fail
            # may leave a serving subgroup behind for _abandon_start to miss
            self._comm = _DeferredComm(startup_comm)
            runner, self.full_config = build_tp_runner(
                model_dir,
                tp,
                0,
                self._comm,
                num_pages,
                page_size,
                vocab,
                placement,
                graph_scratch_page,
                graph_max_batch,
                graph_max_pages,
                graph_warmup_iters,
            )
            # the one collective that legitimately absorbs load skew
            startup_comm.broadcast(make_handshake(model_dir, num_pages, page_size), src=0)
            # Every failure-prone step is done: now the step loop gets bounded
            # operational groups.  Python state deltas stay on gloo; the model
            # wrappers use only the tensor/NCCL group.
            groups = serving_groups(placement.backend)
            self._comm.bind(
                TorchDistCommunicator(
                    group=groups.model, device=placement.device
                )
            )
            self._control_comm = TorchDistCommunicator(group=groups.control)
            self.runner = DistTPModelRunner(self._control_comm, runner)
        except BaseException:
            self._abandon_start()
            raise

    def _abandon_start(self) -> None:
        """Tear down a half-built group without waiting on it.

        Order is the OPPOSITE of the normal shutdown's. There the ranks are
        healthy and destroy is a rendezvous; here they are dead of the same error
        or stuck in a collective nobody will complete, so `destroy_process_group`
        — which every rank must reach — can BLOCK rather than return the original
        error. `contextlib.suppress` catches an exception but cannot bound that
        (review [P1] on #129).

        So the communicator is aborted first, which is non-collective; after that
        nothing can block, and the workers are terminated and reaped. Every step
        is best-effort: this runs while an exception is in flight and must not
        replace it with one of its own.
        """
        import contextlib
        import os

        import torch.distributed as dist

        if dist.is_initialized():
            with contextlib.suppress(Exception):
                self._abort_communicator()
            with contextlib.suppress(Exception):
                dist.destroy_process_group()
        for process in self._ctx.processes:
            if process.is_alive():
                process.terminate()
        for process in self._ctx.processes:
            with contextlib.suppress(Exception):
                process.join(timeout=10)
            if process.is_alive():  # pragma: no cover - terminate was ignored
                with contextlib.suppress(Exception):
                    process.kill()
                    process.join(timeout=5)
        with contextlib.suppress(OSError):
            os.unlink(self._init_file)

    @staticmethod
    def _abort_communicator() -> None:
        """NCCL only: drop the communicator without waiting for peers.

        gloo needs none, and the hook is absent on older torch — both are
        "nothing to abort" rather than an error.
        """
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            return
        group = dist.distributed_c10d._get_default_group()
        backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
        abort = getattr(backend, "abort", None) or getattr(backend, "_abort", None)
        if abort is not None:
            abort()

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

    def failure_type(self) -> str | None:
        """Fatal TP step failure, sanitized for the unauthenticated health API."""
        error = self.runner.fatal_error
        return type(error).__name__ if error is not None else None

    def shutdown(self) -> None:
        import contextlib
        import os

        import torch
        import torch.distributed as dist

        if self.runner.fatal_error is not None:
            # The collective sequence is already untrustworthy. A graceful
            # broadcast/barrier can only hang; abort and reap like failed startup.
            self._abandon_start()
            return
        self.runner.shutdown()  # broadcasts None -> workers leave worker_step_loop
        self.runner.invalidate_graphs()
        if self._placement_backend == "nccl":
            torch.cuda.synchronize()
            # Match the worker-side rendezvous after every rank has dropped its
            # CUDA graphs.  Without this, TP graph serving can complete all
            # inference steps and then hang forever in process-group teardown.
            self._comm.barrier()
        # BEFORE the join, not after: NCCL's destroy_process_group waits for every
        # rank to reach it, so joining first deadlocks rank 0 against workers that
        # are already sitting in their own destroy. gloo never blocks here, which
        # is why the CPU parity gates could not see this.
        if dist.is_initialized():
            # Multiple process groups must be destroyed explicitly in the same
            # order on every rank.  Reverse creation order keeps the graph-owning
            # NCCL subgroup ahead of the gloo control and startup groups.
            dist.destroy_process_group(self._comm.group)
            dist.destroy_process_group(self._control_comm.group)
            dist.destroy_process_group()
        self._ctx.join()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._init_file)
