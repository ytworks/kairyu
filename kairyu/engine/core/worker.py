"""SPMD TP execution: driver-side runner + worker main (m16 D4).

Rank 0 owns the scheduler/EngineCore, broadcasts each immutable state delta,
and is the sole sampling authority. Every rank executes the same model/KV step,
then rank 0 broadcasts one fixed-layout device token packet over the model
communicator. All ranks adopt that packet before advancing. Shutdown remains a
``None`` control broadcast (A11).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from kairyu.engine.core.step_input import StateSync, StepDelta
from kairyu.engine.tokenizer import GrammarVocabulary

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


@dataclass(frozen=True)
class _SamplingOwnershipProbe:
    """Out-of-band request for rank-local TP sampling metadata."""


_SAMPLING_OWNERSHIP_FIELDS = frozenset(
    {
        "rank",
        "control_world_size",
        "control_backend",
        "model_world_size",
        "model_backend",
        "sampling_owner",
        "sampler_present",
        "device",
    }
)


def _communicator_backend(comm) -> str:
    """Return the real process-group backend without labeling it externally."""
    from kairyu.engine.core.comm import FakeCommunicator

    if isinstance(comm, FakeCommunicator):
        return "fake"

    import torch.distributed as dist

    try:
        backend = str(dist.get_backend(comm.group))
    except Exception as error:
        raise RuntimeError(
            "cannot inspect communicator backend for "
            f"{type(comm).__name__}; expected FakeCommunicator or a "
            "torch.distributed communicator with a process group"
        ) from error
    if not backend:
        raise RuntimeError(f"communicator backend for {type(comm).__name__} is empty")
    return backend


def _sampling_ownership_row(control_comm, model_comm, local_runner) -> dict[str, object]:
    """Build one rank's JSON-serializable ownership/topology observation."""
    control_rank = control_comm.rank
    model_rank = model_comm.rank
    if type(control_rank) is not int or type(model_rank) is not int:
        raise RuntimeError(
            "sampling ownership rank metadata must be integers: "
            f"control={control_rank!r}, model={model_rank!r}"
        )
    if control_rank != model_rank:
        raise RuntimeError(
            "sampling ownership communicator rank mismatch: "
            f"control rank={control_rank}, model rank={model_rank}"
        )

    sampling_owner = getattr(local_runner, "sampling_owner", None)
    if type(sampling_owner) is not bool:
        raise RuntimeError(
            f"rank {control_rank} runner has malformed sampling_owner="
            f"{sampling_owner!r}; expected bool"
        )
    if not hasattr(local_runner, "_sampler"):
        raise RuntimeError(f"rank {control_rank} runner does not expose sampler ownership")
    if not hasattr(local_runner, "_device"):
        raise RuntimeError(f"rank {control_rank} runner does not expose its compute device")

    control_world_size = control_comm.world_size
    model_world_size = model_comm.world_size
    if (
        type(control_world_size) is not int
        or control_world_size < 1
        or type(model_world_size) is not int
        or model_world_size < 1
    ):
        raise RuntimeError(
            f"rank {control_rank} has malformed communicator world sizes: "
            f"control={control_world_size!r}, model={model_world_size!r}"
        )
    device = str(local_runner._device)
    if not device:
        raise RuntimeError(f"rank {control_rank} runner device is empty")

    return {
        "rank": control_rank,
        "control_world_size": control_world_size,
        "control_backend": _communicator_backend(control_comm),
        "model_world_size": model_world_size,
        "model_backend": _communicator_backend(model_comm),
        "sampling_owner": sampling_owner,
        "sampler_present": local_runner._sampler is not None,
        "device": device,
    }


def _validate_sampling_ownership_rows(
    rows: object,
    *,
    control_world_size: int,
    model_world_size: int,
    control_backend: str,
    model_backend: str,
) -> tuple[dict[str, object], ...]:
    """Reject incomplete or forged topology evidence with rank-specific errors."""
    if not isinstance(rows, (tuple, list)):
        raise RuntimeError(
            "sampling ownership replies are malformed: expected a rank sequence, "
            f"got {type(rows).__name__}"
        )
    if len(rows) != control_world_size:
        reported = {
            row.get("rank")
            for row in rows
            if isinstance(row, dict) and type(row.get("rank")) is int
        }
        missing = sorted(set(range(control_world_size)) - reported)
        raise RuntimeError(
            "sampling ownership reply count mismatch: "
            f"expected {control_world_size}, received {len(rows)}; "
            f"missing ranks={missing}"
        )

    by_rank: dict[int, dict[str, object]] = {}
    for slot, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"sampling ownership reply in gather slot {slot} is malformed: "
                f"expected dict, got {type(row).__name__}"
            )
        fields = set(row)
        if fields != _SAMPLING_OWNERSHIP_FIELDS:
            missing_fields = sorted(_SAMPLING_OWNERSHIP_FIELDS - fields)
            extra_fields = sorted(fields - _SAMPLING_OWNERSHIP_FIELDS)
            raise RuntimeError(
                f"sampling ownership reply in gather slot {slot} has malformed "
                f"fields: missing={missing_fields}, extra={extra_fields}"
            )
        rank = row["rank"]
        if type(rank) is not int:
            raise RuntimeError(
                f"sampling ownership reply in gather slot {slot} has non-integer rank={rank!r}"
            )
        if rank in by_rank:
            raise RuntimeError(
                f"sampling ownership replies contain duplicate rank {rank} (gather slot {slot})"
            )
        if not 0 <= rank < control_world_size:
            raise RuntimeError(
                f"sampling ownership reply in gather slot {slot} has rank {rank}; "
                f"expected [0, {control_world_size})"
            )
        typed_fields = {
            "control_world_size": int,
            "control_backend": str,
            "model_world_size": int,
            "model_backend": str,
            "sampling_owner": bool,
            "sampler_present": bool,
            "device": str,
        }
        for name, field_type in typed_fields.items():
            if type(row[name]) is not field_type:
                raise RuntimeError(
                    f"sampling ownership reply for rank {rank} has malformed "
                    f"{name}={row[name]!r}; expected {field_type.__name__}"
                )
        expected = {
            "control_world_size": control_world_size,
            "control_backend": control_backend,
            "model_world_size": model_world_size,
            "model_backend": model_backend,
        }
        mismatches = {
            name: (row[name], value) for name, value in expected.items() if row[name] != value
        }
        if mismatches:
            raise RuntimeError(
                f"sampling ownership reply for rank {rank} disagrees with rank 0 "
                f"topology (reported, expected)={mismatches}"
            )
        if not row["device"]:
            raise RuntimeError(f"sampling ownership reply for rank {rank} has an empty device")
        by_rank[rank] = row

    missing = sorted(set(range(control_world_size)) - set(by_rank))
    if missing:
        raise RuntimeError(f"sampling ownership replies are missing ranks={missing}")
    return tuple(by_rank[rank] for rank in sorted(by_rank))


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
    """Driver-side ModelRunner with rank-0-authoritative sampling.

    Drops in where ``TPModelRunner`` sits: the driver's own rank-0 shard runs
    inside this call and samples once. Non-zero ranks execute the passive
    model/KV path, then all ranks adopt rank 0's device token packet.
    """

    def __init__(self, control_comm, local_runner, model_comm=None) -> None:
        self._control_comm = control_comm
        self._model_comm = model_comm if model_comm is not None else control_comm
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
            self._control_comm.broadcast(delta, src=0)
            view = self._sync.apply(delta)  # reconstructs snapshot_step() exactly
            sampled = self._local.execute(chunks, view)
            packet = self._local.make_sampling_token_packet(chunks, view, sampled=sampled)
            packet = self._model_comm.tensor_broadcast(packet, src=0)
            self._local.adopt_sampling_token_packet(chunks, view, packet)
            return sampled
        except Exception as error:
            # Once one TP rank misses a step, retrying on the same groups can only
            # diverge their collective sequences further. Surface a fatal health
            # state and require process replacement.
            self._fatal_error = error
            raise

    @property
    def fatal_error(self) -> Exception | None:
        return self._fatal_error

    def sampling_ownership_metadata(self) -> tuple[dict[str, object], ...]:
        """Collect rank-observed ownership/topology outside the inference path.

        This explicit diagnostic is the only place the metadata all-gather runs;
        normal steps retain only their state and sampled-token broadcasts.
        """
        if self._fatal_error is not None:
            raise RuntimeError(
                "tensor-parallel runner is unavailable after a fatal step failure"
            ) from self._fatal_error
        try:
            delivered = self._control_comm.broadcast(_SamplingOwnershipProbe(), src=0)
            if not isinstance(delivered, _SamplingOwnershipProbe):
                raise RuntimeError(
                    "sampling ownership probe broadcast returned a malformed "
                    f"payload: {type(delivered).__name__}"
                )
            local = _sampling_ownership_row(self._control_comm, self._model_comm, self._local)
            rows = self._control_comm.all_gather(local)
            return _validate_sampling_ownership_rows(
                rows,
                control_world_size=local["control_world_size"],
                model_world_size=local["model_world_size"],
                control_backend=local["control_backend"],
                model_backend=local["model_backend"],
            )
        except Exception as error:
            failure = RuntimeError(f"TP sampling ownership metadata probe failed: {error}")
            self._fatal_error = failure
            raise failure from error

    def release(self, request_id: str) -> None:
        try:
            if self._fatal_error is None:
                self._control_comm.broadcast(ReleaseRequest(request_id), src=0)
        except Exception as error:
            self._fatal_error = error
            raise
        finally:
            release = getattr(self._local, "release", None)
            if release is not None:
                release(request_id)

    def shutdown(self) -> None:
        if self._fatal_error is None:
            self._control_comm.broadcast(_SHUTDOWN, src=0)

    def invalidate_graphs(self) -> None:
        invalidate = getattr(self._local, "invalidate_graphs", None)
        if invalidate is not None:
            invalidate()


def worker_step_loop(control_comm, local_runner, model_comm=None) -> int:
    """Non-zero-rank main loop: execute passive steps until shutdown.

    Returns the number of steps executed (spawn tests assert on it).
    """
    if model_comm is None:
        model_comm = control_comm
    steps = 0
    sync = StateSync()
    while True:
        payload = control_comm.broadcast(_SHUTDOWN, src=0)
        if not isinstance(payload, StepDelta):
            if payload is _SHUTDOWN or payload is None:
                return steps
            if isinstance(payload, ReleaseRequest):
                release = getattr(local_runner, "release", None)
                if release is not None:
                    release(payload.request_id)
                continue
            if isinstance(payload, _SamplingOwnershipProbe):
                local = _sampling_ownership_row(control_comm, model_comm, local_runner)
                control_comm.all_gather(local)
                continue
            raise RuntimeError(
                f"TP worker received an unsupported control payload: {type(payload).__name__}"
            )
        view = sync.apply(payload)  # same delta -> same reconstructed states
        local_runner.execute_passive(payload.chunks, view)
        packet = local_runner.make_sampling_token_packet(payload.chunks, view, sampled=None)
        packet = model_comm.tensor_broadcast(packet, src=0)
        local_runner.adopt_sampling_token_packet(payload.chunks, view, packet)
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
            f"tensor_parallel_size={tp} needs {tp} CUDA devices; found {profile.device_count}"
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

    return dist.new_group(timeout=timedelta(seconds=timeout_s), backend=backend)


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
    vocab: list[str] | GrammarVocabulary,
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
    grammar_vocab = (
        vocab if isinstance(vocab, GrammarVocabulary) else GrammarVocabulary(list(vocab))
    )
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
        sampler=(Sampler(vocab_provider=lambda: grammar_vocab) if rank == 0 else None),
        sampling_owner=rank == 0,
        **graph_options,
    )
    return runner, full_config


def _tp_worker_entry(
    spawn_index: int,
    world_size: int,
    init_file: str,
    model_dir: str,
    num_pages: int,
    page_size: int,
    vocab: list[str] | GrammarVocabulary,
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
    comm.bind(TorchDistCommunicator(group=groups.model, device=placement.device))
    control_comm = TorchDistCommunicator(group=groups.control)
    try:
        worker_step_loop(control_comm, runner, comm)
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
        vocab: list[str] | GrammarVocabulary,
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
            self._comm.bind(TorchDistCommunicator(group=groups.model, device=placement.device))
            self._control_comm = TorchDistCommunicator(group=groups.control)
            self.runner = DistTPModelRunner(self._control_comm, runner, self._comm)
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
        # Reap peers before any potentially rendezvous-like process-group
        # destruction. This ordering remains bounded even if an older backend
        # lacks the private abort hook or abort itself raises.
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
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                dist.destroy_process_group()
        with contextlib.suppress(OSError):
            os.unlink(self._init_file)

    def _abort_communicator(self) -> None:
        """NCCL only: drop every live communicator without waiting for peers.

        The operational model subgroup is a distinct NCCL communicator from the
        startup/default group. Aborting only the latter leaves a worker stuck in
        the sampling-token broadcast when rank 0 fails after model execution.
        Gloo needs no abort, and the hook is absent on older torch — both are
        "nothing to abort" rather than an error.
        """
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            return
        groups = []
        model_comm = getattr(self, "_comm", None)
        if model_comm is not None:
            groups.append(model_comm.group)
        groups.append(dist.distributed_c10d._get_default_group())
        seen: set[int] = set()
        device = torch.device("cuda", torch.cuda.current_device())
        for group in groups:
            if id(group) in seen:
                continue
            seen.add(id(group))
            try:
                backend = group._get_backend(device)
            except Exception:
                continue
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
            index + 1 for index, process in enumerate(self._ctx.processes) if not process.is_alive()
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
