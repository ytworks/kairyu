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
from pathlib import Path

from kairyu.engine.core.step_input import StateSync, StepDelta

_SHUTDOWN = None


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
        assert isinstance(payload, StepDelta)
        view = sync.apply(payload)  # same delta -> same reconstructed states
        local_runner.execute(payload.chunks, view)
        steps += 1


def build_tp_runner(model_dir: str, tp: int, rank: int, comm, num_pages: int, page_size: int):
    """The per-rank sharded PagedModelRunner (pool sized from the tp_view config)."""
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.sampler import Sampler
    from kairyu.models.parallel import build_tp_model

    model, local_config, full_config = build_tp_model(model_dir, tp, rank, comm)
    pool = PagedKVPool(
        num_layers=local_config.num_hidden_layers,
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=local_config.kv_cache_num_heads,
        head_dim=local_config.kv_cache_head_dim,
    )
    runner = PagedModelRunner(model, pool, sampler=Sampler())
    return runner, full_config


def _tp_worker_entry(
    spawn_index: int, world_size: int, init_file: str,
    model_dir: str, num_pages: int, page_size: int,
) -> None:
    """Spawned worker (rank = spawn_index + 1; rank 0 is the driver process).

    Module-level and side-effect-free at import (m16 A6) so torch spawn can
    pickle it. Joins the group, validates the handshake, runs the step loop
    until rank 0 broadcasts shutdown, then tears the group down."""
    import torch

    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

    rank = spawn_index + 1
    torch.set_num_threads(1)
    init_distributed(rank, world_size, f"file://{init_file}")
    comm = TorchDistCommunicator()
    runner, _ = build_tp_runner(model_dir, world_size, rank, comm, num_pages, page_size)
    handshake = comm.broadcast(None, src=0)
    validate_handshake(handshake, model_dir, num_pages, page_size)
    try:
        worker_step_loop(comm, runner)
    finally:
        import torch.distributed as dist

        dist.destroy_process_group()


class DistTPLauncher:
    """Owns the spawned worker processes + the rank-0 DistTPModelRunner.

    Wires real multi-process TP into a single-process serve path: rank 0 lives in
    THIS process, ranks 1..tp-1 are spawned workers. ``shutdown()`` broadcasts the
    terminating None (worker_step_loop returns), joins the workers, and destroys
    the rank-0 group — so ``kairyu serve --tp N`` starts and stops cleanly."""

    def __init__(self, model_dir: str, tp: int, num_pages: int, page_size: int) -> None:
        import tempfile

        import torch.multiprocessing as mp

        from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

        # a fresh, not-yet-created path is the gloo file:// rendezvous point
        self._init_file = tempfile.mktemp(prefix="kairyu-tp-")  # noqa: S306
        self._ctx = mp.spawn(
            _tp_worker_entry,
            args=(tp, self._init_file, model_dir, num_pages, page_size),
            nprocs=tp - 1,
            join=False,
        )
        init_distributed(0, tp, f"file://{self._init_file}")
        self._comm = TorchDistCommunicator()
        runner, self.full_config = build_tp_runner(
            model_dir, tp, 0, self._comm, num_pages, page_size
        )
        self._comm.broadcast(make_handshake(model_dir, num_pages, page_size), src=0)
        self.runner = DistTPModelRunner(self._comm, runner)

    def shutdown(self) -> None:
        self.runner.shutdown()  # broadcasts None -> workers leave worker_step_loop
        self._ctx.join()
        import contextlib
        import os

        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._init_file)
