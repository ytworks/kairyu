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

from kairyu.engine.core.step_input import StepInput, snapshot_step

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

    def execute(self, scheduled, states) -> dict:
        step = snapshot_step(scheduled, states)
        self._comm.broadcast(step, src=0)
        return self._local.execute(step.chunks, step.states_view())

    def shutdown(self) -> None:
        self._comm.broadcast(_SHUTDOWN, src=0)


def worker_step_loop(comm, local_runner) -> int:
    """Non-zero-rank main loop: execute broadcast steps until shutdown.

    Returns the number of steps executed (spawn tests assert on it).
    """
    steps = 0
    while True:
        payload = comm.broadcast(_SHUTDOWN, src=0)
        if payload is _SHUTDOWN or payload is None:
            return steps
        assert isinstance(payload, StepInput)
        local_runner.execute(payload.chunks, payload.states_view())
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
