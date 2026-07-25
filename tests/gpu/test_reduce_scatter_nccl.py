"""reduce_scatter over a real NCCL group (m16 D1's recorded deploy-day gap)."""

import json
import time
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from tests.dist import dist_targets

pytestmark = pytest.mark.gpu

WORLD_SIZE = 2
SPAWN_TIMEOUT_S = 120


def test_nccl_reduce_scatter_matches_all_reduce_sliced(tmp_path: Path) -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("NCCL reduce_scatter needs 2 CUDA devices; CUDA is unavailable")
    if torch.cuda.device_count() < WORLD_SIZE:  # pragma: no cover - small box
        pytest.skip(f"need {WORLD_SIZE} CUDA devices, found {torch.cuda.device_count()}")

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    context = mp.start_processes(
        dist_targets.reduce_scatter_nccl_equivalence,
        args=(WORLD_SIZE, str(tmp_path / "rdv-rs-nccl"), str(out_dir)),
        nprocs=WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    while not context.join(timeout=1):
        if time.monotonic() > deadline:  # pragma: no cover - hang guard
            for process in context.processes:
                process.terminate()
            pytest.fail(f"nccl reduce_scatter timed out after {SPAWN_TIMEOUT_S}s")

    for rank in range(WORLD_SIZE):
        result = json.loads((out_dir / f"rank{rank}.json").read_text())
        assert result["shape"] == [4, 3], rank
        # the real collective must land on the same numbers as the gloo path's
        # all_reduce-then-slice, or the two backends mean different things
        assert result["max_error"] == 0.0, rank
        assert result["device"] == f"cuda:{rank}", rank
