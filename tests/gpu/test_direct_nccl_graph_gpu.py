"""Direct-NCCL graph captures retain superseded workspace allocations."""

import json
import time
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from tests.dist import dist_targets

pytestmark = pytest.mark.gpu

WORLD_SIZE = 2
SPAWN_TIMEOUT_S = 180


def test_small_graph_replays_after_larger_workspace_capture(tmp_path: Path) -> None:
    if not torch.cuda.is_available():  # pragma: no cover - deploy-day only
        pytest.skip("direct NCCL graph regression requires CUDA")
    if torch.cuda.device_count() < WORLD_SIZE:  # pragma: no cover - small box
        pytest.skip(
            f"direct NCCL graph regression requires {WORLD_SIZE} CUDA devices"
        )

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    context = mp.start_processes(
        dist_targets.direct_nccl_graph_workspace_growth,
        args=(
            WORLD_SIZE,
            str(tmp_path / "rdv-direct-nccl-graph-growth"),
            str(out_dir),
        ),
        nprocs=WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    while not context.join(timeout=1):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.terminate()
            pytest.fail(
                f"direct NCCL graph regression timed out after "
                f"{SPAWN_TIMEOUT_S}s"
            )

    for rank in range(WORLD_SIZE):
        result = json.loads(
            (out_dir / f"rank{rank}.json").read_text(encoding="utf-8")
        )
        assert result == {
            "max_error": 0.0,
            "small_gather_retained": True,
            "small_reduce_retained": True,
            "direct_nccl_active": True,
        }
