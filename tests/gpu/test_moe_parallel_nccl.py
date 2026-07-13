"""Two-rank CUDA expert-parallel dispatch/return parity over NCCL."""

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
FLOAT32_TOLERANCE = 1e-6


def test_ep_dispatch_and_return_match_reference_over_nccl(tmp_path: Path) -> None:
    if not torch.cuda.is_available():  # pragma: no cover - deploy-day only
        pytest.skip("NCCL EP parity requires 2 CUDA devices; CUDA is unavailable")
    device_count = torch.cuda.device_count()
    if device_count < WORLD_SIZE:  # pragma: no cover - deploy-day only
        pytest.skip(
            f"NCCL EP parity requires 2 CUDA devices; found {device_count}"
        )

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    init_file = tmp_path / "rdv-ep-block-nccl-parity"
    context = mp.start_processes(
        dist_targets.ep_block_nccl_parity,
        args=(WORLD_SIZE, str(init_file), str(out_dir)),
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
                f"ep_block_nccl_parity timed out after {SPAWN_TIMEOUT_S}s"
            )

    for rank in range(WORLD_SIZE):
        result_file = out_dir / f"rank{rank}.json"
        assert result_file.is_file(), f"rank {rank} produced no result"
        result = json.loads(result_file.read_text())
        assert result["local_experts"] == 2
        assert result["device"] == f"cuda:{rank}"
        assert result["max_error"] <= FLOAT32_TOLERANCE
