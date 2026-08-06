"""EP4 attention-DP captures direct-NCCL CUDA graphs before readiness."""

import json
import time
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from tests.dist import dist_targets

pytestmark = pytest.mark.gpu

WORLD_SIZE = 4
SPAWN_TIMEOUT_S = 180


def test_first_ep4_decode_replays_the_ready_direct_nccl_graph(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():  # pragma: no cover - deploy-day only
        pytest.skip("EP4 attention-DP CUDA graph gate requires CUDA")
    device_count = torch.cuda.device_count()
    if device_count < WORLD_SIZE:  # pragma: no cover - small box
        pytest.skip(
            f"EP4 attention-DP CUDA graph gate requires {WORLD_SIZE} CUDA "
            f"devices; found {device_count}"
        )

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    context = mp.start_processes(
        dist_targets.ep4_attention_dp_cuda_graph_startup_replay,
        args=(
            WORLD_SIZE,
            str(tmp_path / "rdv-ep4-attention-dp-graph-startup"),
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
            pytest.fail(f"EP4 attention-DP CUDA graph gate timed out after {SPAWN_TIMEOUT_S}s")

    for rank in range(WORLD_SIZE):
        result_file = out_dir / f"rank{rank}.json"
        assert result_file.is_file(), f"rank {rank} produced no teardown result"
        result = json.loads(result_file.read_text(encoding="utf-8"))
        assert result == {
            "rank": rank,
            "device": f"cuda:{rank}",
            "direct_active": True,
            "startup_control_distinct": True,
            "ready": {
                "captured": [1],
                "backend_captures": 1,
                "backend_replays": 0,
                "graph_executions": 0,
                "fallbacks": 0,
                "forward_calls": 2,
                "completed_layout_steps": 1,
                "layout_queue_depth": 0,
                "direct_gather_capture_buffers": 4,
                "direct_scatter_capture_buffers": 1,
            },
            "first_decision": "replay",
            "after_backend_captures": 1,
            "after_backend_replays": 1,
            "after_graph_executions": 1,
            "after_fallbacks": 0,
            "after_forward_calls": 2,
            "after_completed_layout_steps": 1,
            "max_error": 0.0,
            "direct_closed": True,
            "distributed_destroyed": True,
            "normal_teardown": True,
        }
