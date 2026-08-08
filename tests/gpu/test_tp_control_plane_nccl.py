"""TP step deltas use NCCL tensors while rare controls stay on Gloo."""

import json
import time
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from tests.dist import dist_targets

pytestmark = pytest.mark.gpu

WORLD_SIZE = 2
ROUNDS = 512
SPAWN_TIMEOUT_S = 120


def test_tp_step_delta_uses_nccl_tensor_control_group(tmp_path: Path) -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("TP control-plane gate needs 2 CUDA devices; CUDA is unavailable")
    if torch.cuda.device_count() < WORLD_SIZE:  # pragma: no cover - small box
        pytest.skip(f"need {WORLD_SIZE} CUDA devices, found {torch.cuda.device_count()}")

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    context = mp.start_processes(
        dist_targets.tp_control_model_group_stress,
        args=(
            WORLD_SIZE,
            str(tmp_path / "rdv-tp-control-model"),
            str(out_dir),
            ROUNDS,
        ),
        nprocs=WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    while not context.join(timeout=1):
        if time.monotonic() > deadline:  # pragma: no cover - hang guard
            for process in context.processes:
                process.terminate()
            pytest.fail(f"TP control/model stress timed out after {SPAWN_TIMEOUT_S}s")

    for rank in range(WORLD_SIZE):
        result_file = out_dir / f"rank{rank}.json"
        assert result_file.is_file(), f"rank {rank} produced no result"
        result = json.loads(result_file.read_text())
        assert result == {
            "control_backend": "gloo",
            "model_backend": "nccl",
            "steps": ROUNDS,
            "last_token": ROUNDS,
            "mode_changes": 1,
            "gloo_step_delta_objects": 0,
            "object_broadcasts": 2,
            "step_headers": ROUNDS,
            "object_headers": 2,
            "control_tensor_broadcasts": ROUNDS + 2,
            "step_payloads": ROUNDS,
            "model_tensor_broadcasts": 2 * ROUNDS,
        }
