"""Two-rank NCCL smoke for DeepSeek V4 fixed packed-FP4 EP."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kairyu.kernels.deepseek_v4_moe_gpu import deepseek_v4_fp4_ep_forward

pytestmark = pytest.mark.gpu


class _ZeroPackedExperts:
    def __init__(self, device: torch.device) -> None:
        ue8m0 = torch.float8_e8m0fnu
        self.gate_up_proj = torch.zeros((2, 128, 64), dtype=torch.int8, device=device)
        self.gate_up_proj_scale_inv = torch.ones(
            (2, 128, 4), dtype=ue8m0, device=device
        )
        self.down_proj = torch.zeros((2, 128, 32), dtype=torch.int8, device=device)
        self.down_proj_scale_inv = torch.ones(
            (2, 128, 2), dtype=ue8m0, device=device
        )

    @staticmethod
    def _apply_gate(value: torch.Tensor) -> torch.Tensor:
        return value[..., :64]


def _ep_target(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(f"cuda:{rank}")
        tokens = 3 if rank == 0 else 1
        hidden = torch.randn(tokens, 128, dtype=torch.bfloat16, device=device)
        indices = torch.tensor(
            [[0, 3], [1, 2], [2, 0]][:tokens], dtype=torch.long, device=device
        )
        weights = torch.full(
            (tokens, 2), 0.5, dtype=torch.bfloat16, device=device
        )
        output = deepseek_v4_fp4_ep_forward(
            _ZeroPackedExperts(device),
            hidden,
            indices,
            weights,
            process_group=dist.group.WORLD,
        )
        torch.cuda.synchronize(device)
        payload = {
            "rank": rank,
            "tokens": tokens,
            "shape": list(output.shape),
            "max_abs": float(output.abs().max().item()),
        }
        (Path(output_dir) / f"rank{rank}.json").write_text(json.dumps(payload))
    finally:
        dist.destroy_process_group()


def test_deepseek_v4_fixed_fp4_ep_over_nccl(tmp_path: Path) -> None:
    world_size = 2
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip("DeepSeek V4 EP smoke requires two CUDA devices")
    if any(torch.cuda.get_device_capability(rank) != (12, 0) for rank in range(world_size)):
        pytest.skip("DeepSeek V4 FP4 production gate requires SM120")
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    context = mp.start_processes(
        _ep_target,
        args=(world_size, str(tmp_path / "rdv"), str(output_dir)),
        nprocs=world_size,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + 180
    while not context.join(timeout=1):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.terminate()
            pytest.fail("DeepSeek V4 FP4 EP smoke timed out")
    for rank, tokens in enumerate((3, 1)):
        payload = json.loads((output_dir / f"rank{rank}.json").read_text())
        assert payload == {
            "rank": rank,
            "tokens": tokens,
            "shape": [tokens, 128],
            "max_abs": 0.0,
        }
