"""Is reduce_scatter+all_gather actually cheaper than all_reduce here? (m16 D1)

m16 records reduce_scatter as a "same-call-site optimization ... for deploy day".
It is not one on its own: `RowParallelLinear` needs the FULL sum, and handing it
a shard changes what the next layer reads. Trading one for the other is sequence
parallelism — the shard has to survive the norm and be re-gathered by the next
column-parallel matmul.

So the question this answers is narrow and worth having a number for: on THIS
fabric, does rs+ag beat ar? If it does not, sequence parallelism buys activation
memory and sharded norms, not comm time, and the design note should say so.

Run: uv run torchrun --nproc-per-node 8 bench/reduce_scatter_bench.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _time(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192, help="tokens in the batch")
    parser.add_argument("--hidden", type=int, default=5120, help="Qwen3-32B hidden size")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    try:
        device = torch.device("cuda", rank)
        payload = torch.randn(args.rows, args.hidden, dtype=torch.bfloat16, device=device)
        shard = torch.empty(
            (args.rows // world, args.hidden), dtype=torch.bfloat16, device=device
        )

        def all_reduce():
            buffer = payload.clone()
            dist.all_reduce(buffer)

        def reduce_scatter_all_gather():
            buffer = payload.clone()
            dist.reduce_scatter_tensor(shard, buffer)
            dist.all_gather_into_tensor(buffer, shard)

        ar = _time(all_reduce, args.iters)
        rs_ag = _time(reduce_scatter_all_gather, args.iters)
        # rs alone is the sequence-parallel half: what the swap costs if the
        # consumer can live with a shard
        rs = _time(lambda: dist.reduce_scatter_tensor(shard, payload.clone()), args.iters)

        if rank == 0:
            bytes_moved = payload.numel() * payload.element_size()
            payload_out = {
                "world_size": world,
                "rows": args.rows,
                "hidden": args.hidden,
                "dtype": "bfloat16",
                "tensor_mb": round(bytes_moved / 1e6, 2),
                "all_reduce_ms": round(ar * 1e3, 3),
                "reduce_scatter_ms": round(rs * 1e3, 3),
                "reduce_scatter_all_gather_ms": round(rs_ag * 1e3, 3),
                "rs_ag_speedup_vs_ar": round(ar / rs_ag, 3),
                "rs_only_speedup_vs_ar": round(ar / rs, 3),
                "note": (
                    "rs+ag replaces ar with identical semantics; rs alone is the "
                    "sequence-parallel half and only valid where the consumer "
                    "accepts a shard"
                ),
            }
            print(json.dumps(payload_out, indent=2))
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(payload_out, indent=2) + "\n")
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
