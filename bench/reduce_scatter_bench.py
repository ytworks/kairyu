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
from pathlib import Path

import torch
import torch.distributed as dist


def _time(fn, iters: int, warmup: int = 5) -> list[float]:
    """Per-trial WORST-RANK elapsed, in ms.

    A collective finishes when its slowest participant does, so a per-rank
    minimum measures nothing about the collective (review [P1] on #134). Each
    trial is bounded by a barrier so the ranks start together, timed with CUDA
    events, and reduced with MAX across ranks. Every sample is kept — a single
    number cannot show straggler spread.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        elapsed = torch.tensor(
            [start_event.elapsed_time(end_event)], device=torch.cuda.current_device()
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(float(elapsed.item()))
    return samples


def _summary(samples: list[float]) -> dict:
    ordered = sorted(samples)
    count = len(ordered)
    return {
        "median_ms": round(ordered[count // 2], 4),
        "min_ms": round(ordered[0], 4),
        "p95_ms": round(ordered[min(count - 1, int(count * 0.95))], 4),
        "max_ms": round(ordered[-1], 4),
        "samples": [round(value, 4) for value in ordered],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192, help="tokens in the batch")
    parser.add_argument("--hidden", type=int, default=5120, help="Qwen3-32B hidden size")
    parser.add_argument("--iters", type=int, default=20, help="trials per round")
    parser.add_argument("--rounds", type=int, default=5, help="interleaved repeats")
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
        # buffers are prepared OUTSIDE the timed region: an 84 MB clone inside it
        # is a device copy the real call site does not make, and it dilutes the
        # difference the benchmark exists to measure (review [P1] on #134)
        buffer = torch.empty_like(payload)

        def all_reduce():
            dist.all_reduce(buffer)

        def reduce_scatter_all_gather():
            dist.reduce_scatter_tensor(shard, buffer)
            dist.all_gather_into_tensor(buffer, shard)

        def reduce_scatter_only():
            dist.reduce_scatter_tensor(shard, buffer)

        # interleaved, not one path after another: a fixed order lets clock or
        # thermal drift masquerade as a difference between paths
        cases = {
            "all_reduce": all_reduce,
            "reduce_scatter_all_gather": reduce_scatter_all_gather,
            "reduce_scatter": reduce_scatter_only,
        }
        samples: dict[str, list[float]] = {name: [] for name in cases}
        for _ in range(args.rounds):
            for name, fn in cases.items():
                buffer.copy_(payload)  # untimed: identical input for every path
                samples[name].extend(_time(fn, args.iters))

        if rank == 0:
            bytes_moved = payload.numel() * payload.element_size()
            stats = {name: _summary(values) for name, values in samples.items()}
            ar_median = stats["all_reduce"]["median_ms"]
            payload_out = {
                "world_size": world,
                "rows": args.rows,
                "hidden": args.hidden,
                "dtype": "bfloat16",
                "tensor_mb": round(bytes_moved / 1e6, 2),
                "rounds": args.rounds,
                "iters_per_round": args.iters,
                "environment": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "nccl": ".".join(str(v) for v in torch.cuda.nccl.version()),
                    "device_name": torch.cuda.get_device_name(0),
                    "device_count": torch.cuda.device_count(),
                },
                "method": (
                    "per-trial WORST-RANK elapsed via CUDA events, barrier-bounded, "
                    "MAX-reduced across ranks; buffers prepared outside the timed "
                    "region; paths interleaved within each round"
                ),
                "timings": stats,
                "rs_ag_speedup_vs_ar": round(
                    ar_median / stats["reduce_scatter_all_gather"]["median_ms"], 3
                ),
                "rs_only_speedup_vs_ar": round(
                    ar_median / stats["reduce_scatter"]["median_ms"], 3
                ),
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
