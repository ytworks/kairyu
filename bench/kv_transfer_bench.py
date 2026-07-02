"""KV transfer plane microbench (Goal G2 gate B2; design m6 D3/D6).

CPU-runnable over the TCP-loopback transport; the GPU/2-node phase points this
same harness at the NCCL/RDMA transport on a real fabric. Per design D6 it
measures the raw link first (one contiguous buffer of the same total bytes),
then paged transfer with the REAL sharded fragment layout (a logical page is
layers x TP-shard fragments), and reports effective GB/s and amortized
us/token next to the full config (G2 section 8: no number without its config).

Run: uv run python bench/kv_transfer_bench.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from kairyu.engine.core.kv_transport import (
    PageFrame,
    SequenceMeta,
    TcpLoopbackTransport,
)

_TOKEN_BYTES_70B_FP8 = 160 * 1024  # 80 layers x 8 KV heads x 128 dim x K+V x FP8


def _build_frames(args: argparse.Namespace) -> tuple[PageFrame, ...]:
    fragment_bytes = args.page_tokens * args.token_bytes // (args.layers * args.tp)
    fragment = bytes(fragment_bytes)
    return tuple(
        PageFrame(page_id=page, fragments=(fragment,) * args.layers)
        for page in range(args.batch_pages)
    )


def _build_raw_frame(args: argparse.Namespace) -> tuple[PageFrame, ...]:
    total = args.batch_pages * args.page_tokens * args.token_bytes // args.tp
    return (PageFrame(page_id=0, fragments=(bytes(total),)),)


async def _measure(
    frames: tuple[PageFrame, ...], batches: int, warmup: int
) -> tuple[float, int]:
    """Send+receive `batches` batches; return (seconds, payload bytes moved)."""
    receiver = TcpLoopbackTransport("decode")
    receiver.register(num_pages=1_000_000)
    address = await receiver.start_server()
    sender = TcpLoopbackTransport("prefill")
    sender.register(num_pages=1_000_000)
    meta = SequenceMeta(token_ids=(0,))
    payload = sum(len(f) for frame in frames for f in frame.fragments)

    async def pump() -> None:
        for _ in range(warmup + batches):
            await sender.send(address, frames, meta)

    async def drain() -> float:
        for _ in range(warmup):
            await receiver.recv(address)
        start = time.perf_counter()
        for _ in range(batches):
            await receiver.recv(address)
        return time.perf_counter() - start

    try:
        _, elapsed = await asyncio.gather(pump(), drain())
    finally:
        await sender.close()
        await receiver.close()
    return elapsed, payload * batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-pages", type=int, default=64)
    parser.add_argument("--page-tokens", type=int, default=16)
    parser.add_argument("--token-bytes", type=int, default=_TOKEN_BYTES_70B_FP8)
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--tp", type=int, default=8, help="TP degree (per-rank shard)")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()

    raw_s, raw_bytes = asyncio.run(_measure(_build_raw_frame(args), args.batches, args.warmup))
    paged_s, paged_bytes = asyncio.run(_measure(_build_frames(args), args.batches, args.warmup))

    raw_gbps = raw_bytes / raw_s / 1e9
    paged_gbps = paged_bytes / paged_s / 1e9
    tokens_moved = args.batch_pages * args.page_tokens * args.batches
    us_per_token = paged_s / tokens_moved * 1e6

    fragment_bytes = args.page_tokens * args.token_bytes // (args.layers * args.tp)
    result = {
        "config": {
            "transport": "tcp-loopback (CPU harness; GPU phase swaps NCCL/RDMA)",
            "batch_pages": args.batch_pages,
            "page_tokens": args.page_tokens,
            "token_bytes": args.token_bytes,
            "layers": args.layers,
            "tp": args.tp,
            "fragment_bytes": fragment_bytes,
            "fragments_per_batch": args.batch_pages * args.layers,
            "batches": args.batches,
        },
        "raw_link_gbps": round(raw_gbps, 3),
        "paged_gbps": round(paged_gbps, 3),
        "paged_vs_raw": round(paged_gbps / raw_gbps, 3) if raw_gbps else None,
        "us_per_token_amortized": round(us_per_token, 3),
    }
    print(json.dumps(result, indent=2))
    print(
        f"raw {raw_gbps:.2f} GB/s | paged {paged_gbps:.2f} GB/s "
        f"({result['paged_vs_raw']:.0%} of raw) | {us_per_token:.2f} us/token"
    )


if __name__ == "__main__":
    main()
