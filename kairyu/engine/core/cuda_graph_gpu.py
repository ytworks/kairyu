"""Real CUDA-graph backend (m17 D1) — deploy-day verified (`pytest -m gpu`).

Contract identical to FakeGraphBackend: capture binds static buffers, replay
re-executes the captured kernels on them. Warmup runs on a side stream (the
torch.cuda.CUDAGraph requirement); one shared memory pool across buckets.
"""

from __future__ import annotations

import torch


class CudaGraphBackend:
    def __init__(self, warmup_iters: int = 3) -> None:
        if not torch.cuda.is_available():  # pragma: no cover - deploy-day only
            raise RuntimeError("CudaGraphBackend requires CUDA")
        self._warmup_iters = warmup_iters
        self._pool = torch.cuda.graph_pool_handle()

    def capture(self, fn, static_batch):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                fn(static_batch)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            static_out = fn(static_batch)

        class _Replayable:
            def replay(self) -> torch.Tensor:
                graph.replay()
                return static_out

        return _Replayable()
