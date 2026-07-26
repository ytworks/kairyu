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
        if warmup_iters < 0:
            raise ValueError(f"warmup_iters must be >= 0, got {warmup_iters}")
        self._warmup_iters = warmup_iters
        self._pool = torch.cuda.graph_pool_handle()
        self.captures = 0
        self.replays = 0

    def invalidate(self) -> None:
        # An output from the old graph may still be owned by a caller when the
        # model swaps weights.  Reusing that graph's pool token while such a
        # tensor is alive trips PyTorch's allocator use-count assertion.  A new
        # token lets replacement captures proceed while the retired allocation
        # drains independently.
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
        self.captures += 1

        class _Replayable:
            def __init__(
                self,
                captured_graph: torch.cuda.CUDAGraph,
                captured_out: torch.Tensor,
            ) -> None:
                # Receive the captured objects as arguments rather than closing
                # over them.  Otherwise the nested class's __init__ closure
                # keeps the CUDAGraph alive even after close() clears the
                # instance fields, and a graph-captured NCCL process group can
                # never finish destruction.
                self._graph: torch.cuda.CUDAGraph | None = captured_graph
                self._static_out: torch.Tensor | None = captured_out

            def replay(self) -> torch.Tensor:
                backend.replays += 1
                assert self._graph is not None and self._static_out is not None
                self._graph.replay()
                return self._static_out

            def close(self) -> None:
                # A captured NCCL graph owns communicator registrations. Drop
                # the CUDAGraph reference deterministically before the process
                # group is destroyed; relying on GC can deadlock TP teardown.
                if self._graph is not None:
                    # Releases the graph executable's allocator/pool use before
                    # the same backend captures a replacement after invalidate.
                    # Clearing only the Python reference leaves that use alive
                    # until finalization and PyTorch rejects the recapture.
                    self._graph.reset()
                self._graph = None
                self._static_out = None

        backend = self
        return _Replayable(graph, static_out)
