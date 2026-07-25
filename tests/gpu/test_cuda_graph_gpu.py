"""CudaGraphBackend + GraphStepExecutor on real hardware (m17 D1/D2).

`CudaGraphBackend` had no GPU coverage at all: the capture/replay policy is
CPU-tested against `FakeGraphBackend`, and the fake cannot catch the one thing
that separates a real graph from a function call — a replay re-runs *recorded
kernels over recorded memory*. Anything the executor rebinds instead of writing
in place is invisible to it, and any buffer on the wrong device is never written
by the GPU at all.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

MAX_PAGES = 4


def _require_cuda() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("CUDA graphs need CUDA")


def _decode_fn(weights: torch.Tensor):
    """A decode-shaped forward that reads every static input.

    Touching all four fields matters: a kernel that ignores, say, seq_lens would
    replay correctly even if that buffer were stale, and the test would pass for
    the wrong reason.
    """

    def run(batch):
        embedded = weights[batch.token_ids]  # [B, H]
        positional = batch.positions.to(embedded.dtype).unsqueeze(1)
        pages = batch.page_tables.to(embedded.dtype).sum(dim=1, keepdim=True)
        lengths = batch.seq_lens.to(embedded.dtype).unsqueeze(1)
        return embedded + positional + pages + lengths

    return run


def _batch(size: int, device: str, *, max_pages: int = MAX_PAGES, offset: int = 0):
    """`offset` shifts every field, so two batches of different sizes are not
    prefixes of one another — otherwise a "the output changed" assertion can pass
    without the replay having read anything new."""
    from kairyu.engine.core.step_executor import build_decode_batch

    return build_decode_batch(
        token_ids=[((i + offset) * 7) % 32 for i in range(size)],
        positions=[i + offset + 3 for i in range(size)],
        page_lists=[[(i + offset + j) % 5 for j in range(2)] for i in range(size)],
        seq_lens=[i + offset + 1 for i in range(size)],
        max_pages=max_pages,
        device=device,
    )


def test_capture_and_replay_match_eager():
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend
    from kairyu.engine.core.step_executor import GraphStepExecutor

    _require_cuda()
    torch.manual_seed(11)
    weights = torch.randn(32, 8, device="cuda:0")
    decode = _decode_fn(weights)
    executor = GraphStepExecutor(
        decode, CudaGraphBackend(), max_batch=8, max_pages=MAX_PAGES, device="cuda:0"
    )

    batch = _batch(4, "cuda:0")
    expected = decode(batch)
    actual = executor.execute_decode(batch)
    assert actual.device.type == "cuda"
    assert torch.allclose(actual, expected)


def test_replay_reflects_new_inputs_not_the_captured_ones():
    """The failure a fake backend cannot produce: a graph replays over the
    buffers it captured, so an input written anywhere else is simply ignored."""
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend
    from kairyu.engine.core.step_executor import GraphStepExecutor

    _require_cuda()
    torch.manual_seed(13)
    weights = torch.randn(32, 8, device="cuda:0")
    decode = _decode_fn(weights)
    executor = GraphStepExecutor(
        decode, CudaGraphBackend(), max_batch=8, max_pages=MAX_PAGES, device="cuda:0"
    )

    first = _batch(4, "cuda:0")
    executor.execute_decode(first)  # captures the bucket

    second = _batch(3, "cuda:0", offset=9)
    replayed = executor.execute_decode(second)
    assert torch.allclose(replayed, decode(second))
    # and the two must actually differ, or the assertion above proves nothing
    assert not torch.allclose(replayed, decode(first)[:3])


def test_padding_rows_do_not_leak_into_the_result():
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend
    from kairyu.engine.core.step_executor import GraphStepExecutor

    _require_cuda()
    torch.manual_seed(17)
    weights = torch.randn(32, 8, device="cuda:0")
    decode = _decode_fn(weights)
    executor = GraphStepExecutor(
        decode, CudaGraphBackend(), max_batch=8, max_pages=MAX_PAGES, device="cuda:0"
    )

    for size in (1, 2, 3, 5, 8):
        batch = _batch(size, "cuda:0")
        out = executor.execute_decode(batch)
        assert out.shape[0] == size, f"bucket padding survived for size {size}"
        assert torch.allclose(out, decode(batch)), size


def test_oversize_batch_falls_back_to_eager():
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend
    from kairyu.engine.core.step_executor import GraphStepExecutor

    _require_cuda()
    torch.manual_seed(19)
    weights = torch.randn(32, 8, device="cuda:0")
    decode = _decode_fn(weights)
    executor = GraphStepExecutor(
        decode, CudaGraphBackend(), max_batch=4, max_pages=MAX_PAGES, device="cuda:0"
    )

    # beyond the largest bucket: must run, not crash (D2)
    batch = _batch(9, "cuda:0")
    assert torch.allclose(executor.execute_decode(batch), decode(batch))

    # a page table wider than the captured static buffer is the other fallback
    wide = _batch(2, "cuda:0", max_pages=MAX_PAGES + 3)
    assert torch.allclose(executor.execute_decode(wide), decode(wide))


def test_invalidate_forces_recapture():
    """A weight SWAP, not an in-place update.

    `weights.mul_(2.0)` rebinds nothing — the captured graph reads the same
    storage, so it sees the new values and the assertions pass even when
    `invalidate()` is a no-op (verified: changed_without_invalidate=True).
    Only rebinding to a NEW tensor is invisible to a stale graph.
    """
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend
    from kairyu.engine.core.step_executor import GraphStepExecutor

    _require_cuda()
    torch.manual_seed(23)
    holder = {"weights": torch.randn(32, 8, device="cuda:0")}

    def decode(batch):
        return _decode_fn(holder["weights"])(batch)

    backend = CudaGraphBackend()
    executor = GraphStepExecutor(
        decode, backend, max_batch=8, max_pages=MAX_PAGES, device="cuda:0"
    )

    batch = _batch(4, "cuda:0")
    first = executor.execute_decode(batch).clone()
    captures_after_first = len(executor._captured)

    # new storage: the captured kernels still point at the old one. The old
    # tensor is kept alive on purpose — freeing it lets the caching allocator
    # hand the same block to the new randn, and the graph then reads the NEW
    # values through the old pointer (observed: only row 0 differed).
    _pinned = holder["weights"]
    holder["weights"] = torch.randn(32, 8, device="cuda:0")
    eager_after_swap = decode(batch).clone()

    stale = executor.execute_decode(batch)
    assert torch.allclose(stale, first), "the graph was not actually captured"
    assert not torch.allclose(stale, eager_after_swap)

    executor.invalidate()
    assert not executor._captured, "invalidate() left a capture behind"

    fresh = executor.execute_decode(batch)
    assert torch.allclose(fresh, eager_after_swap), "no recapture after invalidate"
    assert len(executor._captured) == captures_after_first
    assert _pinned is not holder["weights"]


def test_static_buffers_live_on_the_capture_device():
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend
    from kairyu.engine.core.step_executor import GraphStepExecutor

    _require_cuda()
    weights = torch.randn(32, 8, device="cuda:0")
    executor = GraphStepExecutor(
        _decode_fn(weights),
        CudaGraphBackend(),
        max_batch=8,
        max_pages=MAX_PAGES,
        device="cuda:0",
    )
    executor.execute_decode(_batch(4, "cuda:0"))

    _replayable, static = next(iter(executor._captured.values()))
    for name in ("token_ids", "positions", "page_tables", "seq_lens"):
        tensor = getattr(static, name)
        assert tensor.device.type == "cuda", f"{name} captured on {tensor.device}"
