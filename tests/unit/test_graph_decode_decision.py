"""Read-only CUDA-graph dispatch previews for coordinated model forwards."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from kairyu.engine.core.model_runner import PagedModelRunner
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.step_executor import (
    FakeGraphBackend,
    GraphDecodeDecision,
    GraphStepExecutor,
    build_decode_batch,
)


def _decode(batch):
    return batch.token_ids[:, None]


def _batch(size: int, *, max_pages: int = 1):
    return build_decode_batch(
        token_ids=list(range(size)),
        positions=list(range(size)),
        page_lists=[tuple(range(max_pages)) for _ in range(size)],
        seq_lens=[max_pages] * size,
        max_pages=max_pages,
        scratch_page=0,
    )


class _WarmCaptureBackend(FakeGraphBackend):
    @property
    def capture_model_forward_count(self) -> int:
        return 4  # three warmups plus the capture forward


def _executor(*, max_batch: int = 8, max_pages: int = 4):
    return GraphStepExecutor(
        _decode,
        _WarmCaptureBackend(),
        max_batch=max_batch,
        max_pages=max_pages,
        scratch_page=0,
    )


def test_prospective_decision_is_typed_read_only_and_tracks_lifecycle():
    executor = _executor()

    capture = executor.decode_decision(batch_size=3, max_pages=2)

    assert capture == GraphDecodeDecision(
        kind="capture",
        bucket_size=4,
        capture_model_forward_count=4,
    )
    assert executor.execution_stats() == {
        "graph_executions": 0,
        "eager_fallbacks": 0,
        "captured_buckets": (),
    }
    with pytest.raises(FrozenInstanceError):
        capture.bucket_size = 8  # type: ignore[misc]

    executor.execute_decode(_batch(3, max_pages=2))

    assert executor.decode_decision(batch_size=3, max_pages=2) == (
        GraphDecodeDecision(
            kind="replay",
            bucket_size=4,
            capture_model_forward_count=0,
        )
    )


@pytest.mark.parametrize(
    ("batch_size", "max_pages"),
    [(9, 1), (3, 5)],
    ids=["oversize-batch", "oversize-page-table"],
)
def test_eager_fallback_reports_the_unpadded_execution_shape(
    batch_size: int,
    max_pages: int,
):
    decision = _executor().decode_decision(
        batch_size=batch_size,
        max_pages=max_pages,
    )

    assert decision == GraphDecodeDecision(
        kind="eager_fallback",
        bucket_size=batch_size,
        capture_model_forward_count=1,
    )


@pytest.mark.parametrize(
    ("batch_size", "max_pages"),
    [(0, 1), (True, 1), (1, 0), (1, True)],
)
def test_prospective_shape_must_be_explicit_and_positive(
    batch_size: int,
    max_pages: int,
):
    with pytest.raises(ValueError):
        _executor().decode_decision(
            batch_size=batch_size,
            max_pages=max_pages,
        )


def _runner(executor):
    runner = object.__new__(PagedModelRunner)
    runner._graph = executor
    return runner


def _state(*pages: int, decode_pages: tuple[int, ...] = ()):
    return SimpleNamespace(
        allocation=SimpleNamespace(pages=list(pages)),
        decode_pages=list(decode_pages),
        outputs=[71, 72],
    )


def test_runner_preview_uses_only_decode_count_and_complete_page_width():
    runner = _runner(_executor(max_batch=4, max_pages=2))
    scheduled = (
        ScheduledChunk("prefill", 8, True),
        ScheduledChunk("decode-a", 1, False, 2),
        ScheduledChunk("decode-b", 1, False, 2),
    )
    states = {
        "prefill": _state(0),
        "decode-a": _state(1, decode_pages=(2, 3)),
        "decode-b": _state(4),
    }
    before = {
        request_id: (
            tuple(state.allocation.pages),
            tuple(state.decode_pages),
            tuple(state.outputs),
        )
        for request_id, state in states.items()
    }

    decision = runner.decode_graph_decision(scheduled, states)

    # decode-a owns three total pages, so this is eager fallback even though
    # its base allocation alone would fit the two-page graph buffer.
    assert decision == GraphDecodeDecision(
        kind="eager_fallback",
        bucket_size=2,
        capture_model_forward_count=1,
    )
    assert runner.required_decode_model_forward_count(scheduled, states) == 1
    assert before == {
        request_id: (
            tuple(state.allocation.pages),
            tuple(state.decode_pages),
            tuple(state.outputs),
        )
        for request_id, state in states.items()
    }
    assert runner._graph.execution_stats()["captured_buckets"] == ()


def test_runner_preview_reports_capture_then_zero_for_replay():
    executor = _executor(max_batch=4, max_pages=2)
    runner = _runner(executor)
    scheduled = (
        ScheduledChunk("decode-a", 1, False, 2),
        ScheduledChunk("decode-b", 1, False, 2),
    )
    states = {"decode-a": _state(1), "decode-b": _state(2)}

    assert runner.decode_graph_decision(scheduled, states) == GraphDecodeDecision(
        kind="capture",
        bucket_size=2,
        capture_model_forward_count=4,
    )
    assert runner.required_decode_model_forward_count(scheduled, states) == 4
    executor.execute_decode(_batch(2))
    assert runner.decode_graph_decision(scheduled, states) == GraphDecodeDecision(
        kind="replay",
        bucket_size=2,
        capture_model_forward_count=0,
    )
    assert runner.required_decode_model_forward_count(scheduled, states) == 0


def test_runner_preview_returns_zero_without_a_single_token_decode():
    runner = _runner(_executor())
    scheduled = (
        ScheduledChunk("prefill", 8, True),
        ScheduledChunk("verify", 3, False, 2),
    )

    assert runner.decode_graph_decision(
        scheduled,
        {"prefill": _state(0), "verify": _state(1)},
    ) is None
    assert runner.required_decode_model_forward_count(
        scheduled,
        {"prefill": _state(0), "verify": _state(1)},
    ) == 0


def test_runner_without_graph_needs_one_eager_forward():
    runner = _runner(None)
    scheduled = (ScheduledChunk("decode", 1, False, 2),)
    states = {"decode": _state(1)}

    assert runner.decode_graph_decision(scheduled, states) == GraphDecodeDecision(
        kind="eager_fallback",
        bucket_size=1,
        capture_model_forward_count=1,
    )
    assert runner.required_decode_model_forward_count(scheduled, states) == 1


def test_cuda_backend_reports_warmups_plus_capture(monkeypatch):
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.graph_pool_handle", object)

    backend = CudaGraphBackend(warmup_iters=3)

    assert backend.capture_model_forward_count == 4
