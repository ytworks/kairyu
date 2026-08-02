"""Coordinated attention-DP CUDA graph protocol coverage (#168)."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from kairyu.engine.core import worker as worker_module
from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.sampling_types import EngineSampling
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.step_executor import (
    DecodeBatch,
    FakeGraphBackend,
    GraphDecodeDecision,
    GraphStepExecutor,
)
from kairyu.engine.core.step_input import RequestSnapshot, StepDelta


def _snapshot(request_id: str, page: int, *, width: int = 1) -> RequestSnapshot:
    return RequestSnapshot(
        request_id=request_id,
        prompt_token_ids=(1,),
        computed_prompt=1,
        outputs=(),
        in_flight=0,
        page_ids=tuple(range(page, page + width)),
        decode_page_ids=(),
        eos_token_id=None,
        max_new_tokens=1,
        sampling=EngineSampling(),
    )


def _skew_step(
    counts: tuple[int, int, int, int] = (3, 1, 0, 7),
) -> tuple[
    tuple[ScheduledChunk, ...],
    tuple[tuple[str, int], ...],
    dict[str, RequestSnapshot],
]:
    chunks: list[ScheduledChunk] = []
    owners: list[tuple[str, int]] = []
    states: dict[str, RequestSnapshot] = {}
    page = 10
    for rank, count in enumerate(counts):
        for index in range(count):
            request_id = f"rank-{rank}-{index}"
            chunks.append(ScheduledChunk(request_id, 1, False, index))
            owners.append((request_id, rank))
            states[request_id] = _snapshot(
                request_id,
                page,
                width=3 if request_id == "rank-3-6" else 1,
            )
            page += 4
    return tuple(chunks), tuple(owners), states


class _DecisionRunner:
    def __init__(self, decision: GraphDecodeDecision) -> None:
        self.decision = decision
        self.seen: list[tuple[int, int]] = []

    def decode_graph_decision_for_shape(self, *, batch_size: int, max_pages: int):
        self.seen.append((batch_size, max_pages))
        return self.decision


def _payload(
    decision: GraphDecodeDecision,
    *,
    cuda_graph_decode: bool = True,
) -> tuple[worker_module._EPAttentionDPStep, dict[str, RequestSnapshot]]:
    chunks, owners, states = _skew_step()
    delta = StepDelta(
        chunks=chunks,
        new=tuple(states.values()),
        updates=(),
        dropped=(),
    )
    return (
        worker_module._EPAttentionDPStep(
            delta,
            owners,
            worker_module._EPAttentionDPGraphPlan(
                cuda_graph_decode=cuda_graph_decode,
                batch_size=7,
                max_pages=3,
                decision=decision,
            ),
        ),
        states,
    )


def test_skewed_ownership_uses_one_shared_bucket_and_unique_dummy_pages() -> None:
    decision = GraphDecodeDecision("capture", 8, 4)
    payload, states = _payload(decision)
    runner = _DecisionRunner(decision)

    worker_module._ep_attention_dp_validate_graph_plan(
        runner,
        payload,
        states,
        world_size=4,
    )
    assert runner.seen == [(7, 3)]

    for rank, real_count in enumerate((3, 1, 0, 7)):
        chunks, local_states, layouts, dummy_ids = (
            worker_module._ep_attention_dp_plan(
                payload,
                states,
                rank=rank,
                world_size=4,
                prefill_scratch_pages=(100, 101),
                decode_scratch_pages=tuple(range(102, 110)),
            )
        )
        decodes = tuple(chunk for chunk in chunks if not chunk.is_prefill)
        assert len(decodes) == 8
        assert len(layouts) == 4
        assert all(layout == (8, 8, 8, 8) for layout in layouts)
        assert len(dummy_ids) == 8 - real_count
        dummy_pages = tuple(local_states[request_id].page_ids[0] for request_id in dummy_ids)
        assert len(dummy_pages) == len(set(dummy_pages))
        assert set(dummy_pages) <= set(range(102, 110))


def test_replay_arms_no_python_layout_but_still_pads_each_rank() -> None:
    decision = GraphDecodeDecision("replay", 8, 0)
    payload, states = _payload(decision)

    for rank in range(4):
        chunks, _local_states, layouts, _dummy_ids = (
            worker_module._ep_attention_dp_plan(
                payload,
                states,
                rank=rank,
                world_size=4,
                prefill_scratch_pages=(100, 101),
                decode_scratch_pages=tuple(range(102, 110)),
            )
        )
        assert sum(not chunk.is_prefill for chunk in chunks) == 8
        assert layouts == ()


def test_mixed_prefill_keeps_one_eager_layout_before_capture_repetitions() -> None:
    decode_chunks, decode_owners, states = _skew_step()
    prefills = (
        ScheduledChunk("prefill-0", 1, True, 0),
        ScheduledChunk("prefill-1", 4, True, 0),
    )
    states = {
        "prefill-0": _snapshot("prefill-0", 200),
        "prefill-1": RequestSnapshot(
            request_id="prefill-1",
            prompt_token_ids=(1, 2, 3, 4),
            computed_prompt=4,
            outputs=(),
            in_flight=0,
            page_ids=(201,),
            decode_page_ids=(),
            eos_token_id=None,
            max_new_tokens=1,
        ),
        **states,
    }
    chunks = prefills + decode_chunks
    owners = (("prefill-0", 0), ("prefill-1", 1)) + decode_owners
    decision = GraphDecodeDecision("capture", 8, 4)
    payload = worker_module._EPAttentionDPStep(
        StepDelta(chunks, tuple(states.values()), (), ()),
        owners,
        worker_module._EPAttentionDPGraphPlan(True, 7, 3, decision),
    )

    _chunks, _states, layouts, _dummy_ids = worker_module._ep_attention_dp_plan(
        payload,
        states,
        rank=2,
        world_size=4,
        prefill_scratch_pages=(300, 301),
        decode_scratch_pages=tuple(range(302, 310)),
    )
    assert layouts[0] == (2, 5, 2, 2)
    assert layouts[1:] == ((8, 8, 8, 8),) * 4


def test_global_page_overflow_decision_keeps_every_rank_eager() -> None:
    decision = GraphDecodeDecision("eager_fallback", 7, 1)
    payload, states = _payload(decision)
    for _rank in range(4):
        runner = _DecisionRunner(decision)
        worker_module._ep_attention_dp_validate_graph_plan(
            runner,
            payload,
            states,
            world_size=4,
        )
        assert runner.seen == [(7, 3)]


@pytest.mark.parametrize(
    (
        "decision",
        "cuda_graph_decode",
        "expected_gathers",
        "expected_reductions",
    ),
    [
        (GraphDecodeDecision("capture", 8, 4), True, 1, 0),
        (GraphDecodeDecision("eager_fallback", 7, 1), False, 1, 0),
        (GraphDecodeDecision("replay", 8, 0), True, 0, 1),
    ],
    ids=["capture", "eager", "certified-replay"],
)
def test_graph_preflight_skips_only_certified_replay(
    decision: GraphDecodeDecision,
    cuda_graph_decode: bool,
    expected_gathers: int,
    expected_reductions: int,
) -> None:
    world_size = 4
    control = FakeCommunicator.create_group(world_size)
    payload, states = _payload(
        decision,
        cuda_graph_decode=cuda_graph_decode,
    )
    runners = tuple(_DecisionRunner(decision) for _ in range(world_size))

    with ThreadPoolExecutor(max_workers=world_size) as pool:
        futures = tuple(
            pool.submit(
                worker_module._ep_attention_dp_graph_preflight,
                control[rank],
                runners[rank],
                payload,
                states,
            )
            for rank in range(world_size)
        )
        assert [future.result(timeout=2) for future in futures] == [None] * world_size

    assert [communicator._gather_round for communicator in control] == [
        expected_gathers
    ] * world_size
    assert [communicator._reduce_round for communicator in control] == [
        expected_reductions
    ] * world_size
    assert [runner.seen for runner in runners] == [[(7, 3)]] * world_size


def test_capture_preflight_still_reports_one_rank_graph_state_skew_to_all() -> None:
    world_size = 4
    control = FakeCommunicator.create_group(world_size, timeout_s=1)
    capture = GraphDecodeDecision("capture", 8, 4)
    payload, states = _payload(capture)
    runners = tuple(
        _DecisionRunner(
            GraphDecodeDecision("replay", 8, 0) if rank == 2 else capture
        )
        for rank in range(world_size)
    )

    with ThreadPoolExecutor(max_workers=world_size) as pool:
        futures = tuple(
            pool.submit(
                worker_module._ep_attention_dp_graph_preflight,
                control[rank],
                runners[rank],
                payload,
                states,
            )
            for rank in range(world_size)
        )
        for future in futures:
            with pytest.raises(
                RuntimeError,
                match="rank 2: RuntimeError: attention-DP graph decision differs",
            ):
                future.result(timeout=2)

    assert [communicator._gather_round for communicator in control] == [1] * world_size


def test_replay_rank_skew_reduces_status_before_all_rank_diagnostics() -> None:
    world_size = 4
    control = FakeCommunicator.create_group(world_size, timeout_s=1)
    replay = GraphDecodeDecision("replay", 8, 0)
    payload, states = _payload(replay)
    runners = tuple(
        _DecisionRunner(
            GraphDecodeDecision("capture", 8, 4) if rank == 2 else replay
        )
        for rank in range(world_size)
    )

    with ThreadPoolExecutor(max_workers=world_size) as pool:
        futures = tuple(
            pool.submit(
                worker_module._ep_attention_dp_graph_preflight,
                control[rank],
                runners[rank],
                payload,
                states,
            )
            for rank in range(world_size)
        )
        for future in futures:
            with pytest.raises(
                RuntimeError,
                match="rank 2: RuntimeError: attention-DP graph decision differs",
            ):
                future.result(timeout=2)

    assert [communicator._reduce_round for communicator in control] == [1] * world_size
    assert [communicator._gather_round for communicator in control] == [1] * world_size


def _batch(size: int = 2) -> DecodeBatch:
    return DecodeBatch(
        token_ids=torch.arange(size),
        positions=torch.arange(size),
        page_tables=torch.zeros((size, 1), dtype=torch.int32),
        seq_lens=torch.ones(size, dtype=torch.int32),
        write_from=torch.zeros(size, dtype=torch.int64),
    )


def test_coordinated_eager_override_cannot_be_contradicted_by_local_graph() -> None:
    calls: list[int] = []
    backend = FakeGraphBackend()
    executor = GraphStepExecutor(
        lambda batch: calls.append(batch.batch_size) or batch.token_ids[:, None],
        backend,
        max_batch=8,
        max_pages=4,
        scratch_page=9,
    )
    decision = GraphDecodeDecision("eager_fallback", 7, 1)
    executor.coordinate_next_decode(decision)
    output = executor.execute_decode(_batch())
    executor.assert_coordinated_decode_consumed()

    assert output.shape == (2, 1)
    assert calls == [2]
    assert backend.captures == 0
    assert executor.execution_stats()["eager_fallbacks"] == 1


def test_coordinated_override_is_one_shot_and_exactly_consumed() -> None:
    executor = GraphStepExecutor(
        lambda batch: batch.token_ids[:, None],
        FakeGraphBackend(),
        max_batch=8,
        max_pages=4,
        scratch_page=9,
    )
    decision = GraphDecodeDecision("capture", 2, 1)
    executor.coordinate_next_decode(decision)
    with pytest.raises(RuntimeError, match="already armed"):
        executor.coordinate_next_decode(decision)
    with pytest.raises(RuntimeError, match="not consumed"):
        executor.assert_coordinated_decode_consumed()
    executor.execute_decode(_batch())
    executor.assert_coordinated_decode_consumed()
