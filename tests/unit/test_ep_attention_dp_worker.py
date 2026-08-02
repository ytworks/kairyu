"""Request-owned attention-DP worker protocol coverage for G4 M-A3 (#168)."""

from concurrent.futures import ThreadPoolExecutor
from types import MethodType, SimpleNamespace

import pytest

from kairyu.engine.core import worker as worker_module
from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.sampling_types import EngineSampling, SampledToken
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.step_input import RequestSnapshot


class _AttentionDPRunner:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self._model = SimpleNamespace(rank=rank, prepared=[])
        self._device = f"cuda:{rank}"
        self._sampler = object()
        self.sampling_owner = True
        self.attention_dp_scratch_pages = (64, 65)
        self.executions: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self.released: list[str] = []
        self.graph_invalidations = 0

    def execute(self, chunks, states):
        self.executions.append(
            (
                tuple(chunk.request_id for chunk in chunks),
                tuple(states),
            )
        )
        sampled: dict[str, tuple[SampledToken, ...]] = {}
        for chunk in chunks:
            state = states[chunk.request_id]
            emits = (
                not chunk.is_prefill
                or (
                    state.prefill_done
                    and state.computed_prompt == len(state.prompt_token_ids)
                )
            )
            if emits:
                sampled[chunk.request_id] = (
                    SampledToken(self.rank * 100 + chunk.position),
                )
        return sampled

    def release(self, request_id: str) -> None:
        self.released.append(request_id)

    def invalidate_graphs(self) -> None:
        self.graph_invalidations += 1


class _MalformedPacketRunner(_AttentionDPRunner):
    def execute(self, chunks, states):
        sampled = super().execute(chunks, states)
        if sampled:
            request_id = next(iter(sampled))
            sampled[request_id] = ("malformed",)
        return sampled


class _ExecutionFailureRunner(_AttentionDPRunner):
    def execute(self, chunks, states):
        super().execute(chunks, states)
        raise RuntimeError("injected owner-local execution failure")


def _snapshot(
    request_id: str,
    *,
    prompt_tokens: int,
    computed_prompt: int,
    pages: tuple[int, ...],
    cached_tokens: int = 0,
    sampling: EngineSampling | None = None,
) -> RequestSnapshot:
    return RequestSnapshot(
        request_id=request_id,
        prompt_token_ids=tuple(range(prompt_tokens)),
        computed_prompt=computed_prompt,
        outputs=(),
        in_flight=0,
        page_ids=pages,
        decode_page_ids=(),
        eos_token_id=None,
        max_new_tokens=1,
        num_cached_tokens=cached_tokens,
        sampling=EngineSampling() if sampling is None else sampling,
    )


@pytest.mark.parametrize(
    "reduced",
    (
        [0.0],
        (0,),
        (float("nan"),),
        (-1.0,),
        (2.0,),
        (0.0, 0.0),
    ),
)
def test_fast_status_reduction_fails_closed_on_malformed_result(reduced) -> None:
    communicator = SimpleNamespace(
        rank=0,
        world_size=1,
        all_reduce=lambda _values: reduced,
    )

    with pytest.raises(
        RuntimeError,
        match="fast status reduction is malformed",
    ):
        worker_module._ep_attention_dp_fast_status(
            communicator,
            worker_module._EPAttentionDPReply(rank=0, sampled={}),
        )


def test_attention_dp_fast_worker_omits_final_gather_and_preserves_affinity(
    monkeypatch,
):
    from kairyu.models import moe_parallel

    def prepare(model, layouts):
        assert not hasattr(model, "current_layouts")
        model.current_layouts = layouts
        model.prepared.append(layouts)

    def consumed(model):
        assert hasattr(model, "current_layouts")
        del model.current_layouts

    monkeypatch.setattr(moe_parallel, "prepare_attention_dp_layouts", prepare)
    monkeypatch.setattr(
        moe_parallel,
        "assert_attention_dp_layouts_consumed",
        consumed,
    )

    world_size = 4
    control = FakeCommunicator.create_group(world_size)
    model = FakeCommunicator.create_group(world_size)
    for communicator in model:
        communicator.attention_dp_direct_nccl_active = True
        communicator.synchronized_ragged_layouts = []

        def synchronize(self, layouts):
            self.synchronized_ragged_layouts.append(layouts)
            return layouts

        communicator.synchronize_attention_dp_ragged_layouts = MethodType(
            synchronize,
            communicator,
        )
    local = tuple(_AttentionDPRunner(rank) for rank in range(world_size))
    driver = worker_module.DistEPModelRunner(
        control[0],
        local[0],
        model[0],
        expert_parallel_size=world_size,
        attention_dp=True,
        pipeline_depth=5,
        page_size=16,
    )

    with ThreadPoolExecutor(max_workers=world_size - 1) as pool:
        workers = tuple(
            pool.submit(
                worker_module.worker_step_loop,
                control[rank],
                local[rank],
                model[rank],
                parallelism_prefix="EP",
            )
            for rank in range(1, world_size)
        )

        first = _snapshot(
            "seed",
            prompt_tokens=4,
            computed_prompt=4,
            pages=(7,),
        )
        assert driver.execute(
            (ScheduledChunk("seed", 4, True),),
            {"seed": first},
        ) == {"seed": (SampledToken(0),)}
        driver.release("seed")

        cache_hit = _snapshot(
            "cache-hit",
            prompt_tokens=17,
            computed_prompt=17,
            pages=(7, 8),
            cached_tokens=16,
        )
        assert driver.execute(
            (ScheduledChunk("cache-hit", 1, True, 16),),
            {"cache-hit": cache_hit},
        ) == {"cache-hit": (SampledToken(16),)}
        driver.release("cache-hit")

        requests = tuple(f"balanced-{index}" for index in range(world_size))
        chunks = tuple(
            ScheduledChunk(request_id, 1, True, index)
            for index, request_id in enumerate(requests)
        )
        states = {
            request_id: _snapshot(
                request_id,
                prompt_tokens=1,
                computed_prompt=1,
                pages=(20 + index,),
            )
            for index, request_id in enumerate(requests)
        }
        assert driver.execute(chunks, states) == {
            "balanced-0": (SampledToken(100),),
            "balanced-1": (SampledToken(201),),
            "balanced-2": (SampledToken(302),),
            "balanced-3": (SampledToken(3),),
        }

        evidence = driver.attention_dp_ownership_metadata()
        driver.shutdown()
        assert [worker.result(timeout=2) for worker in workers] == [3, 3, 3]

    assert evidence["assignment_count"] == 6
    assert evidence["active_request_count"] == 4
    assert evidence["owner_counts"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert tuple(
        (control_comm._gather_round, model_comm._gather_round)
        for control_comm, model_comm in zip(control, model, strict=True)
    ) == ((3, 3),) * world_size
    assert [communicator._reduce_round for communicator in control] == [3] * 4
    assert tuple(
        (row["request_id"], row["owner_rank"], row["cache_affine"])
        for row in evidence["assignments"]
    ) == (
        ("seed", 0, False),
        ("cache-hit", 0, True),
        ("balanced-0", 1, False),
        ("balanced-1", 2, False),
        ("balanced-2", 3, False),
        ("balanced-3", 0, False),
    )

    for rank, runner in enumerate(local):
        real_ids = tuple(
            request_id
            for chunks_seen, _states_seen in runner.executions
            for request_id in chunks_seen
            if not request_id.startswith(
                worker_module._EP_ATTENTION_DP_SCRATCH_PREFIX
            )
        )
        expected = {
            0: ("seed", "cache-hit", "balanced-3"),
            1: ("balanced-0",),
            2: ("balanced-1",),
            3: ("balanced-2",),
        }[rank]
        assert real_ids == expected
        assert len(runner._model.prepared) == 3
        assert all(
            layout == ((5, 2, 2, 2),)
            for layout in runner._model.prepared[:1]
        )
        assert runner._model.prepared[1] == ((2, 2, 2, 2),)
        assert runner._model.prepared[2] == ((2, 2, 2, 2),)
        assert any(
            request_id.startswith(
                worker_module._EP_ATTENTION_DP_SCRATCH_PREFIX
            )
            for request_id in runner.released
        )
        assert model[rank].synchronized_ragged_layouts == [
            ((5, 2, 2, 2),)
        ]


@pytest.mark.parametrize(
    "failing_runner",
    (_ExecutionFailureRunner, _MalformedPacketRunner),
)
def test_one_rank_fast_failure_rejects_on_every_rank_without_collective_skew(
    monkeypatch,
    failing_runner,
) -> None:
    from kairyu.models import moe_parallel

    monkeypatch.setattr(
        moe_parallel,
        "prepare_attention_dp_layouts",
        lambda model, layouts: setattr(model, "current_layouts", layouts),
    )
    monkeypatch.setattr(
        moe_parallel,
        "assert_attention_dp_layouts_consumed",
        lambda model: delattr(model, "current_layouts"),
    )

    world_size = 4
    control = FakeCommunicator.create_group(world_size, timeout_s=1)
    model = FakeCommunicator.create_group(world_size, timeout_s=1)
    protocol_events: list[list[str]] = [[] for _ in range(world_size)]
    for rank in range(world_size):
        original_packet_gather = model[rank].tensor_all_gather
        original_status_reduce = control[rank].all_reduce

        def packet_gather(self, tensor, *, _rank=rank, _call=original_packet_gather):
            protocol_events[_rank].append("packet")
            return _call(tensor)

        def status_reduce(self, values, *, _rank=rank, _call=original_status_reduce):
            protocol_events[_rank].append("status")
            return _call(values)

        model[rank].tensor_all_gather = MethodType(packet_gather, model[rank])
        control[rank].all_reduce = MethodType(status_reduce, control[rank])
    local = tuple(
        (
            failing_runner(rank)
            if rank == 2
            else _AttentionDPRunner(rank)
        )
        for rank in range(world_size)
    )
    driver = worker_module.DistEPModelRunner(
        control[0],
        local[0],
        model[0],
        expert_parallel_size=world_size,
        attention_dp=True,
        pipeline_depth=1,
        page_size=16,
    )
    with ThreadPoolExecutor(max_workers=world_size - 1) as pool:
        workers = tuple(
            pool.submit(
                worker_module.worker_step_loop,
                control[rank],
                local[rank],
                model[rank],
                parallelism_prefix="EP",
            )
            for rank in range(1, world_size)
        )
        requests = tuple(f"packet-{rank}" for rank in range(world_size))
        chunks = tuple(
            ScheduledChunk(request_id, 1, True, rank)
            for rank, request_id in enumerate(requests)
        )
        states = {
            request_id: _snapshot(
                request_id,
                prompt_tokens=1,
                computed_prompt=1,
                pages=(rank + 1,),
            )
            for rank, request_id in enumerate(requests)
        }
        with pytest.raises(
            RuntimeError,
            match=(
                "rank 2: local execution or token-packet encoding failed"
            ),
        ):
            driver.execute(chunks, states)
        driver.shutdown()
        for worker in workers:
            with pytest.raises(
                RuntimeError,
                match=(
                    "rank 2: local execution or token-packet encoding failed"
                ),
            ):
                worker.result(timeout=2)

    assert [communicator._gather_round for communicator in model] == [1] * 4
    # Graph preflight plus the failure-only bounded diagnostic gather.
    assert [communicator._gather_round for communicator in control] == [2] * 4
    assert [communicator._reduce_round for communicator in control] == [1] * 4
    assert protocol_events == [["packet", "status"]] * world_size


def test_non_fast_sampling_retains_final_control_gather_with_rank_symmetry(
    monkeypatch,
) -> None:
    from kairyu.models import moe_parallel

    monkeypatch.setattr(
        moe_parallel,
        "prepare_attention_dp_layouts",
        lambda model, layouts: setattr(model, "current_layouts", layouts),
    )
    monkeypatch.setattr(
        moe_parallel,
        "assert_attention_dp_layouts_consumed",
        lambda model: delattr(model, "current_layouts"),
    )

    world_size = 4
    control = FakeCommunicator.create_group(world_size, timeout_s=1)
    model = FakeCommunicator.create_group(world_size, timeout_s=1)
    local = tuple(_AttentionDPRunner(rank) for rank in range(world_size))
    driver = worker_module.DistEPModelRunner(
        control[0],
        local[0],
        model[0],
        expert_parallel_size=world_size,
        attention_dp=True,
        pipeline_depth=1,
        page_size=16,
    )
    state = _snapshot(
        "logprobs",
        prompt_tokens=1,
        computed_prompt=1,
        pages=(1,),
        sampling=EngineSampling(logprobs=0),
    )
    with ThreadPoolExecutor(max_workers=world_size - 1) as pool:
        workers = tuple(
            pool.submit(
                worker_module.worker_step_loop,
                control[rank],
                local[rank],
                model[rank],
                parallelism_prefix="EP",
            )
            for rank in range(1, world_size)
        )
        assert driver.execute(
            (ScheduledChunk("logprobs", 1, True),),
            {"logprobs": state},
        ) == {"logprobs": (SampledToken(0),)}
        driver.shutdown()
        assert [worker.result(timeout=2) for worker in workers] == [1, 1, 1]

    assert tuple(
        (control_comm._gather_round, model_comm._gather_round)
        for control_comm, model_comm in zip(control, model, strict=True)
    ) == ((2, 0),) * world_size


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"expert_parallel_size": 2}, "expert_parallel_size=4"),
        ({"pipeline_depth": 0}, "positive integer"),
        ({"kv_cache_dtype": "fp8"}, "kv_cache_dtype='bfloat16'"),
        ({"pd_separation": True}, "P-D separation"),
        ({"graph_scratch_page": 7}, "graph scratch page"),
        ({"dram_kv_tier_capacity_pages": 1}, "DRAM KV tier"),
        ({"dram_kv_tier_profile": "/tmp/profile.json"}, "DRAM KV tier"),
        ({"speculative": "ngram"}, "speculative decoding"),
    ],
)
def test_attention_dp_rejects_outside_the_pinned_production_envelope(
    override,
    message,
):
    options = {
        "expert_parallel_size": 4,
        "pipeline_depth": 5,
    }
    options.update(override)
    with pytest.raises((TypeError, ValueError), match=message):
        worker_module._validate_ep_attention_dp_mode(**options)


def test_attention_dp_accepts_fully_reserved_cuda_graph_envelope() -> None:
    worker_module._validate_ep_attention_dp_mode(
        expert_parallel_size=4,
        pipeline_depth=5,
        decode_mode="cuda_graph",
        graph_scratch_page=74,
        graph_max_batch=8,
        graph_max_pages=32,
        graph_warmup_iters=2,
    )


@pytest.mark.parametrize(
    ("stats", "message"),
    [
        ({"enabled": False, "capability_gap": None}, "remain enabled"),
        (
            {
                "enabled": True,
                "capability_gap": "backend has no native batched prefill",
            },
            "requires native batched prefill",
        ),
    ],
)
def test_attention_dp_rejects_unsafe_prefill_runner(stats, message) -> None:
    runner = SimpleNamespace(prefill_execution_stats=lambda *, reset: stats)
    with pytest.raises(RuntimeError, match=message):
        worker_module._validate_attention_dp_batched_prefill_runner(runner)


def test_attention_dp_rejects_prefill_disable_before_control_broadcast() -> None:
    control = FakeCommunicator.create_group(4)
    model = FakeCommunicator.create_group(4)
    driver = worker_module.DistEPModelRunner(
        control[0],
        _AttentionDPRunner(0),
        model[0],
        expert_parallel_size=4,
        attention_dp=True,
        pipeline_depth=5,
        page_size=16,
    )

    with pytest.raises(ValueError, match="cannot disable batched prefill"):
        driver.set_batched_prefill_enabled(False)
    assert control[0]._group.broadcasts == {}


def test_attention_dp_rejects_live_graph_invalidation_until_shutdown() -> None:
    world_size = 4
    control = FakeCommunicator.create_group(world_size)
    model = FakeCommunicator.create_group(world_size)
    local = tuple(_AttentionDPRunner(rank) for rank in range(world_size))
    driver = worker_module.DistEPModelRunner(
        control[0],
        local[0],
        model[0],
        expert_parallel_size=world_size,
        attention_dp=True,
        pipeline_depth=5,
        page_size=16,
    )

    with ThreadPoolExecutor(max_workers=world_size - 1) as pool:
        workers = tuple(
            pool.submit(
                worker_module.worker_step_loop,
                control[rank],
                local[rank],
                model[rank],
                parallelism_prefix="EP",
            )
            for rank in range(1, world_size)
        )
        with pytest.raises(
            RuntimeError,
            match="cannot invalidate CUDA graphs while serving",
        ):
            driver.invalidate_graphs()
        assert local[0].graph_invalidations == 0
        assert not control[0]._group.broadcasts.get(0)

        driver.shutdown()
        assert [worker.result(timeout=2) for worker in workers] == [0, 0, 0]

    driver.invalidate_graphs()
    assert local[0].graph_invalidations == 1


def test_attention_dp_launcher_rejects_wrong_graph_scratch_before_spawn(
    monkeypatch,
) -> None:
    import torch.multiprocessing as mp

    touched: list[str] = []
    monkeypatch.setattr(
        worker_module,
        "_ep_placement",
        lambda *_args, **_kwargs: touched.append("placement"),
    )
    monkeypatch.setattr(
        mp,
        "spawn",
        lambda *_args, **_kwargs: touched.append("spawn"),
    )

    with pytest.raises(ValueError, match="graph_scratch_page"):
        worker_module.DistEPLauncher(
            "/model",
            expert_parallel_size=4,
            num_pages=64,
            page_size=16,
            vocab=["token"],
            attention_dp=True,
            decode_mode="cuda_graph",
            graph_scratch_page=73,
            graph_max_batch=8,
            graph_max_pages=32,
        )
    assert touched == []
