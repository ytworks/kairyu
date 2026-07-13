"""TPModelRunner: driver/rank step protocol over FakeCommunicator (design m5 D1-D3)."""

import pytest

from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.engine_core import EngineCore
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.engine.core.step_input import RequestSnapshot
from kairyu.engine.core.tp_runner import TPModelRunner, validate_tp_degree

PAGE = 4
VOCAB = 50_000


class _DeterministicRunner:
    """_ToyRunner-style CPU runner: samples from prompt sum and chunk position."""

    def __init__(self, offset: int = 0) -> None:
        self.offset = offset
        self.states_seen: list[dict] = []
        self.released: list[str] = []

    def release(self, request_id: str) -> None:
        self.released.append(request_id)

    def execute(self, scheduled, states) -> dict[str, tuple[SampledToken, ...]]:
        self.states_seen.append(dict(states))
        sampled = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if not chunk.is_prefill or state.prefill_done:
                seed = sum(state.request.prompt_token_ids)
                token = (seed + 31 * chunk.position + self.offset) % VOCAB
                sampled[chunk.request_id] = (SampledToken(token),)
        return sampled


def _scheduler(num_pages: int = 64, budget: int = 32) -> Scheduler:
    cache = RadixKVCache(num_pages=num_pages, page_size=PAGE)
    return Scheduler(cache, max_num_batched_tokens=budget, page_size=PAGE)


def _tp_runner(world_size: int, offsets: tuple[int, ...] | None = None) -> TPModelRunner:
    offsets = offsets or (0,) * world_size
    return TPModelRunner(
        rank_runners=tuple(_DeterministicRunner(offset) for offset in offsets),
        comms=FakeCommunicator.create_group(world_size),
    )


def test_validate_tp_degree_accepts_divisors_of_kv_heads():
    for degree in (1, 2, 4, 8):
        validate_tp_degree(degree)  # 8 KV heads: all divisors valid


@pytest.mark.parametrize("degree", [0, -1])
def test_validate_tp_degree_rejects_nonpositive(degree):
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        validate_tp_degree(degree)


@pytest.mark.parametrize("degree", [3, 5, 16])
def test_validate_tp_degree_rejects_non_divisors(degree):
    with pytest.raises(ValueError, match="divide"):
        validate_tp_degree(degree)


def test_validate_tp_degree_rejects_invalid_kv_heads():
    with pytest.raises(ValueError, match="num_kv_heads"):
        validate_tp_degree(2, num_kv_heads=0)


def test_constructor_rejects_empty_ranks():
    with pytest.raises(ValueError, match="empty"):
        TPModelRunner(rank_runners=(), comms=())


def test_constructor_rejects_runner_comm_length_mismatch():
    comms = FakeCommunicator.create_group(2)
    with pytest.raises(ValueError, match="length"):
        TPModelRunner(rank_runners=(_DeterministicRunner(),), comms=comms)


def test_constructor_rejects_comms_from_wrong_sized_group():
    comms = FakeCommunicator.create_group(3)
    runners = (_DeterministicRunner(), _DeterministicRunner())
    with pytest.raises(ValueError, match="world_size"):
        TPModelRunner(rank_runners=runners, comms=comms[:2])


def test_tp2_run_matches_single_runner_output():
    prompts = {"a": (1, 2, 3, 4, 5), "b": (10, 20, 30)}

    def _run(runner) -> dict[str, tuple[int, ...]]:
        scheduler = _scheduler()
        engine = EngineCore(scheduler=scheduler, runner=runner)
        for request_id, prompt in prompts.items():
            engine.add_request(EngineRequest(request_id, prompt, max_new_tokens=3))
        return engine.run_to_completion()

    assert _run(_DeterministicRunner()) == _run(_tp_runner(2))


def test_all_ranks_execute_on_immutable_snapshots():
    runner = _tp_runner(2)
    scheduler = _scheduler()
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4), max_new_tokens=1))
    plan = scheduler.schedule()
    sampled = runner.execute(plan.scheduled, scheduler.states)
    assert set(sampled) == {"a"}
    for rank_runner in runner._rank_runners:
        assert len(rank_runner.states_seen) == 1
        for entry in rank_runner.states_seen[0].values():
            assert isinstance(entry, RequestSnapshot)


def test_rank_divergence_raises_runtime_error():
    runner = _tp_runner(2, offsets=(0, 1))  # rank 1 samples differently
    scheduler = _scheduler()
    scheduler.add_request(EngineRequest("a", (1, 2, 3, 4), max_new_tokens=1))
    plan = scheduler.schedule()
    with pytest.raises(RuntimeError, match="TP rank"):
        runner.execute(plan.scheduled, scheduler.states)


def test_empty_schedule_returns_empty_sampled_dict():
    runner = _tp_runner(2)
    assert runner.execute((), {}) == {}


def test_release_is_forwarded_to_every_rank_runner():
    runner = _tp_runner(3)
    runner.release("finished")
    assert [rank.released for rank in runner._rank_runners] == [
        ["finished"],
        ["finished"],
        ["finished"],
    ]
