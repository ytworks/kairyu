import time

from kairyu.engine.core.engine_core import EngineCore
from kairyu.engine.core.overlap import OverlapEngineCore
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler

PAGE = 4


class PositionRunner:
    """Deterministic stub: the sampled token depends only on the scheduled
    position, never on previously committed outputs — exactly the property a
    real overlap runner has (it patches the last-token slot device-side)."""

    def __init__(self, latency_s: float = 0.0, events: list | None = None) -> None:
        self._latency_s = latency_s
        self.events = events if events is not None else []
        self._step = 0

    def execute(self, scheduled, states) -> dict[str, int]:
        step = self._step
        self._step += 1
        if self._latency_s:
            time.sleep(self._latency_s)
        sampled = {
            chunk.request_id: (SampledToken(1000 + chunk.position),)
            for chunk in scheduled
            if not chunk.is_prefill or states[chunk.request_id].prefill_done
        }
        self.events.append(f"executed:{step}")
        return sampled


def _requests(n: int = 3) -> list[EngineRequest]:
    return [
        EngineRequest(
            f"r{i}",
            prompt_token_ids=tuple(range(i * 50 + 1, i * 50 + 7)),
            max_new_tokens=4,
        )
        for i in range(n)
    ]


def _build(engine_cls, runner, budget=16):
    cache = RadixKVCache(num_pages=256, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=budget, max_num_seqs=8)
    return engine_cls(scheduler=scheduler, runner=runner)


def test_overlap_outputs_equal_serial_engine_outputs():
    serial = _build(EngineCore, PositionRunner())
    overlap = _build(OverlapEngineCore, PositionRunner())
    for request in _requests():
        serial.add_request(request)
        overlap.add_request(request)
    assert overlap.run_to_completion() == serial.run_to_completion()


def test_runner_receives_frozen_snapshot_not_live_state():
    # E3: the overlap loop must hand the runner a torn-free snapshot, not the
    # live scheduler state a concurrent update() mutates. RequestSnapshot is
    # frozen, so seeing it proves the freeze happened before dispatch.
    from kairyu.engine.core.step_input import RequestSnapshot

    seen_types: list[type] = []

    class TypeCapturingRunner(PositionRunner):
        def execute(self, scheduled, states):
            for chunk in scheduled:
                seen_types.append(type(states[chunk.request_id]))
            return super().execute(scheduled, states)

    engine = _build(OverlapEngineCore, TypeCapturingRunner())
    for request in _requests(2):
        engine.add_request(request)
    engine.run_to_completion()
    assert seen_types  # ran
    assert all(t is RequestSnapshot for t in seen_types)


def test_next_step_is_scheduled_while_previous_executes():
    events: list[str] = []
    runner = PositionRunner(latency_s=0.02, events=events)
    engine = _build(OverlapEngineCore, runner)
    engine.events = events
    for request in _requests(2):
        engine.add_request(request)
    engine.run_to_completion()
    # the pipeline must plan step 1 before step 0 finishes on the device
    assert events.index("scheduled:1") < events.index("executed:0")


def test_no_overscheduling_beyond_max_new_tokens():
    engine = _build(OverlapEngineCore, PositionRunner())
    engine.add_request(EngineRequest("a", (1, 2, 3, 4, 5, 6), max_new_tokens=3))
    outputs = engine.run_to_completion()
    assert outputs["a"] == (1000, 1001, 1002)  # positions 0,1,2 — exactly max_new_tokens


def test_overlap_rejects_oversized_prompt_gracefully():
    # C2: an unadmittable prompt is rejected (empty output), not a fatal stall.
    cache = RadixKVCache(num_pages=1, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=64)
    engine = OverlapEngineCore(scheduler=scheduler, runner=PositionRunner())
    engine.add_request(EngineRequest("big", tuple(range(1, 100)), max_new_tokens=1))
    outputs = engine.run_to_completion()  # no RuntimeError
    assert outputs["big"] == ()
    assert scheduler.finish_reason("big") == "length"
