from kairyu.engine.core.engine_core import EngineCore
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler

PAGE = 4


class EchoRunner:
    """CPU stub ModelRunner: 'samples' prompt_len + step for determinism."""

    def __init__(self) -> None:
        self.steps_executed = 0

    def execute(self, scheduled, states) -> dict[str, tuple[SampledToken, ...]]:
        self.steps_executed += 1
        sampled = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if state.prefill_done:
                sampled[chunk.request_id] = (SampledToken(1000 + len(state.outputs)),)
        return sampled


def _engine(num_pages=64, budget=8, max_seqs=4) -> tuple[EngineCore, EchoRunner]:
    cache = RadixKVCache(num_pages=num_pages, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=budget, max_num_seqs=max_seqs)
    runner = EchoRunner()
    return EngineCore(scheduler=scheduler, runner=runner), runner


def test_run_to_completion_produces_requested_tokens():
    engine, _ = _engine()
    engine.add_request(EngineRequest("a", tuple(range(1, 11)), max_new_tokens=3))
    outputs = engine.run_to_completion()
    assert outputs["a"] == (1000, 1001, 1002)


def test_concurrent_requests_all_finish():
    engine, _ = _engine(budget=8, max_seqs=2)
    for i in range(4):
        prompt = tuple(range(i * 100 + 1, i * 100 + 7))
        engine.add_request(EngineRequest(f"r{i}", prompt, max_new_tokens=2))
    outputs = engine.run_to_completion()
    assert set(outputs) == {"r0", "r1", "r2", "r3"}
    assert all(len(tokens) == 2 for tokens in outputs.values())


def test_step_returns_finished_ids():
    engine, _ = _engine(budget=64)
    engine.add_request(EngineRequest("a", (1, 2, 3, 4), max_new_tokens=1))
    finished = engine.step()
    assert finished == ("a",)
    assert engine.has_unfinished() is False


def test_oversized_prompt_is_rejected_gracefully_not_stalling():
    # C2: a request whose prompt can never fit must be rejected (finished with
    # reason "length" and empty output), not crash the engine with a stall that
    # would take down every concurrent request.
    cache = RadixKVCache(num_pages=1, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=64)
    engine = EngineCore(scheduler=scheduler, runner=EchoRunner())
    engine.add_request(EngineRequest("big", tuple(range(1, 100)), max_new_tokens=1))
    outputs = engine.run_to_completion()  # no RuntimeError
    assert outputs["big"] == ()  # rejected, nothing generated
    assert scheduler.finish_reason("big") == "length"
    assert engine.has_unfinished() is False


def test_oversized_prompt_does_not_block_concurrent_requests():
    # C2: an unadmittable request must not wedge a normal request behind it.
    cache = RadixKVCache(num_pages=2, page_size=PAGE)  # capacity = 8 tokens
    scheduler = Scheduler(cache, max_num_batched_tokens=64)
    engine = EngineCore(scheduler=scheduler, runner=EchoRunner())
    engine.add_request(EngineRequest("big", tuple(range(1, 100)), max_new_tokens=1))
    engine.add_request(EngineRequest("ok", (1, 2, 3, 4), max_new_tokens=1))
    outputs = engine.run_to_completion()
    assert outputs["big"] == ()  # rejected
    assert len(outputs["ok"]) == 1  # normal request still completed
