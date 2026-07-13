"""Distributed TP driver/worker control protocol."""

from concurrent.futures import ThreadPoolExecutor

from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.step_input import StepDelta
from kairyu.engine.core.worker import DistTPModelRunner, worker_step_loop


class _ReleaseRunner:
    def __init__(self) -> None:
        self.released: list[str] = []

    def execute(self, scheduled, states):
        return {}

    def release(self, request_id: str) -> None:
        self.released.append(request_id)


def test_dist_release_reaches_driver_and_idle_worker():
    comms = FakeCommunicator.create_group(2)
    local = (_ReleaseRunner(), _ReleaseRunner())
    driver = DistTPModelRunner(comms[0], local[0])
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(worker_step_loop, comms[1], local[1])
        try:
            driver.release("finished")
        finally:
            driver.shutdown()
        assert worker.result(timeout=2) == 0
    assert local[0].released == ["finished"]
    assert local[1].released == ["finished"]


def test_step_delta_dropped_does_not_release_request():
    comms = FakeCommunicator.create_group(2)
    local = (_ReleaseRunner(), _ReleaseRunner())
    driver = DistTPModelRunner(comms[0], local[0])
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(worker_step_loop, comms[1], local[1])
        comms[0].broadcast(
            StepDelta(chunks=(), new=(), updates=(), dropped=("preempted",)), src=0
        )
        driver.shutdown()
        assert worker.result(timeout=2) == 1
    assert local[0].released == []
    assert local[1].released == []
