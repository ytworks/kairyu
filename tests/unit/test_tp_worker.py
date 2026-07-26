"""Distributed TP driver/worker control protocol."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from kairyu.engine.core import worker as worker_module
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


def test_serving_groups_keep_control_off_the_model_backend(monkeypatch):
    created: list[tuple[str, float]] = []

    def fake_group(backend: str, *, timeout_s: float):
        created.append((backend, timeout_s))
        return f"group-{len(created)}-{backend}"

    monkeypatch.setattr(worker_module, "serving_group", fake_group)
    groups = worker_module.serving_groups("nccl")

    assert created == [
        ("gloo", worker_module._CONTROL_IDLE_TIMEOUT_S),
        ("nccl", worker_module._SERVE_OP_TIMEOUT_S),
    ]
    assert groups.control == "group-1-gloo"
    assert groups.model == "group-2-nccl"


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


class _BrokenComm:
    def __init__(self) -> None:
        self.broadcasts = 0

    def broadcast(self, payload, src):
        self.broadcasts += 1
        raise RuntimeError("control transport broke")


def test_dist_runner_marks_a_failed_step_fatal_and_stops_collectives():
    comm = _BrokenComm()
    local = _ReleaseRunner()
    driver = DistTPModelRunner(comm, local)

    with pytest.raises(RuntimeError, match="control transport broke"):
        driver.execute((), {})
    assert isinstance(driver.fatal_error, RuntimeError)

    with pytest.raises(RuntimeError, match="unavailable"):
        driver.execute((), {})
    driver.release("request")
    driver.shutdown()

    assert comm.broadcasts == 1
    assert local.released == ["request"]
