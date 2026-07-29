"""Distributed TP driver/worker control protocol."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from kairyu.engine.core import worker as worker_module
from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.step_input import RequestSnapshot, StepDelta
from kairyu.engine.core.worker import DistTPModelRunner, worker_step_loop


class _ReleaseRunner:
    def __init__(self) -> None:
        self.released: list[str] = []

    def execute(self, scheduled, states):
        return {}

    def execute_passive(self, scheduled, states):
        return {}

    def make_sampling_token_packet(self, scheduled, states, sampled=None):
        return torch.full((len(scheduled),), -1, dtype=torch.int64)

    def adopt_sampling_token_packet(self, scheduled, states, packet) -> None:
        pass

    def release(self, request_id: str) -> None:
        self.released.append(request_id)


class _AuthorityRunner(_ReleaseRunner):
    def __init__(self, authoritative: tuple[int, ...]) -> None:
        super().__init__()
        self.authoritative = authoritative
        self.sampled_calls = 0
        self.passive_calls = 0
        self.packet_sources: list[object] = []
        self.adopted: list[tuple[int, ...]] = []

    def execute(self, scheduled, states):
        self.sampled_calls += 1
        return {"a": ("rank-0-public-result",)}

    def execute_passive(self, scheduled, states):
        self.passive_calls += 1
        return {}

    def make_sampling_token_packet(self, scheduled, states, sampled=None):
        self.packet_sources.append(sampled)
        if sampled is None:
            return torch.full((len(scheduled),), -1, dtype=torch.int64)
        assert len(self.authoritative) == len(scheduled)
        return torch.tensor(self.authoritative, dtype=torch.int64)

    def adopt_sampling_token_packet(self, scheduled, states, packet) -> None:
        assert packet.shape == (len(scheduled),)
        self.adopted.append(tuple(packet.tolist()))


class _DiagnosticRunner(_AuthorityRunner):
    def __init__(
        self,
        authoritative: tuple[int, ...],
        *,
        sampling_owner: bool,
        sampler_present: bool,
        device: str,
    ) -> None:
        super().__init__(authoritative)
        self.sampling_owner = sampling_owner
        self._sampler = object() if sampler_present else None
        self._device = torch.device(device)


def _snapshot(request_id: str) -> RequestSnapshot:
    return RequestSnapshot(
        request_id=request_id,
        prompt_token_ids=(1, 2),
        computed_prompt=2,
        outputs=(),
        in_flight=0,
        page_ids=(0,),
        decode_page_ids=(0,),
        eos_token_id=None,
        max_new_tokens=4,
    )


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


def test_rank_zero_samples_once_and_workers_adopt_its_fixed_packet():
    control = FakeCommunicator.create_group(2)
    model = FakeCommunicator.create_group(2)
    local = (
        _AuthorityRunner((17, -1)),
        _AuthorityRunner((999, 999)),
    )
    driver = DistTPModelRunner(control[0], local[0], model[0])
    chunks = (
        ScheduledChunk("a", 1, False, 2),
        ScheduledChunk("b", 1, True, 0),
    )
    states = {"a": _snapshot("a"), "b": _snapshot("b")}

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(worker_step_loop, control[1], local[1], model[1])
        try:
            sampled = driver.execute(chunks, states)
        finally:
            driver.shutdown()
        assert worker.result(timeout=2) == 1

    assert sampled == {"a": ("rank-0-public-result",)}
    assert local[0].sampled_calls == 1
    assert local[0].passive_calls == 0
    assert local[1].sampled_calls == 0
    assert local[1].passive_calls == 1
    assert local[0].packet_sources == [sampled]
    assert local[1].packet_sources == [None]
    assert local[0].adopted == [(17, -1)]
    assert local[1].adopted == [(17, -1)]


def test_sampling_ownership_metadata_is_rank_sorted_and_step_loop_resumes():
    control = FakeCommunicator.create_group(3)
    model = FakeCommunicator.create_group(3)
    local = (
        _DiagnosticRunner((17,), sampling_owner=True, sampler_present=True, device="cuda:0"),
        _DiagnosticRunner((999,), sampling_owner=False, sampler_present=False, device="cuda:1"),
        _DiagnosticRunner((999,), sampling_owner=False, sampler_present=False, device="cuda:2"),
    )
    driver = DistTPModelRunner(control[0], local[0], model[0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [
            pool.submit(worker_step_loop, control[rank], local[rank], model[rank])
            for rank in (1, 2)
        ]
        try:
            rows = driver.sampling_ownership_metadata()
            sampled = driver.execute(
                (ScheduledChunk("a", 1, True, 0),),
                {"a": _snapshot("a")},
            )
        finally:
            driver.shutdown()
        assert [worker.result(timeout=2) for worker in workers] == [1, 1]

    assert rows == (
        {
            "rank": 0,
            "control_world_size": 3,
            "control_backend": "fake",
            "model_world_size": 3,
            "model_backend": "fake",
            "sampling_owner": True,
            "sampler_present": True,
            "device": "cuda:0",
        },
        {
            "rank": 1,
            "control_world_size": 3,
            "control_backend": "fake",
            "model_world_size": 3,
            "model_backend": "fake",
            "sampling_owner": False,
            "sampler_present": False,
            "device": "cuda:1",
        },
        {
            "rank": 2,
            "control_world_size": 3,
            "control_backend": "fake",
            "model_world_size": 3,
            "model_backend": "fake",
            "sampling_owner": False,
            "sampler_present": False,
            "device": "cuda:2",
        },
    )
    assert sampled == {"a": ("rank-0-public-result",)}


def test_sampling_ownership_metadata_reports_missing_rank():
    control = FakeCommunicator.create_group(2, timeout_s=0.01)
    model = FakeCommunicator.create_group(2)
    local = _DiagnosticRunner((), sampling_owner=True, sampler_present=True, device="cpu")
    driver = DistTPModelRunner(control[0], local, model[0])

    with pytest.raises(
        RuntimeError,
        match=r"metadata probe failed: all_gather timeout: rank 0 round 0 \(1/2",
    ):
        driver.sampling_ownership_metadata()

    assert isinstance(driver.fatal_error, RuntimeError)


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (
            lambda row: (row, {**row, "rank": 0}),
            "duplicate rank 0",
        ),
        (
            lambda row: (row, {key: value for key, value in row.items() if key != "device"}),
            r"gather slot 1 has malformed fields: missing=\['device'\]",
        ),
    ],
)
def test_sampling_ownership_metadata_rejects_bad_replies(monkeypatch, corrupt, message):
    control = FakeCommunicator.create_group(2)
    model = FakeCommunicator.create_group(2)
    local = _DiagnosticRunner((), sampling_owner=True, sampler_present=True, device="cpu")
    driver = DistTPModelRunner(control[0], local, model[0])
    monkeypatch.setattr(control[0], "all_gather", lambda row: corrupt(row))

    with pytest.raises(RuntimeError, match=message):
        driver.sampling_ownership_metadata()

    assert isinstance(driver.fatal_error, RuntimeError)


def test_sampling_ownership_metadata_uses_torch_process_group_backend(monkeypatch):
    group = object()
    comm = type("_TorchLikeComm", (), {"group": group})()
    seen: list[object] = []

    def fake_get_backend(actual_group):
        seen.append(actual_group)
        return "nccl"

    monkeypatch.setattr(torch.distributed, "get_backend", fake_get_backend)

    assert worker_module._communicator_backend(comm) == "nccl"
    assert seen == [group]


def test_step_delta_dropped_does_not_release_request():
    control = FakeCommunicator.create_group(2)
    model = FakeCommunicator.create_group(2)
    local = (_ReleaseRunner(), _ReleaseRunner())
    driver = DistTPModelRunner(control[0], local[0], model[0])
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(worker_step_loop, control[1], local[1], model[1])
        control[0].broadcast(
            StepDelta(chunks=(), new=(), updates=(), dropped=("preempted",)), src=0
        )
        model[0].tensor_broadcast(torch.empty(0, dtype=torch.int64), src=0)
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


class _BrokenTensorComm:
    def __init__(self) -> None:
        self.broadcasts = 0

    def tensor_broadcast(self, tensor, src):
        self.broadcasts += 1
        raise RuntimeError("sampling token broadcast broke")


def test_sampling_packet_failure_marks_runner_fatal():
    (control,) = FakeCommunicator.create_group(1)
    model = _BrokenTensorComm()
    local = _AuthorityRunner(())
    driver = DistTPModelRunner(control, local, model)

    with pytest.raises(RuntimeError, match="sampling token broadcast broke"):
        driver.execute((), {})
    assert isinstance(driver.fatal_error, RuntimeError)

    with pytest.raises(RuntimeError, match="unavailable"):
        driver.execute((), {})

    assert model.broadcasts == 1
    assert local.sampled_calls == 1
    assert local.adopted == []
