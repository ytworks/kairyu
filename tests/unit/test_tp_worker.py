"""Distributed TP driver/worker control protocol."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from kairyu.engine.core import worker as worker_module
from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.step_input import RequestSnapshot, StepDelta
from kairyu.engine.core.worker import (
    DistTPModelRunner,
    make_handshake,
    validate_handshake,
    worker_step_loop,
)


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
        self.batched_prefill_enabled = True
        self.prefill_stats = {
            "enabled": True,
            "rows": 8,
            "model_calls": 1,
        }
        self.batched_verification_enabled = True
        self.verification_stats = {
            "enabled": True,
            "batches": 4,
            "tokens": 12,
        }
        self.decode_page_table_cache_enabled = True
        self.page_table_stats = {
            "enabled": True,
            "builds": 5,
            "rows": 16,
        }

    def set_batched_prefill_enabled(self, enabled: bool) -> None:
        self.batched_prefill_enabled = enabled
        self.prefill_stats["enabled"] = enabled

    def prefill_execution_stats(self, *, reset: bool = False):
        result = dict(self.prefill_stats)
        if reset:
            self.prefill_stats["rows"] = 0
            self.prefill_stats["model_calls"] = 0
        return result

    def set_batched_verification_enabled(self, enabled: bool) -> None:
        self.batched_verification_enabled = enabled
        self.verification_stats["enabled"] = enabled

    def verification_execution_stats(self, *, reset: bool = False):
        result = dict(self.verification_stats)
        if reset:
            self.verification_stats["batches"] = 0
            self.verification_stats["tokens"] = 0
        return result

    def set_decode_page_table_cache_enabled(self, enabled: bool) -> None:
        self.decode_page_table_cache_enabled = enabled
        self.page_table_stats["enabled"] = enabled

    def decode_page_table_cache_stats(self, *, reset: bool = False):
        result = dict(self.page_table_stats)
        if reset:
            self.page_table_stats["builds"] = 0
            self.page_table_stats["rows"] = 0
        return result


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


def _backend_identity(
    *,
    kv_mode: str = "paged-materialized",
    sm: int = 120,
) -> str:
    return json.dumps(
        {
            "architecture": {
                "arch": "cuda",
                "device_name": "RTX PRO 6000 Blackwell Server Edition",
                "kernel_tier": "fa2",
                "sm": sm,
            },
            "components": {
                "decode": "flashinfer",
                "kv_mode": kv_mode,
                "prefill": "flashattention4",
            },
            "requested": "flashattention4",
            "resolved": "flashattention4",
            "source": "env",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.mark.parametrize(
    "worker_identity",
    [
        _backend_identity(kv_mode="paged-direct"),
        _backend_identity(sm=100),
    ],
    ids=["component-mismatch", "hardware-mismatch"],
)
def test_startup_handshake_requires_the_same_attention_backend_identity(
    tmp_path,
    worker_identity,
):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "test"}))
    driver_identity = _backend_identity()
    handshake = make_handshake(
        str(tmp_path),
        num_pages=64,
        page_size=16,
        attention_backend_identity=driver_identity,
    )

    validate_handshake(
        handshake,
        str(tmp_path),
        num_pages=64,
        page_size=16,
        attention_backend_identity=driver_identity,
    )
    with pytest.raises(RuntimeError, match="backend identity"):
        validate_handshake(
            handshake,
            str(tmp_path),
            num_pages=64,
            page_size=16,
            attention_backend_identity=worker_identity,
        )


def test_build_tp_runner_probes_and_binds_the_rank_local_device(monkeypatch):
    from kairyu.engine.core import attention as attention_module
    from kairyu.engine.core import attention_selector, hw_profile, kv_pool, model_runner, sampler
    from kairyu.models import parallel

    placement = worker_module.TPPlacement("cuda:3", torch.bfloat16, "nccl")
    profile = SimpleNamespace(
        arch="cuda",
        device_name="rank-3-gpu",
        sm=120,
        kernel_tier="fa2",
    )
    decision = SimpleNamespace(
        requested="auto",
        resolved="flashinfer",
        source="hw_profile",
        components={
            "prefill": "flashinfer",
            "decode": "flashinfer",
            "kv_mode": "paged-direct",
        },
        architecture={
            "arch": "cuda",
            "device_name": "rank-3-gpu",
            "sm": 120,
            "kernel_tier": "fa2",
        },
    )
    backend = SimpleNamespace(selection_decision=decision)
    local_config = SimpleNamespace(
        num_hidden_layers=2,
        kv_cache_num_heads=4,
        kv_cache_head_dim=16,
    )
    full_config = object()
    seen: dict[str, object] = {}

    def fake_probe(device):
        seen["probe_device"] = device
        return profile

    def fake_select_backend(actual_profile, *, device):
        seen["selection"] = (actual_profile, device)
        return backend

    def fake_identity(actual_decision):
        seen["identity_decision"] = actual_decision
        return "canonical-rank-local-identity"

    def fake_build_tp_model(
        model_dir,
        tp,
        rank,
        comm,
        *,
        dtype,
        device,
        attention_backend,
    ):
        seen["model"] = {
            "model_dir": model_dir,
            "tp": tp,
            "rank": rank,
            "comm": comm,
            "dtype": dtype,
            "device": device,
            "attention_backend": attention_backend,
        }
        return object(), local_config, full_config

    def fake_pool(**kwargs):
        seen["pool"] = kwargs
        return object()

    class FakeRunner:
        def __init__(self, model, pool, **kwargs):
            seen["runner"] = {
                "model": model,
                "pool": pool,
                **kwargs,
            }

    monkeypatch.setattr(hw_profile, "probe", fake_probe)
    monkeypatch.setattr(attention_module, "select_backend", fake_select_backend)
    monkeypatch.setattr(
        attention_selector,
        "attention_backend_identity",
        fake_identity,
        raising=False,
    )
    monkeypatch.setattr(parallel, "build_tp_model", fake_build_tp_model)
    monkeypatch.setattr(kv_pool, "PagedKVPool", fake_pool)
    monkeypatch.setattr(model_runner, "PagedModelRunner", FakeRunner)
    monkeypatch.setattr(sampler, "Sampler", lambda **kwargs: SimpleNamespace(**kwargs))

    comm = object()
    runner, returned_full_config = worker_module.build_tp_runner(
        "/model",
        tp=4,
        rank=3,
        comm=comm,
        num_pages=64,
        page_size=16,
        vocab=["token"],
        placement=placement,
    )

    assert seen["probe_device"] == "cuda:3"
    assert seen["selection"] == (profile, "cuda:3")
    assert seen["identity_decision"] is decision
    assert seen["model"] == {
        "model_dir": "/model",
        "tp": 4,
        "rank": 3,
        "comm": comm,
        "dtype": torch.bfloat16,
        "device": "cuda:3",
        "attention_backend": backend,
    }
    assert seen["pool"]["device"] == "cuda:3"
    assert runner.attention_backend_decision is decision
    assert runner.attention_backend_identity == "canonical-rank-local-identity"
    assert returned_full_config is full_config


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


def test_idle_shutdown_batches_and_deduplicates_releases_exactly_once():
    comms = FakeCommunicator.create_group(2)
    local = (_ReleaseRunner(), _ReleaseRunner())
    driver = DistTPModelRunner(comms[0], local[0])
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(worker_step_loop, comms[1], local[1])
        driver.release("first")
        driver.release("first")
        driver.release("second")
        driver.release("second")
        driver.shutdown()
        assert worker.result(timeout=2) == 0

    assert local[0].released == ["first", "second"]
    assert local[1].released == ["first", "second"]


class _RecordingControl:
    def __init__(self, comm) -> None:
        self._comm = comm
        self.payloads: list[object] = []

    def __getattr__(self, name):
        return getattr(self._comm, name)

    def broadcast(self, payload, src):
        self.payloads.append(payload)
        return self._comm.broadcast(payload, src)


class _BlockingStepControl:
    def __init__(self) -> None:
        self.step_entered = Event()
        self.continue_step = Event()
        self.non_step_payload_entered = Event()
        self.payloads: list[object] = []

    def broadcast(self, payload, src):
        self.payloads.append(payload)
        if isinstance(payload, StepDelta):
            self.step_entered.set()
            if not self.continue_step.wait(timeout=5):
                raise RuntimeError("test did not release the blocked step")
        else:
            self.non_step_payload_entered.set()
        return payload

    def tensor_broadcast(self, tensor, src):
        return tensor


def test_out_of_band_control_waits_for_the_active_step_transaction():
    control = _BlockingStepControl()
    local = _DiagnosticRunner(
        (),
        sampling_owner=True,
        sampler_present=True,
        device="cpu",
    )
    driver = DistTPModelRunner(control, local)
    toggle_started = Event()

    def toggle() -> None:
        toggle_started.set()
        driver.set_batched_prefill_enabled(False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        execute = pool.submit(driver.execute, (), {})
        assert control.step_entered.wait(timeout=5)
        pending_toggle = pool.submit(toggle)
        assert toggle_started.wait(timeout=5)
        try:
            # The toggle thread is running, but its control payload cannot be
            # inserted between StepDelta and the model/token collectives.
            assert not control.non_step_payload_entered.wait(timeout=0.1)
            assert len(control.payloads) == 1
        finally:
            control.continue_step.set()
        execute.result(timeout=5)
        pending_toggle.result(timeout=5)

    assert control.non_step_payload_entered.is_set()
    assert len(control.payloads) == 2
    assert local.batched_prefill_enabled is False
    driver.shutdown()


def test_same_id_release_after_snapshot_survives_the_older_ack():
    control = _BlockingStepControl()
    local = _ReleaseRunner()
    driver = DistTPModelRunner(control, local)
    driver.release("reused")

    with ThreadPoolExecutor(max_workers=1) as pool:
        execute = pool.submit(driver.execute, (), {})
        assert control.step_entered.wait(timeout=5)
        try:
            # This is a newer release generation than the one in the blocked
            # StepDelta. Its matching-generation ack must leave it pending.
            driver.release("reused")
        finally:
            control.continue_step.set()
        execute.result(timeout=5)

    assert local.released == ["reused"]
    driver.shutdown()
    assert local.released == ["reused", "reused"]
    assert control.payloads[0].released_ids == ("reused",)
    assert control.payloads[1].released_ids == ("reused",)


class _ReleaseOrderRunner(_AuthorityRunner):
    def __init__(
        self,
        authoritative: tuple[int, ...],
        *,
        block_first_execute: bool = False,
    ) -> None:
        super().__init__(authoritative)
        self.events: list[str] = []
        self.first_execute_entered = Event()
        self.continue_first_execute = Event()
        self._block_first_execute = block_first_execute

    def execute(self, scheduled, states):
        request_id = scheduled[0].request_id
        self.events.append(f"execute:{request_id}")
        if self._block_first_execute and len(self.events) == 1:
            self.first_execute_entered.set()
            if not self.continue_first_execute.wait(timeout=5):
                raise RuntimeError("test did not release the blocked execute")
        return super().execute(scheduled, states)

    def execute_passive(self, scheduled, states):
        self.events.append(f"execute-passive:{scheduled[0].request_id}")
        return super().execute_passive(scheduled, states)

    def release(self, request_id: str) -> None:
        self.events.append(f"release:{request_id}")
        super().release(request_id)


def test_release_returns_during_execute_and_piggybacks_on_the_next_step():
    control = FakeCommunicator.create_group(2)
    model = FakeCommunicator.create_group(2)
    recorded_control = _RecordingControl(control[0])
    local = (
        _ReleaseOrderRunner((17,), block_first_execute=True),
        _ReleaseOrderRunner((999,)),
    )
    driver = DistTPModelRunner(recorded_control, local[0], model[0])
    chunks = (ScheduledChunk("reused", 1, True, 0),)
    states = {"reused": _snapshot("reused")}

    with ThreadPoolExecutor(max_workers=3) as pool:
        worker = pool.submit(worker_step_loop, control[1], local[1], model[1])
        first_execute = pool.submit(driver.execute, chunks, states)
        assert local[0].first_execute_entered.wait(timeout=5)
        release = pool.submit(driver.release, "reused")
        try:
            # release() neither waits for the active protocol transaction nor
            # inserts a second control broadcast underneath it.
            release.result(timeout=1)
            assert len(recorded_control.payloads) == 1
            assert local[0].released == []
            assert local[1].released == []
        finally:
            local[0].continue_first_execute.set()
        first_execute.result(timeout=5)

        # Reusing the same id with an identical snapshot must still be a full
        # "new" entry, and both ranks release old local state before executing it.
        driver.execute(chunks, states)
        driver.shutdown()
        assert worker.result(timeout=5) == 2

    step_deltas = [
        payload for payload in recorded_control.payloads if isinstance(payload, StepDelta)
    ]
    assert len(step_deltas) == 2
    assert step_deltas[0].released_ids == ()
    assert step_deltas[1].released_ids == ("reused",)
    assert [snapshot.request_id for snapshot in step_deltas[1].new] == ["reused"]
    assert step_deltas[1].updates == ()
    assert local[0].events == [
        "execute:reused",
        "release:reused",
        "execute:reused",
    ]
    assert local[1].events == [
        "execute-passive:reused",
        "release:reused",
        "execute-passive:reused",
    ]


class _BlockingShutdownControl:
    def __init__(self) -> None:
        self.entered = Event()
        self.continue_shutdown = Event()
        self.payloads: list[object] = []

    def broadcast(self, payload, src):
        self.payloads.append(payload)
        self.entered.set()
        if not self.continue_shutdown.wait(timeout=5):
            raise RuntimeError("test did not release the blocked shutdown")
        return payload


def test_release_racing_shutdown_is_included_or_rejected():
    control = _BlockingShutdownControl()
    local = _ReleaseRunner()
    driver = DistTPModelRunner(control, local)
    driver.release("before-fence")

    with ThreadPoolExecutor(max_workers=1) as pool:
        shutdown = pool.submit(driver.shutdown)
        assert control.entered.wait(timeout=5)
        try:
            with pytest.raises(RuntimeError, match="shutting down"):
                driver.release("after-fence")
        finally:
            control.continue_shutdown.set()
        shutdown.result(timeout=5)

    # Shutdown is idempotent and does not open a second race window.
    driver.shutdown()
    assert len(control.payloads) == 1
    assert control.payloads[0].released_ids == ("before-fence",)
    assert local.released == ["before-fence"]


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


def test_prefill_mode_and_stats_probe_reach_every_rank_and_loop_resumes():
    control = FakeCommunicator.create_group(3)
    model = FakeCommunicator.create_group(3)
    local = tuple(
        _DiagnosticRunner(
            (17,) if rank == 0 else (999,),
            sampling_owner=rank == 0,
            sampler_present=rank == 0,
            device=f"cuda:{rank}",
        )
        for rank in range(3)
    )
    driver = DistTPModelRunner(control[0], local[0], model[0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [
            pool.submit(worker_step_loop, control[rank], local[rank], model[rank])
            for rank in (1, 2)
        ]
        try:
            driver.set_batched_prefill_enabled(False)
            rows = driver.prefill_execution_stats(reset=True)
            sampled = driver.execute(
                (ScheduledChunk("a", 1, True, 0),),
                {"a": _snapshot("a")},
            )
        finally:
            driver.shutdown()
        assert [worker.result(timeout=2) for worker in workers] == [1, 1]

    assert [row["rank"] for row in rows] == [0, 1, 2]
    assert [row["device"] for row in rows] == ["cuda:0", "cuda:1", "cuda:2"]
    assert all(row["stats"]["enabled"] is False for row in rows)
    assert all(row["stats"]["rows"] == 8 for row in rows)
    assert all(runner.prefill_stats["rows"] == 0 for runner in local)
    assert sampled == {"a": ("rank-0-public-result",)}


def test_prefill_stats_rank_failure_is_gathered_without_control_hang():
    control = FakeCommunicator.create_group(3, timeout_s=0.2)
    model = FakeCommunicator.create_group(3, timeout_s=0.2)
    local = [
        _DiagnosticRunner(
            (),
            sampling_owner=rank == 0,
            sampler_present=rank == 0,
            device=f"cuda:{rank}",
        )
        for rank in range(3)
    ]

    def fail_stats(*, reset: bool = False):
        raise RuntimeError("probe exploded")

    local[1].prefill_execution_stats = fail_stats
    driver = DistTPModelRunner(control[0], local[0], model[0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [
            pool.submit(
                worker_step_loop,
                control[rank],
                local[rank],
                model[rank],
            )
            for rank in (1, 2)
        ]
        with pytest.raises(
            RuntimeError,
            match=r"rank 1: RuntimeError: probe exploded",
        ):
            driver.prefill_execution_stats()
        # The diagnostic failure marks the public runner fatal, so stop the
        # still-healthy fake workers with the raw control protocol.
        control[0].broadcast(None, src=0)
        assert [worker.result(timeout=2) for worker in workers] == [0, 0]

    assert isinstance(driver.fatal_error, RuntimeError)


def test_verification_mode_and_stats_probe_reach_every_rank_and_loop_resumes():
    control = FakeCommunicator.create_group(3)
    model = FakeCommunicator.create_group(3)
    local = tuple(
        _DiagnosticRunner(
            (17,) if rank == 0 else (999,),
            sampling_owner=rank == 0,
            sampler_present=rank == 0,
            device=f"cuda:{rank}",
        )
        for rank in range(3)
    )
    driver = DistTPModelRunner(control[0], local[0], model[0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [
            pool.submit(worker_step_loop, control[rank], local[rank], model[rank])
            for rank in (1, 2)
        ]
        try:
            driver.set_batched_verification_enabled(False)
            rows = driver.verification_execution_stats(reset=True)
            sampled = driver.execute(
                (ScheduledChunk("a", 1, True, 0),),
                {"a": _snapshot("a")},
            )
        finally:
            driver.shutdown()
        assert [worker.result(timeout=2) for worker in workers] == [1, 1]

    assert [row["rank"] for row in rows] == [0, 1, 2]
    assert [row["device"] for row in rows] == ["cuda:0", "cuda:1", "cuda:2"]
    assert all(row["stats"]["enabled"] is False for row in rows)
    assert all(row["stats"]["batches"] == 4 for row in rows)
    assert all(runner.verification_stats["batches"] == 0 for runner in local)
    assert sampled == {"a": ("rank-0-public-result",)}


def test_verification_stats_rank_failure_is_gathered_without_control_hang():
    control = FakeCommunicator.create_group(3, timeout_s=0.2)
    model = FakeCommunicator.create_group(3, timeout_s=0.2)
    local = [
        _DiagnosticRunner(
            (),
            sampling_owner=rank == 0,
            sampler_present=rank == 0,
            device=f"cuda:{rank}",
        )
        for rank in range(3)
    ]

    def fail_stats(*, reset: bool = False):
        raise RuntimeError("verification probe exploded")

    local[1].verification_execution_stats = fail_stats
    driver = DistTPModelRunner(control[0], local[0], model[0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [
            pool.submit(
                worker_step_loop,
                control[rank],
                local[rank],
                model[rank],
            )
            for rank in (1, 2)
        ]
        with pytest.raises(
            RuntimeError,
            match=r"rank 1: RuntimeError: verification probe exploded",
        ):
            driver.verification_execution_stats()
        control[0].broadcast(None, src=0)
        assert [worker.result(timeout=2) for worker in workers] == [0, 0]

    assert isinstance(driver.fatal_error, RuntimeError)


def test_page_table_mode_and_stats_probe_reach_every_rank_and_loop_resumes():
    control = FakeCommunicator.create_group(3)
    model = FakeCommunicator.create_group(3)
    local = tuple(
        _DiagnosticRunner(
            (17,) if rank == 0 else (999,),
            sampling_owner=rank == 0,
            sampler_present=rank == 0,
            device=f"cuda:{rank}",
        )
        for rank in range(3)
    )
    driver = DistTPModelRunner(control[0], local[0], model[0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [
            pool.submit(worker_step_loop, control[rank], local[rank], model[rank])
            for rank in (1, 2)
        ]
        try:
            driver.set_decode_page_table_cache_enabled(False)
            rows = driver.decode_page_table_cache_stats(reset=True)
            sampled = driver.execute(
                (ScheduledChunk("a", 1, True, 0),),
                {"a": _snapshot("a")},
            )
        finally:
            driver.shutdown()
        assert [worker.result(timeout=2) for worker in workers] == [1, 1]

    assert [row["rank"] for row in rows] == [0, 1, 2]
    assert [row["device"] for row in rows] == ["cuda:0", "cuda:1", "cuda:2"]
    assert all(row["stats"]["enabled"] is False for row in rows)
    assert all(row["stats"]["builds"] == 5 for row in rows)
    assert all(runner.page_table_stats["builds"] == 0 for runner in local)
    assert all(runner.decode_page_table_cache_enabled is False for runner in local)
    assert sampled == {"a": ("rank-0-public-result",)}


def test_page_table_stats_rank_failure_is_gathered_and_marks_driver_fatal():
    control = FakeCommunicator.create_group(3, timeout_s=0.2)
    model = FakeCommunicator.create_group(3, timeout_s=0.2)
    local = [
        _DiagnosticRunner(
            (),
            sampling_owner=rank == 0,
            sampler_present=rank == 0,
            device=f"cuda:{rank}",
        )
        for rank in range(3)
    ]

    def fail_stats(*, reset: bool = False):
        raise RuntimeError("page-table probe exploded")

    local[1].decode_page_table_cache_stats = fail_stats
    driver = DistTPModelRunner(control[0], local[0], model[0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [
            pool.submit(
                worker_step_loop,
                control[rank],
                local[rank],
                model[rank],
            )
            for rank in (1, 2)
        ]
        with pytest.raises(
            RuntimeError,
            match=r"rank 1: RuntimeError: page-table probe exploded",
        ):
            driver.decode_page_table_cache_stats()
        assert isinstance(driver.fatal_error, RuntimeError)
        with pytest.raises(RuntimeError, match="unavailable"):
            driver.decode_page_table_cache_stats()
        control[0].broadcast(None, src=0)
        assert [worker.result(timeout=2) for worker in workers] == [0, 0]


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


def test_pending_release_survives_diff_failure_until_fatal_shutdown():
    comm = _BrokenComm()
    local = _ReleaseRunner()
    driver = DistTPModelRunner(comm, local)

    driver.release("queued-before-diff")
    with pytest.raises(KeyError):
        driver.execute(
            (ScheduledChunk("missing", 1, True, 0),),
            {},
        )
    assert isinstance(driver.fatal_error, KeyError)

    driver.release("queued-after-diff")
    driver.shutdown()

    assert comm.broadcasts == 0
    assert local.released == ["queued-before-diff", "queued-after-diff"]


def test_dist_runner_marks_a_failed_step_fatal_and_stops_collectives():
    comm = _BrokenComm()
    local = _ReleaseRunner()
    driver = DistTPModelRunner(comm, local)

    driver.release("queued-before-failure")
    with pytest.raises(RuntimeError, match="control transport broke"):
        driver.execute((), {})
    assert isinstance(driver.fatal_error, RuntimeError)

    with pytest.raises(RuntimeError, match="unavailable"):
        driver.execute((), {})
    driver.release("request")
    driver.shutdown()

    assert comm.broadcasts == 1
    assert local.released == ["queued-before-failure", "request"]


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
