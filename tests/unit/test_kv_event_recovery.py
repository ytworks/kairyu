"""Focused regressions for the reliable KV-event recovery state machine."""

from __future__ import annotations

import json
import sys
import threading
import uuid
from types import SimpleNamespace

import pytest

from kairyu.orchestration.kv_index import (
    KvEventIndex,
    ZmqKvEventPublisher,
    ZmqKvEventSubscriber,
)


def _stored(*hashes: str) -> dict[str, object]:
    return {"type": "BlockStored", "block_hashes": list(hashes)}


class _FakeSubscriberSocket:
    def __init__(self, messages: list[list[bytes]]) -> None:
        self._messages = list(messages)

    def recv_multipart(self, *, flags: int) -> list[bytes]:
        del flags
        if not self._messages:
            import zmq

            raise zmq.Again()
        return self._messages.pop(0)


def _subscriber_with_messages(
    index: KvEventIndex,
    messages: list[list[bytes]],
) -> ZmqKvEventSubscriber:
    subscriber = object.__new__(ZmqKvEventSubscriber)
    subscriber._socket = _FakeSubscriberSocket(messages)
    subscriber._index = index
    return subscriber


def test_delayed_sequence_zero_cannot_retrust_same_epoch_gap() -> None:
    index = KvEventIndex()

    assert index.apply_sequenced("r1", "epoch-a", 2, _stored("late")) == "gap"
    assert index.apply_sequenced("r1", "epoch-a", 0, _stored("first")) == "applied"

    view = index.view("r1")
    assert view.stream_epoch == "epoch-a"
    assert view.last_sequence == 0
    assert view.state == "syncing"
    assert not view.trusted
    assert index.overlap("r1", ["first"]) is None


def test_atomic_routing_waits_for_matching_high_water_confirmation() -> None:
    index = KvEventIndex(now=lambda: 1.0)

    assert (
        index.apply_sequenced(
            "r1",
            "epoch-a",
            0,
            _stored("h0"),
        )
        == "applied"
    )
    assert index.is_live("r1")
    assert not index.all_live(("r1",))
    assert index.route_overlaps(("r1",), ("h0",)) is None

    assert index.heartbeat(
        "r1",
        stream_epoch="epoch-a",
        high_sequence=0,
    )
    assert index.all_live(("r1",))
    assert index.route_overlaps(("r1",), ("h0",)) == (1,)


def test_exact_membership_scoring_releases_lock_and_revalidates_feed() -> None:
    index = KvEventIndex(now=lambda: 1.0)
    assert index.install_snapshot("r1", "epoch-a", 0, ["h0"])
    scoring = threading.Event()
    resume = threading.Event()

    class BlockingSet(set[str]):
        def __contains__(self, value: object) -> bool:
            scoring.set()
            assert resume.wait(timeout=2)
            return super().__contains__(value)

    with index._lock:
        index._replicas["r1"].hashes = BlockingSet({"h0"})

    result: list[tuple[int, ...] | None] = []
    route_thread = threading.Thread(
        target=lambda: result.append(index.route_overlaps(("r1",), ("h0",)))
    )
    route_thread.start()
    assert scoring.wait(timeout=1)

    changed = threading.Event()

    def mark_gap() -> None:
        index.mark_gap("r1", stream_epoch="epoch-a")
        changed.set()

    mutation_thread = threading.Thread(target=mark_gap)
    mutation_thread.start()
    try:
        assert changed.wait(timeout=1), "membership scoring retained the index lock"
    finally:
        resume.set()
        route_thread.join(timeout=2)
        mutation_thread.join(timeout=2)

    assert not route_thread.is_alive()
    assert not mutation_thread.is_alive()
    assert result == [None]


def test_retired_epoch_cannot_replace_or_clear_current_epoch() -> None:
    index = KvEventIndex(now=lambda: 1.0)
    assert index.apply_sequenced("r1", "epoch-old", 0, _stored("old")) == "applied"
    assert index.install_snapshot("r1", "epoch-new", 3, ["new"])
    before = index.view("r1")

    assert (
        index.apply_sequenced(
            "r1",
            "epoch-old",
            1,
            {"type": "AllBlocksCleared"},
        )
        == "stale_epoch"
    )
    assert not index.install_snapshot("r1", "epoch-old", 9, ["stale"])
    assert not index.heartbeat(
        "r1",
        stream_epoch="epoch-old",
        high_sequence=1,
    )
    assert index.view("r1") == before


def test_inactive_epoch_flood_cannot_evict_forgotten_active_epoch() -> None:
    index = KvEventIndex()
    index.register_replica("r1")
    assert index.apply_sequenced("r1", "epoch-old", 0, _stored("old")) == "applied"
    index.forget_replica("r1")

    for sequence in range(32):
        assert (
            index.apply_sequenced(
                "r1",
                f"unknown-{sequence}",
                0,
                _stored(f"noise-{sequence}"),
            )
            == "inactive"
        )

    index.register_replica("r1")
    assert index.apply_sequenced("r1", "epoch-old", 1, _stored("stale")) == "stale_epoch"
    assert index.view("r1").block_hashes == ()


def test_active_epoch_flood_cannot_evict_forgotten_active_epoch() -> None:
    index = KvEventIndex(now=lambda: 1.0)
    assert index.install_snapshot("r1", "epoch-forgotten", 0, ["old"])
    index.forget_replica("r1")
    index.register_replica("r1")

    for sequence in range(32):
        assert (
            index.apply_sequenced(
                "r1",
                f"epoch-noise-{sequence}",
                0,
                _stored(f"noise-{sequence}"),
            )
            == "applied"
        )

    assert (
        index.apply_sequenced(
            "r1",
            "epoch-forgotten",
            0,
            _stored("revived"),
        )
        == "stale_epoch"
    )
    assert not index.install_snapshot(
        "r1",
        "epoch-forgotten",
        0,
        ["revived"],
    )


def test_inactive_epoch_rejection_does_not_poison_reregistered_epoch() -> None:
    index = KvEventIndex()
    index.register_replica("r1")
    assert index.apply_sequenced("r1", "epoch-old", 0, _stored("old")) == "applied"
    index.forget_replica("r1")

    assert index.apply_sequenced("r1", "epoch-new", 0, _stored("ignored")) == "inactive"
    assert not index.mark_gap("r1", stream_epoch="epoch-new")
    assert not index.install_snapshot("r1", "epoch-new", 0, ["ignored"])
    assert not index.mark_synchronized("r1", "epoch-new", 0)
    assert not index.heartbeat(
        "r1",
        stream_epoch="epoch-new",
        high_sequence=0,
    )

    index.register_replica("r1")
    assert index.apply_sequenced("r1", "epoch-new", 0, _stored("accepted")) == "applied"
    view = index.view("r1")
    assert view.stream_epoch == "epoch-new"
    assert view.block_hashes == ("accepted",)


def test_two_frame_message_cannot_downgrade_sequenced_replica(caplog) -> None:
    pytest.importorskip("zmq")
    index = KvEventIndex()
    assert index.apply_sequenced("r1", "epoch-a", 0, _stored("kept")) == "applied"
    subscriber = _subscriber_with_messages(
        index,
        [
            [
                b"r1",
                json.dumps({"type": "AllBlocksCleared"}).encode(),
            ]
        ],
    )

    assert subscriber.drain() == 1

    view = index.view("r1")
    assert view.stream_epoch == "epoch-a"
    assert view.last_sequence == 0
    assert view.block_hashes == ("kept",)
    assert view.state == "gap"
    assert not view.trusted
    assert "ValueError" in caplog.text


def test_wrong_wire_version_heartbeat_is_rejected_without_refresh(caplog) -> None:
    pytest.importorskip("zmq")
    clock = {"now": 0.0}
    index = KvEventIndex(now=lambda: clock["now"])
    assert index.apply_sequenced("r1", "epoch-a", 0, _stored("kept")) == "applied"
    clock["now"] = 0.2
    envelope = {
        "wire_version": 999,
        "kind": "heartbeat",
        "stream_epoch": "epoch-a",
        "high_sequence": 0,
        "emitted_ns": 1,
    }
    subscriber = _subscriber_with_messages(
        index,
        [
            [
                b"r1",
                (0).to_bytes(8, "big", signed=True),
                json.dumps(envelope).encode(),
            ]
        ],
    )

    assert subscriber.drain() == 1

    view = index.view("r1")
    assert view.state == "gap"
    assert not view.trusted
    assert view.age_s == pytest.approx(0.2)
    assert "ValueError" in caplog.text


def test_subscriber_rejects_non_owner_thread_before_touching_socket() -> None:
    class UntouchedSocket:
        close_calls = 0

        def close(self, *, linger: int) -> None:
            del linger
            self.close_calls += 1

    subscriber = object.__new__(ZmqKvEventSubscriber)
    subscriber._owner_thread_id = threading.get_ident()
    subscriber._closed = False
    subscriber._paused = False
    subscriber._socket = UntouchedSocket()
    errors: list[BaseException] = []

    def pause_from_non_owner() -> None:
        try:
            subscriber.pause()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=pause_from_non_owner)
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "non-owner thread" in str(errors[0])
    assert subscriber._socket.close_calls == 0


def test_publisher_closes_partially_created_sockets_on_startup_failure(
    monkeypatch,
) -> None:
    class FakeSocket:
        def __init__(self, kind: int) -> None:
            self.kind = kind
            self.closed = threading.Event()

        def bind(self, endpoint: str) -> None:
            del endpoint
            if self.kind == 2:
                raise OSError("replay bind failed")

        def close(self, *, linger: int) -> None:
            del linger
            self.closed.set()

    class FakeContext:
        def __init__(self) -> None:
            self.sockets: list[FakeSocket] = []

        def socket(self, kind: int) -> FakeSocket:
            socket = FakeSocket(kind)
            self.sockets.append(socket)
            return socket

    context = FakeContext()
    fake_zmq = SimpleNamespace(
        PUB=1,
        ROUTER=2,
        Context=SimpleNamespace(instance=lambda: context),
    )
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)

    with pytest.raises(RuntimeError, match="socket setup failed"):
        ZmqKvEventPublisher(
            "inproc://publisher",
            "r1",
            replay_endpoint="inproc://replay",
        )

    assert len(context.sockets) == 2
    assert all(socket.closed.wait(timeout=1) for socket in context.sockets)


def test_atomic_view_cannot_mix_sequence_metadata_and_hashes() -> None:
    reader_entered_now = threading.Event()
    release_reader = threading.Event()
    block_reader = {"enabled": False}

    def now() -> float:
        if block_reader["enabled"] and threading.current_thread().name == "reader":
            reader_entered_now.set()
            assert release_reader.wait(timeout=1)
        return 1.0

    index = KvEventIndex(now=now)
    assert index.apply_sequenced("r1", "epoch-a", 0, _stored("h0")) == "applied"
    block_reader["enabled"] = True
    observed: list[object] = []
    reader_errors: list[BaseException] = []
    writer_errors: list[BaseException] = []
    writer_started = threading.Event()

    def read_view() -> None:
        try:
            observed.append(index.view("r1"))
        except BaseException as error:
            reader_errors.append(error)

    def apply_next() -> None:
        writer_started.set()
        try:
            index.apply_sequenced("r1", "epoch-a", 1, _stored("h1"))
        except BaseException as error:
            writer_errors.append(error)

    reader = threading.Thread(target=read_view, name="reader")
    writer = threading.Thread(target=apply_next, name="writer")
    reader.start()
    assert reader_entered_now.wait(timeout=1)
    writer.start()
    assert writer_started.wait(timeout=1)
    release_reader.set()
    reader.join(timeout=1)
    writer.join(timeout=1)

    assert not reader.is_alive()
    assert not writer.is_alive()
    assert not reader_errors
    assert not writer_errors
    assert len(observed) == 1
    old_view = observed[0]
    assert old_view.last_sequence == 0
    assert old_view.block_hashes == ("h0",)
    new_view = index.view("r1")
    assert new_view.last_sequence == 1
    assert new_view.block_hashes == ("h0", "h1")


def test_publisher_rotate_epoch_resets_source_without_restarting_thread() -> None:
    pytest.importorskip("zmq")
    publisher = ZmqKvEventPublisher(
        f"inproc://kv-epoch-rotation-{uuid.uuid4().hex}",
        "r1",
        heartbeat_s=60.0,
        stream_epoch="epoch-old",
    )
    publisher_identity = id(publisher)
    source_thread = publisher._thread
    source_thread_ident = source_thread.ident
    old_cache_sink = publisher.event_sink()
    try:
        old_cache_sink(_stored("old"))
        assert publisher.high_sequence == 0
        assert publisher.authoritative_hashes() == ("old",)
        assert len(publisher._journal) == 1

        assert publisher.rotate_epoch(stream_epoch="epoch-new") == "epoch-new"
        new_cache_sink = publisher.event_sink()
        assert id(publisher) == publisher_identity
        assert publisher._thread is source_thread
        assert publisher._thread.ident == source_thread_ident
        assert publisher._thread.is_alive()
        assert publisher.stream_epoch == "epoch-new"
        assert publisher.high_sequence == -1
        assert publisher.authoritative_hashes() == ()
        assert tuple(publisher._journal) == ()

        with pytest.raises(RuntimeError, match="retired cache epoch"):
            old_cache_sink(_stored("late-old"))
        with pytest.raises(RuntimeError, match="retired cache epoch"):
            publisher(_stored("ambiguous-direct-call"))
        assert publisher.high_sequence == -1
        assert publisher.authoritative_hashes() == ()

        new_cache_sink(_stored("new"))
        assert publisher.high_sequence == 0
        assert publisher.authoritative_hashes() == ("new",)
        assert len(publisher._journal) == 1
        sequence, payload = publisher._journal[0]
        envelope = json.loads(payload)
        assert sequence == 0
        assert envelope["sequence"] == 0
        assert envelope["stream_epoch"] == "epoch-new"
        assert envelope["event"] == _stored("new")
    finally:
        publisher.close()

    with pytest.raises(RuntimeError, match="closed"):
        publisher.rotate_epoch(stream_epoch="epoch-after-close")
