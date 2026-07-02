"""FakeCommunicator behavior: the CPU seam for TP collectives (design m5 D2)."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from kairyu.engine.core.comm import FakeCommunicator

_FAST_TIMEOUT_S = 0.05
_TEST_JOIN_TIMEOUT_S = 5.0


def _group(world_size: int, timeout_s: float = 5.0) -> tuple[FakeCommunicator, ...]:
    return FakeCommunicator.create_group(world_size, timeout_s=timeout_s)


def test_create_group_assigns_sequential_ranks_and_shared_world_size():
    comms = _group(3)
    assert [comm.rank for comm in comms] == [0, 1, 2]
    assert all(comm.world_size == 3 for comm in comms)


def test_create_group_rejects_nonpositive_world_size():
    with pytest.raises(ValueError, match="world_size"):
        FakeCommunicator.create_group(0)


def test_constructor_rejects_rank_outside_group():
    comms = _group(2)
    with pytest.raises(ValueError, match="rank"):
        FakeCommunicator(rank=2, group=comms[0]._group)


def test_broadcast_delivers_src_object_to_every_rank():
    comms = _group(3)
    payload = {"step": 1, "chunks": (1, 2)}
    sent = comms[0].broadcast(payload, src=0)
    received = [comms[rank].broadcast(None, src=0) for rank in (1, 2)]
    assert sent is payload
    assert all(obj is payload for obj in received)


def test_broadcast_supports_multiple_rounds_in_order():
    comms = _group(2)
    comms[0].broadcast("first", src=0)
    comms[0].broadcast("second", src=0)
    assert comms[1].broadcast(None, src=0) == "first"
    assert comms[1].broadcast(None, src=0) == "second"


def test_broadcast_rejects_invalid_src():
    comms = _group(2)
    with pytest.raises(ValueError, match="src"):
        comms[0].broadcast("x", src=2)


def test_broadcast_before_src_fails_fast_instead_of_hanging():
    comms = _group(2, timeout_s=_FAST_TIMEOUT_S)
    with pytest.raises(RuntimeError, match="broadcast"):
        comms[1].broadcast(None, src=0)


def test_send_recv_preserves_fifo_order_per_pair():
    comms = _group(2)
    comms[0].send(1, "a")
    comms[0].send(1, "b")
    assert comms[1].recv(0) == "a"
    assert comms[1].recv(0) == "b"


def test_send_to_self_is_deliverable():
    comms = _group(1)
    comms[0].send(0, "loop")
    assert comms[0].recv(0) == "loop"


def test_recv_without_message_fails_fast():
    comms = _group(2, timeout_s=_FAST_TIMEOUT_S)
    with pytest.raises(RuntimeError, match="recv"):
        comms[0].recv(1)


def test_send_and_recv_reject_out_of_range_ranks():
    comms = _group(2)
    with pytest.raises(ValueError, match="dst"):
        comms[0].send(5, "x")
    with pytest.raises(ValueError, match="src"):
        comms[0].recv(-1)


def test_all_reduce_sums_elementwise_across_ranks():
    comms = _group(3)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(comms[rank].all_reduce, (float(rank), 10.0 * rank))
            for rank in range(3)
        ]
        results = [future.result(timeout=_TEST_JOIN_TIMEOUT_S) for future in futures]
    assert all(result == (3.0, 30.0) for result in results)


def test_all_reduce_world_size_one_returns_own_values():
    (comm,) = _group(1)
    assert comm.all_reduce((1.5, 2.5)) == (1.5, 2.5)


def test_all_reduce_mismatched_lengths_raises_for_all_ranks():
    comms = _group(2, timeout_s=1.0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(comms[0].all_reduce, (1.0, 2.0)),
            pool.submit(comms[1].all_reduce, (1.0,)),
        ]
        errors = []
        for future in futures:
            with pytest.raises(ValueError, match="length"):
                future.result(timeout=_TEST_JOIN_TIMEOUT_S)
            errors.append(True)
    assert errors == [True, True]


def test_all_gather_returns_values_ordered_by_rank():
    comms = _group(3)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(comms[rank].all_gather, f"v{rank}") for rank in range(3)]
        results = [future.result(timeout=_TEST_JOIN_TIMEOUT_S) for future in futures]
    assert all(result == ("v0", "v1", "v2") for result in results)


def test_barrier_releases_all_ranks_together():
    comms = _group(4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(comms[rank].barrier) for rank in range(4)]
        for future in futures:
            assert future.result(timeout=_TEST_JOIN_TIMEOUT_S) is None


def test_barrier_times_out_when_a_rank_never_arrives():
    comms = _group(2, timeout_s=_FAST_TIMEOUT_S)
    with pytest.raises(RuntimeError, match="barrier"):
        comms[0].barrier()
