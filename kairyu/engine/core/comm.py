"""Collective-communication seam for the TP driver/worker step protocol (design m5 D2).

``Communicator`` is the only place collective communication appears in the
engine; the GPU phase adds ``NcclCommunicator`` behind the same protocol.
``FakeCommunicator`` is the in-process, deterministic implementation for CPU
tests: broadcast and send/recv work sequentially (source first, receivers
after), while all_reduce / all_gather / barrier synchronize concurrent rank
threads. Every wait fails fast with a clear ``RuntimeError`` after
``timeout_s`` instead of hanging a test run.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Protocol

_DEFAULT_TIMEOUT_S = 5.0


class Communicator(Protocol):
    """Minimal collective seam (design m5 D2): no custom collectives beyond these."""

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def broadcast(self, obj: object, src: int) -> object:
        """Deliver ``src``'s object to every rank; returns the delivered object."""
        ...

    def all_reduce(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """Elementwise sum of every rank's values."""
        ...

    def all_gather(self, value: object) -> tuple:
        """Every rank's value, ordered by rank."""
        ...

    def barrier(self) -> None:
        """Block until every rank in the group has arrived."""
        ...

    def send(self, dst: int, obj: object) -> None:
        """Point-to-point send; per-(src, dst) FIFO ordering."""
        ...

    def recv(self, src: int) -> object:
        """Receive the next object sent from ``src`` to this rank."""
        ...


class _FakeGroup:
    """State shared by every FakeCommunicator in one group."""

    def __init__(self, world_size: int, timeout_s: float) -> None:
        self.world_size = world_size
        self.timeout_s = timeout_s
        self.condition = threading.Condition()
        self.broadcasts: dict[int, list[object]] = {}
        self.fifos: dict[tuple[int, int], deque[object]] = {}
        self.reduce_rounds: list[dict[int, tuple[float, ...]]] = []
        self.reduce_errors: list[str | None] = []
        self.gather_rounds: list[dict[int, object]] = []
        self.barrier_rounds: list[set[int]] = []


class FakeCommunicator:
    """In-process deterministic Communicator; construct groups via create_group()."""

    def __init__(self, rank: int, group: _FakeGroup) -> None:
        if not 0 <= rank < group.world_size:
            raise ValueError(f"rank must be in [0, {group.world_size}), got {rank}")
        self._rank = rank
        self._group = group
        self._broadcast_pos: dict[int, int] = {}
        self._reduce_round = 0
        self._gather_round = 0
        self._barrier_round = 0

    @classmethod
    def create_group(
        cls, world_size: int, timeout_s: float = _DEFAULT_TIMEOUT_S
    ) -> tuple[FakeCommunicator, ...]:
        """One communicator per rank, all sharing the same in-process group state."""
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        group = _FakeGroup(world_size=world_size, timeout_s=timeout_s)
        return tuple(cls(rank=rank, group=group) for rank in range(world_size))

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._group.world_size

    def _check_rank(self, value: int, name: str) -> None:
        if not 0 <= value < self.world_size:
            raise ValueError(f"{name} must be in [0, {self.world_size}), got {value}")

    def broadcast(self, obj: object, src: int) -> object:
        self._check_rank(src, "src")
        group = self._group
        position = self._broadcast_pos.get(src, 0)
        with group.condition:
            history = group.broadcasts.setdefault(src, [])
            if self._rank == src:
                history.append(obj)
                group.condition.notify_all()
            elif not group.condition.wait_for(
                lambda: len(history) > position, timeout=group.timeout_s
            ):
                raise RuntimeError(
                    f"broadcast timeout: rank {self._rank} waited for src {src} "
                    f"round {position} but nothing was broadcast"
                )
            self._broadcast_pos[src] = position + 1
            return history[position]

    def all_reduce(self, values: tuple[float, ...]) -> tuple[float, ...]:
        group = self._group
        round_index = self._reduce_round
        self._reduce_round += 1
        contribution = tuple(float(value) for value in values)
        with group.condition:
            while len(group.reduce_rounds) <= round_index:
                group.reduce_rounds.append({})
                group.reduce_errors.append(None)
            contributions = group.reduce_rounds[round_index]
            error = group.reduce_errors[round_index]
            if error is None and any(
                len(other) != len(contribution) for other in contributions.values()
            ):
                error = (
                    f"all_reduce length mismatch in round {round_index}: rank "
                    f"{self._rank} contributed {len(contribution)} values"
                )
                group.reduce_errors[round_index] = error
                group.condition.notify_all()
            if error is not None:
                raise ValueError(error)
            contributions[self._rank] = contribution
            group.condition.notify_all()
            done = group.condition.wait_for(
                lambda: len(contributions) == self.world_size
                or group.reduce_errors[round_index] is not None,
                timeout=group.timeout_s,
            )
            error = group.reduce_errors[round_index]
            if error is not None:
                raise ValueError(error)
            if not done:
                raise RuntimeError(
                    f"all_reduce timeout: rank {self._rank} round {round_index} "
                    f"({len(contributions)}/{self.world_size} ranks arrived)"
                )
            ordered = [contributions[rank] for rank in range(self.world_size)]
            return tuple(sum(column) for column in zip(*ordered, strict=True))

    def all_gather(self, value: object) -> tuple:
        group = self._group
        round_index = self._gather_round
        self._gather_round += 1
        with group.condition:
            while len(group.gather_rounds) <= round_index:
                group.gather_rounds.append({})
            contributions = group.gather_rounds[round_index]
            contributions[self._rank] = value
            group.condition.notify_all()
            if not group.condition.wait_for(
                lambda: len(contributions) == self.world_size, timeout=group.timeout_s
            ):
                raise RuntimeError(
                    f"all_gather timeout: rank {self._rank} round {round_index} "
                    f"({len(contributions)}/{self.world_size} ranks arrived)"
                )
            return tuple(contributions[rank] for rank in range(self.world_size))

    def barrier(self) -> None:
        group = self._group
        round_index = self._barrier_round
        self._barrier_round += 1
        with group.condition:
            while len(group.barrier_rounds) <= round_index:
                group.barrier_rounds.append(set())
            arrived = group.barrier_rounds[round_index]
            arrived.add(self._rank)
            group.condition.notify_all()
            if not group.condition.wait_for(
                lambda: len(arrived) == self.world_size, timeout=group.timeout_s
            ):
                raise RuntimeError(
                    f"barrier timeout: rank {self._rank} round {round_index} "
                    f"({len(arrived)}/{self.world_size} ranks arrived)"
                )

    def send(self, dst: int, obj: object) -> None:
        self._check_rank(dst, "dst")
        group = self._group
        with group.condition:
            group.fifos.setdefault((self._rank, dst), deque()).append(obj)
            group.condition.notify_all()

    def recv(self, src: int) -> object:
        self._check_rank(src, "src")
        group = self._group
        with group.condition:
            fifo = group.fifos.setdefault((src, self._rank), deque())
            if not group.condition.wait_for(lambda: bool(fifo), timeout=group.timeout_s):
                raise RuntimeError(
                    f"recv timeout: rank {self._rank} waited for a message from src {src}"
                )
            return fifo.popleft()
