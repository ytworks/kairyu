"""Immutable-length views over append-only engine state."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar, overload

_T = TypeVar("_T")


class FrozenPrefix(Sequence[_T], Generic[_T]):
    """Expose the current prefix of a sequence without copying it.

    The backing sequence may grow, but this view's length never changes.  That
    makes append-only scheduler lists safe to hand to an overlapping runner
    while keeping snapshot construction O(1).  Pickling materializes only the
    visible prefix, so a live backing list never leaks later appends.
    """

    __slots__ = ("_values", "_length")

    def __init__(self, values: Sequence[_T]) -> None:
        if isinstance(values, FrozenPrefix):
            self._values = values._values
            self._length = len(values)
        else:
            self._values = values
            self._length = len(values)

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, index: int) -> _T: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[_T, ...]: ...

    def __getitem__(self, index: int | slice) -> _T | tuple[_T, ...]:
        if isinstance(index, slice):
            return tuple(
                self._values[position]
                for position in range(*index.indices(self._length))
            )
        position = index + self._length if index < 0 else index
        if position < 0 or position >= self._length:
            raise IndexError("frozen prefix index out of range")
        return self._values[position]

    def __iter__(self) -> Iterator[_T]:
        for position in range(self._length):
            yield self._values[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence) or len(other) != self._length:
            return False
        return all(left == right for left, right in zip(self, other, strict=True))

    def __repr__(self) -> str:
        return repr(tuple(self))

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __reduce__(self):
        return type(self), (tuple(self),)
