"""Request-local reuse of models prevalidated outside the event loop."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

_ModelT = TypeVar("_ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class _PrevalidatedModel:
    input_value: object
    value: BaseModel


_prevalidated_model: contextvars.ContextVar[_PrevalidatedModel | None] = contextvars.ContextVar(
    "kairyu_prevalidated_request_model", default=None
)


@contextmanager
def reuse_prevalidated_model(
    input_value: object,
    value: BaseModel,
) -> Iterator[None]:
    """Make one exact parsed input reusable during FastAPI dependency solving."""

    token = _prevalidated_model.set(_PrevalidatedModel(input_value, value))
    try:
        yield
    finally:
        _prevalidated_model.reset(token)


def prevalidated_model_for(
    model: type[_ModelT],
    input_value: object,
) -> _ModelT | None:
    """Return the request-local model only for its exact parsed input object."""

    prepared = _prevalidated_model.get()
    if (
        prepared is None
        or prepared.input_value is not input_value
        or not isinstance(prepared.value, model)
    ):
        return None
    return prepared.value
