"""Backend factory registry: names (used by the DSL) to backend constructors."""

from __future__ import annotations

from collections.abc import Callable

from kairyu.engine.backend import EngineBackend
from kairyu.engine.mock import MockBackend

BackendFactory = Callable[..., EngineBackend]

_FACTORIES: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    _FACTORIES[name] = factory


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def create_backend(name: str, **kwargs: object) -> EngineBackend:
    factory = _FACTORIES.get(name)
    if factory is None:
        known = ", ".join(available_backends())
        raise ValueError(f"unknown backend {name!r}; known backends: {known}")
    return factory(**kwargs)


register_backend("mock", MockBackend)
