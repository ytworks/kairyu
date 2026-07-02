"""Backend factory registry: names (used by the DSL) to backend constructors."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from kairyu.engine.backend import EngineBackend
from kairyu.engine.mock import MockBackend

BackendFactory = Callable[..., EngineBackend]

_FACTORIES: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    _FACTORIES[name] = factory


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


# Backends that register themselves when their module is imported.
_LAZY_MODULES = {
    "openai": "kairyu.engine.openai_backend",
    "vllm": "kairyu.engine.vllm_backend",
    "kairyu": "kairyu.engine.kairyu_backend",
}


def create_backend(name: str, **kwargs: object) -> EngineBackend:
    if name not in _FACTORIES and name in _LAZY_MODULES:
        importlib.import_module(_LAZY_MODULES[name])
    factory = _FACTORIES.get(name)
    if factory is None:
        known = ", ".join(sorted({*available_backends(), *_LAZY_MODULES}))
        raise ValueError(f"unknown backend {name!r}; known backends: {known}")
    return factory(**kwargs)


register_backend("mock", MockBackend)
