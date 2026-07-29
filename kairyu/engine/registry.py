"""Backend factory registry: names (used by the DSL) to backend constructors."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from kairyu.engine.backend import EngineBackend
from kairyu.engine.config_validation import validate_backend_options
from kairyu.engine.mock import MockBackend

BackendFactory = Callable[..., EngineBackend]

_FACTORIES: dict[str, BackendFactory] = {}
_BUILTIN_FACTORIES: dict[str, BackendFactory] = {}

# Backends that register themselves when their module is imported.
_LAZY_MODULES = {
    "openai": "kairyu.engine.openai_backend",
    "vllm": "kairyu.engine.vllm_backend",
    "kairyu": "kairyu.engine.kairyu_backend",
    "kairyu-proc": "kairyu.engine.zmq_backend",
}


def register_backend(name: str, factory: BackendFactory) -> None:
    _FACTORIES[name] = factory
    if (name == "mock" and factory is MockBackend) or (
        factory.__module__ == _LAZY_MODULES.get(name)
    ):
        _BUILTIN_FACTORIES[name] = factory


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def known_backends() -> tuple[str, ...]:
    """Return registered and built-in lazy backend names without importing them."""

    return tuple(sorted({*_FACTORIES, *_LAZY_MODULES}))


def _factory_for_config_validation(name: str) -> BackendFactory | None:
    """Internal lazy lookup used by the side-effect-free custom validator."""

    return _FACTORIES.get(name)


def uses_builtin_backend_contract(name: str) -> bool:
    """Whether ``name`` still resolves to its built-in constructor contract."""

    factory = _FACTORIES.get(name)
    if factory is None:
        return name in _LAZY_MODULES
    return _BUILTIN_FACTORIES.get(name) is factory


def create_backend(name: str, **kwargs: object) -> EngineBackend:
    if name not in _FACTORIES and name not in _LAZY_MODULES:
        known = ", ".join(known_backends())
        raise ValueError(f"unknown backend {name!r}; known backends: {known}")
    validate_backend_options(name, kwargs)
    if name not in _FACTORIES and name in _LAZY_MODULES:
        importlib.import_module(_LAZY_MODULES[name])
    factory = _FACTORIES.get(name)
    if factory is None:
        raise RuntimeError("validated backend did not register its factory")
    return factory(**kwargs)


register_backend("mock", MockBackend)
