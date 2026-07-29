from pathlib import Path

import pytest

from kairyu.engine import registry as registry_module
from kairyu.engine.config_validation import validate_backend_options
from kairyu.engine.mock import MockBackend
from kairyu.engine.registry import available_backends, create_backend, register_backend


def test_mock_backend_is_preregistered():
    backend = create_backend("mock")
    assert isinstance(backend, MockBackend)


def test_kwargs_forwarded_to_factory():
    backend = create_backend("mock", responses={"x": "y"})
    assert backend._responses == {"x": "y"}


def test_unknown_backend_lists_known_names():
    with pytest.raises(ValueError, match="mock"):
        create_backend("does-not-exist")


def test_lazy_backends_resolve_without_prior_import():
    backend = create_backend("openai", base_url="https://api.example.com/v1", model="m")
    assert type(backend).__name__ == "OpenAICompatBackend"


def test_register_custom_backend():
    register_backend("custom-test", lambda **kwargs: MockBackend(**kwargs))
    assert "custom-test" in available_backends()
    assert isinstance(create_backend("custom-test"), MockBackend)


@pytest.mark.parametrize(
    "options",
    [
        {"tensor_parallel_size": 0},
        {"responses": 1},
    ],
)
def test_mock_rejects_invalid_constructor_options(options):
    with pytest.raises(ValueError):
        validate_backend_options("mock", options)


def test_native_cuda_graph_without_model_is_rejected_before_import():
    with pytest.raises(ValueError, match="cuda_graph"):
        validate_backend_options("kairyu", {"decode_mode": "cuda_graph"})


def test_vllm_non_priority_policy_is_rejected_before_import():
    with pytest.raises(ValueError, match="scheduling_policy"):
        validate_backend_options(
            "vllm",
            {"model": "model", "scheduling_policy": "fcfs"},
        )


def test_custom_backend_signature_is_checked_without_construction():
    constructed = False

    def custom_factory(*, required):
        nonlocal constructed
        constructed = True
        return MockBackend(responses={"required": str(required)})

    register_backend("signature-test", custom_factory)

    with pytest.raises(ValueError, match="registered factory"):
        validate_backend_options("signature-test", {})

    assert not constructed


def test_builtin_name_override_uses_registered_factory_contract(monkeypatch):
    def replacement_factory(*, replacement):
        return MockBackend(responses={"replacement": str(replacement)})

    monkeypatch.setitem(
        registry_module._FACTORIES,
        "mock",
        replacement_factory,
    )

    validate_backend_options("mock", {"replacement": "accepted"})


@pytest.mark.parametrize(
    ("backend", "options"),
    [
        ("openai", {"base_url": "", "model": ""}),
        ("kairyu", {"speculative_tokens": -1}),
        ("kairyu", {"model_path": Path("")}),
        ("kairyu-proc", {"death_timeout_s": -1}),
    ],
)
def test_preflight_does_not_invent_constructor_constraints(backend, options):
    validate_backend_options(backend, options)
