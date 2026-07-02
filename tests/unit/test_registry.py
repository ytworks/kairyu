import pytest

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
