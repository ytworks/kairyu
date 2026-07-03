"""Shared fixtures: an in-process mock gateway (single model + orchestrations)."""

import httpx
import pytest

from kairyu.bench.types import BenchConfig, BenchTarget
from kairyu.dsl.loader import build_orchestrator, load_spec
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app

ORCHESTRATOR_YAML = """
workers:
  - { name: tier1, backend: mock }
  - { name: tier2, backend: mock }
"""


@pytest.fixture()
def gateway_app():
    return create_app(
        engines={"m": MockBackend()},
        orchestrators={
            "kairyu-auto": build_orchestrator(load_spec(ORCHESTRATOR_YAML)),
            "kairyu-auto-max": build_orchestrator(load_spec(ORCHESTRATOR_YAML)),
        },
    )


@pytest.fixture()
def http_factory(gateway_app):
    def factory() -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=gateway_app)
        return httpx.AsyncClient(transport=transport, base_url="http://gw")

    return factory


def make_target(model: str = "m", **kwargs) -> BenchTarget:
    return BenchTarget(base_url="http://gw/v1", model=model, **kwargs)


def make_config(tmp_path, models=("m", "kairyu-auto"), **overrides) -> BenchConfig:
    defaults = dict(
        targets=tuple(make_target(model) for model in models),
        offline_fixtures=True,
        download=False,
        smoke=True,
        results_dir=str(tmp_path / "results"),
        cache_dir=str(tmp_path / "cache"),
        run_id="test-run",
    )
    defaults.update(overrides)
    return BenchConfig(**defaults)
