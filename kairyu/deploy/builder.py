"""DeploymentSpec -> running FastAPI app: engines, pools, prober, lifespan (m7 D3/D4)."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI

from kairyu.deploy.prober import HealthProber
from kairyu.deploy.spec import DeploymentSpec, load_deployment_spec
from kairyu.dsl.loader import build_orchestrator, load_spec
from kairyu.engine.backend import EngineBackend
from kairyu.engine.registry import create_backend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.orchestration.replica import ReplicaPool


def _server_settings(spec: DeploymentSpec) -> ServerSettings:
    return ServerSettings(
        **{field: getattr(spec.server, field) for field in ServerSettings.model_fields}
    )


def build_app_from_spec(spec: DeploymentSpec, base_dir: Path | None = None) -> FastAPI:
    """Construct engines, pools, orchestrator, and the app with a prober lifespan."""
    engines: dict[str, EngineBackend] = {
        name: create_backend(entry.backend, **entry.options)
        for name, entry in spec.engines.items()
    }

    probers: list[HealthProber] = []
    for name, pool_spec in spec.pools.items():
        replicas = [
            create_backend(entry.backend, **entry.options) for entry in pool_spec.replicas
        ]
        pool = ReplicaPool(
            replicas,
            unhealthy_after=pool_spec.unhealthy_after,
            queue_depth_threshold=pool_spec.queue_depth_threshold,
        )
        engines[name] = pool
        health_urls = [entry.resolved_health_url() for entry in pool_spec.replicas]
        if any(url is not None for url in health_urls):
            probers.append(
                HealthProber(name, pool, health_urls, pool_spec.probe_interval_s)
            )

    orchestrator: Orchestrator | None = None
    if spec.orchestrator is not None:
        orchestrator_path = Path(spec.orchestrator.spec)
        if base_dir is not None and not orchestrator_path.is_absolute():
            orchestrator_path = base_dir / orchestrator_path
        orchestrator = build_orchestrator(load_spec(orchestrator_path))

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = [asyncio.create_task(prober.run()) for prober in probers]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            for engine in engines.values():
                await engine.shutdown()

    app = create_app(
        engines=engines,
        orchestrator=orchestrator,
        settings=_server_settings(spec),
        lifespan=lifespan,
    )
    app.state.deployment_spec = spec
    app.state.probers = tuple(probers)
    return app


def build_app_from_config(path: str | Path) -> FastAPI:
    """Load a DeploymentSpec YAML file and build the app (used by `kairyu serve`)."""
    config_path = Path(path)
    spec = load_deployment_spec(config_path)
    return build_app_from_spec(spec, base_dir=config_path.parent)
