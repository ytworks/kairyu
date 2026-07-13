"""DeploymentSpec -> running FastAPI app: engines, pools, prober, lifespan (m7 D3/D4)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI

from kairyu.batch.store import BatchStore
from kairyu.batch.worker import BatchWorker
from kairyu.deploy.prober import HealthProber
from kairyu.deploy.spec import DeploymentSpec, load_deployment_spec
from kairyu.dsl.loader import build_orchestrator, load_spec
from kairyu.engine.backend import EngineBackend
from kairyu.engine.registry import create_backend
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.orchestration.replica import ReplicaPool


def _server_settings(spec: DeploymentSpec) -> ServerSettings:
    return ServerSettings(
        **{field: getattr(spec.server, field) for field in ServerSettings.model_fields}
    )


def _preflight_server(
    spec: DeploymentSpec,
) -> tuple[
    ServerSettings,
    TenantConfig | None,
    frozenset[str],
    frozenset[str],
]:
    settings = _server_settings(spec)
    api_keys = settings.resolve_api_keys()
    admin_keys = settings.resolve_admin_keys()
    section = spec.tenants
    if section is None:
        return settings, None, api_keys, admin_keys
    tenant_config = TenantConfig.from_mapping(
        key_tenants=section.key_tenants,
        limits={
            tenant: TenantLimits(
                requests_per_minute=limits.requests_per_minute,
                tokens_per_minute=limits.tokens_per_minute,
            )
            for tenant, limits in section.limits.items()
        },
        default_tenant=section.default_tenant,
        resolved_api_keys=api_keys,
    )
    return settings, tenant_config, api_keys, admin_keys


def build_app_from_spec(spec: DeploymentSpec, base_dir: Path | None = None) -> FastAPI:
    """Construct engines, pools, orchestrator, and the app with a prober lifespan."""
    server_settings, tenant_config, api_keys, admin_keys = _preflight_server(spec)
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

    chat_templates: dict[str, ChatTemplate] = {}
    for model_name, source in spec.chat_templates.items():
        template_source = source
        if source.endswith(".jinja") and base_dir is not None:
            path = Path(source)
            if not path.is_absolute():
                template_source = str(base_dir / path)
        chat_templates[model_name] = ChatTemplate.load(template_source)

    def _load_orchestrator(spec_path: str) -> Orchestrator:
        path = Path(spec_path)
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        return build_orchestrator(load_spec(path))

    orchestrator: Orchestrator | None = None
    if spec.orchestrator is not None:
        orchestrator = _load_orchestrator(spec.orchestrator.spec)
    orchestrators: dict[str, Orchestrator] = {
        name: _load_orchestrator(section.spec)
        for name, section in spec.orchestrators.items()
    }

    workers: list[BatchWorker] = []  # filled after create_app (worker needs app metrics)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = [asyncio.create_task(prober.run()) for prober in probers]
        tasks += [asyncio.create_task(worker.run()) for worker in workers]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                # a task that already crashed re-raises its stored exception on
                # await; swallow it (and the CancelledError) so one dead task
                # cannot skip the remaining awaits AND the engine shutdowns (M7)
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for engine in engines.values():
                try:
                    await engine.shutdown()  # each engine shuts down independently
                except Exception:
                    logging.getLogger("kairyu.deploy").exception("engine shutdown failed")

    app = create_app(
        engines=engines,
        orchestrator=orchestrator,
        orchestrators=orchestrators,
        settings=server_settings,
        lifespan=lifespan,
        chat_templates=chat_templates,
        tenant_config=tenant_config,
        resolved_api_keys=api_keys,
        resolved_admin_keys=admin_keys,
    )
    app.state.deployment_spec = spec
    app.state.probers = tuple(probers)

    if spec.batch is not None:
        from kairyu.entrypoints.server.batch_routes import add_batch_routes

        store = BatchStore(spec.batch.data_dir)
        store.recover_orphans()
        worker = BatchWorker(
            store,
            engines,
            max_concurrency=spec.batch.max_concurrency,
            metrics=app.state.metrics,
            chat_templates=chat_templates,  # batch and HTTP must render identically
            usage_ledger=getattr(app.state, "usage_ledger", None),
            tenant_limiter=getattr(app.state, "tenant_limiter", None),
        )
        workers.append(worker)
        add_batch_routes(app, store, worker)
        app.state.batch_store = store
        app.state.batch_worker = worker
    return app


def build_app_from_config(path: str | Path) -> FastAPI:
    """Load a DeploymentSpec YAML file and build the app (used by `kairyu serve`)."""
    config_path = Path(path)
    spec = load_deployment_spec(config_path)
    return build_app_from_spec(spec, base_dir=config_path.parent)
