"""DeploymentSpec -> running FastAPI app: engines, pools, prober, lifespan (m7 D3/D4)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from functools import partial
from pathlib import Path

import httpx
from fastapi import FastAPI

from kairyu.batch.store import BatchStore
from kairyu.batch.worker import BatchWorker
from kairyu.deploy.prober import HealthProber
from kairyu.deploy.registry import (
    KubernetesEndpointSliceDiscovery,
    PoolReconciler,
    openai_replica_factory,
)
from kairyu.deploy.spec import DeploymentSpec, load_deployment_spec
from kairyu.dsl.loader import build_orchestrator, load_spec
from kairyu.engine.backend import (
    EngineBackend,
    shutdown_all,
    shutdown_all_cancellation_safe,
)
from kairyu.engine.registry import create_backend
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.extra_routes import (
    EmbeddingBackend,
    MockEmbeddingBackend,
)
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.orchestration.replica import ReplicaPool
from kairyu.orchestration.router import JsonlRouterLog

_EMBEDDING_BACKEND_FACTORIES: dict[str, Callable[..., EmbeddingBackend]] = {
    "mock": MockEmbeddingBackend
}
_SERVICE_ACCOUNT_NAMESPACE_PATH = Path(
    "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
)


class _DynamicPoolHTTPClientOwner:
    """Exactly-once, cancellation-safe owner of one dynamic pool's client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self._close_task: asyncio.Task[None] | None = None

    async def shutdown(self) -> None:
        task = self._close_task
        if task is None:
            if self.client.is_closed:
                return
            task = asyncio.create_task(self.client.aclose())
            self._close_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as cancelled:
            # The client is not discoverable after application teardown. Finish
            # its close before propagating cancellation so sockets cannot leak.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            try:
                task.result()
            except BaseException as cleanup_error:
                raise cancelled from cleanup_error
            raise


def _create_dynamic_pool_http_client() -> httpx.AsyncClient:
    """Build the fleet-wide data/probe connection pool for one ReplicaPool.

    Dynamic discovery has no declared maximum fleet size. Keep both aggregate
    limits open so they cannot fall below the server concurrency cap or evict a
    ready replica's warmed connection. EndpointSlice membership is a trusted
    control-plane input; inbound request concurrency and the prober's own
    semaphore bound active connections, while finite idle expiry bounds sockets
    retained for churned-out pod IPs.
    """

    return httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=None,
            max_keepalive_connections=None,
            keepalive_expiry=30.0,
        )
    )


def _create_embedding_backend(backend: str, *, dimensions: int) -> EmbeddingBackend:
    factory = _EMBEDDING_BACKEND_FACTORIES.get(backend)
    if factory is None:
        known = ", ".join(sorted(_EMBEDDING_BACKEND_FACTORIES))
        raise ValueError(
            f"unknown embedding backend {backend!r}; known backends: {known}"
        )
    return factory(dimensions=dimensions)


def _resolve_path(path: str, base_dir: Path | None) -> Path:
    resolved = Path(path)
    if base_dir is not None and not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved


def _resolve_kubernetes_namespace(configured: str | None) -> str:
    if configured is not None:
        return configured
    namespace = _SERVICE_ACCOUNT_NAMESPACE_PATH.read_text(
        encoding="utf-8"
    ).strip()
    if not namespace:
        raise ValueError("mounted Kubernetes service-account namespace is empty")
    return namespace


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
                request_burst=limits.request_burst,
                token_burst=limits.token_burst,
                max_in_flight=limits.max_in_flight,
                interactive_priority=limits.interactive_priority,
                batch_priority=limits.batch_priority,
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
    embedding_backends = {
        name: _create_embedding_backend(
            section.backend, dimensions=section.dimensions
        )
        for name, section in spec.embeddings.items()
    }
    engines: dict[str, EngineBackend] = {
        name: create_backend(entry.backend, **entry.options)
        for name, entry in spec.engines.items()
    }

    probers: list[HealthProber] = []
    reconcilers: list[tuple[PoolReconciler, float]] = []
    dynamic_http_client_owners: list[_DynamicPoolHTTPClientOwner] = []
    placement_log_paths: dict[str, Path] = {}
    claimed_log_paths: dict[Path, str] = {}
    for name, pool_spec in spec.pools.items():
        if pool_spec.placement_log_path is None:
            continue
        path = _resolve_path(pool_spec.placement_log_path, base_dir)
        canonical = path.resolve()
        claimed_by = claimed_log_paths.get(canonical)
        if claimed_by is not None:
            raise ValueError(
                "pool placement_log_path values must be unique; "
                f"{name!r} and {claimed_by!r} both resolve to {canonical}"
            )
        claimed_log_paths[canonical] = name
        placement_log_paths[name] = path

    for name, pool_spec in spec.pools.items():
        pool_log = (
            JsonlRouterLog(placement_log_paths[name])
            if name in placement_log_paths
            else None
        )
        replicas = [
            create_backend(entry.backend, **entry.options)
            for entry in pool_spec.replicas
        ]
        dynamic = pool_spec.discovery is not None
        pool = ReplicaPool(
            replicas,
            log=pool_log,
            unhealthy_after=pool_spec.unhealthy_after,
            queue_depth_threshold=pool_spec.queue_depth_threshold,
            allow_empty=dynamic,
        )
        engines[name] = pool
        if not dynamic:
            health_urls = [
                entry.resolved_health_url() for entry in pool_spec.replicas
            ]
            if any(url is not None for url in health_urls):
                probers.append(
                    HealthProber(
                        name,
                        pool,
                        health_urls,
                        pool_spec.probe_interval_s,
                    )
                )
            continue

        discovery = pool_spec.discovery
        assert discovery is not None
        http_client_owner = _DynamicPoolHTTPClientOwner(
            _create_dynamic_pool_http_client()
        )
        dynamic_http_client_owners.append(http_client_owner)
        source = KubernetesEndpointSliceDiscovery(
            discovery.service,
            _resolve_kubernetes_namespace(discovery.namespace),
            discovery.port,
            model=discovery.model,
            api_key_env=discovery.api_key_env,
            scheme=discovery.scheme,
            address_family_preference=discovery.address_family_preference,
        )

        event_sink = None
        if pool_log is not None:

            def event_sink(
                actions: dict[str, object],
                *,
                _pool: ReplicaPool = pool,
                _log: JsonlRouterLog = pool_log,
            ) -> None:
                replica_ids = tuple(
                    str(replica_id)
                    for replica_id in actions.get(
                        "replica_ids",
                        _pool.replica_ids,
                    )
                )
                healthy_ids = tuple(
                    str(replica_id)
                    for replica_id in actions.get(
                        "healthy_ids",
                        tuple(
                            current_id
                            for current_id, healthy in (
                                _pool.healthy_by_id().items()
                            )
                            if healthy
                        ),
                    )
                )
                eligible_ids = tuple(
                    str(replica_id)
                    for replica_id in actions.get(
                        "eligible_ids",
                        _pool.eligible_ids,
                    )
                )
                raw_generations = actions.get("generation_by_id", {})
                generation_by_id = (
                    {
                        str(replica_id): str(generation)
                        for replica_id, generation in raw_generations.items()
                    }
                    if isinstance(raw_generations, dict)
                    else {}
                )
                _log.record_membership(
                    actions,
                    replica_ids=replica_ids,
                    healthy_ids=healthy_ids,
                    eligible_ids=eligible_ids,
                    generation_by_id=generation_by_id,
                )

        reconciler = PoolReconciler(
            pool,
            source,
            factory=partial(
                openai_replica_factory,
                client=http_client_owner.client,
            ),
            default_model=discovery.model or name,
            event_sink=event_sink,
        )
        reconcilers.append((reconciler, discovery.poll_interval_s))
        # Dynamic candidates carry their readiness URL on the pool entry.
        # An empty mapping deliberately avoids any bootstrap replica.
        probers.append(
            HealthProber(
                name,
                pool,
                {},
                pool_spec.probe_interval_s,
                client=http_client_owner.client,
                close_client=False,
            )
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
        tasks = [
            asyncio.create_task(reconciler.run(interval_s))
            for reconciler, interval_s in reconcilers
        ]
        tasks += [asyncio.create_task(prober.run()) for prober in probers]
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
            resources = list(engines.values())
            if orchestrator is not None:
                resources.append(orchestrator)
            resources.extend(orchestrators.values())
            # create_app owns the usage ledger in an outer lifespan wrapper,
            # so it closes only after workers stop and this shutdown completes
            # or raises an ExceptionGroup.
            try:
                await shutdown_all(resources, "application")
            finally:
                # Individual dynamic backends and probers are non-owners. Close
                # their common pool only after all of them and the reconciler
                # loops have stopped, even if another resource shutdown fails.
                await shutdown_all_cancellation_safe(
                    dynamic_http_client_owners,
                    "dynamic pool HTTP clients",
                )

    app = create_app(
        engines=engines,
        orchestrator=orchestrator,
        orchestrators=orchestrators,
        settings=server_settings,
        lifespan=lifespan,
        chat_templates=chat_templates,
        tenant_config=tenant_config,
        embedding_backends=embedding_backends,
        resolved_api_keys=api_keys,
        resolved_admin_keys=admin_keys,
        price_sheet=spec.pricing,
    )
    app.state.deployment_spec = spec
    app.state.probers = tuple(probers)
    app.state.reconcilers = tuple(
        reconciler for reconciler, _interval_s in reconcilers
    )
    app.state.dynamic_pool_http_clients = tuple(
        owner.client for owner in dynamic_http_client_owners
    )

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
            tenant_config=tenant_config,
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
