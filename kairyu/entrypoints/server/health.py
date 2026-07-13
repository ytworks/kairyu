"""Liveness, readiness, and metrics endpoints (design m7 D4)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from kairyu.engine.backend import EngineBackend
from kairyu.entrypoints.server.metrics import ServerMetrics
from kairyu.orchestration.replica import ReplicaPool


def add_health_routes(
    app: FastAPI,
    engines: Mapping[str, EngineBackend],
    metrics: ServerMetrics | None,
    admin_keys: Iterable[str] = (),
) -> None:
    admin_key_set = frozenset(admin_keys)

    def _forbidden_if_not_admin(request: Request) -> JSONResponse | None:
        # when admin keys are configured, /admin/* state changes require one, so
        # an ordinary data-plane key cannot drain the node (S5). With no admin
        # keys set, behavior is unchanged (auth-gated when api keys exist).
        if not admin_key_set:
            return None
        caller = request.scope.get("state", {}).get("api_key")
        if caller in admin_key_set:
            return None
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "message": "admin privilege required",
                    "type": "invalid_request_error",
                    "code": "admin_required",
                }
            },
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/admin/drain")
    async def drain_node(request: Request):
        """Node-level drain (m10a A5): flips /readyz to 503; gateway-side pool
        drains go through PoolReconciler membership instead.

        NOT in the auth exempt list; additionally requires an ADMIN key when
        admin keys are configured (S5). Keyless deployments are the node-to-node
        trusted-mesh mode (m7 D5) by explicit choice."""
        denied = _forbidden_if_not_admin(request)
        if denied is not None:
            return denied
        app.state.draining = True
        return {"status": "draining"}

    @app.post("/admin/undrain")
    async def undrain_node(request: Request):
        """Clear the drain flag so the node reports ready again (S5) — without
        this a drained node could only recover via a process restart."""
        denied = _forbidden_if_not_admin(request)
        if denied is not None:
            return denied
        app.state.draining = False
        return {"status": "ready"}

    @app.get("/readyz")
    async def readyz():
        # m10a A5: a drained node reports unready so the prober/load-balancer
        # stops sending NEW work; in-flight requests keep completing.
        if getattr(app.state, "draining", False):
            return JSONResponse(
                status_code=503, content={"status": "draining"}
            )
        # Engines are constructed by the time the app exists; pools additionally
        # need >=1 validated, non-ejected replica or every request would fail.
        # Declared remote readiness URLs therefore remain false here until the
        # startup prober succeeds; backend traffic is never implicit validation.
        degraded = {
            name: engine.healthy
            for name, engine in engines.items()
            if isinstance(engine, ReplicaPool) and not any(engine.healthy)
        }
        if degraded:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unready",
                    "pools": {name: list(health) for name, health in degraded.items()},
                },
            )
        return {"status": "ready"}

    if metrics is not None:

        @app.get("/metrics")
        async def metrics_endpoint() -> Response:
            body, content_type = metrics.render()
            return Response(content=body, media_type=content_type)
