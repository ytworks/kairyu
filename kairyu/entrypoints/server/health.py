"""Liveness, readiness, and metrics endpoints (design m7 D4)."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from kairyu.engine.backend import EngineBackend
from kairyu.entrypoints.server.metrics import ServerMetrics
from kairyu.orchestration.replica import ReplicaPool


def add_health_routes(
    app: FastAPI,
    engines: Mapping[str, EngineBackend],
    metrics: ServerMetrics | None,
) -> None:
    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        # Engines are constructed by the time the app exists; pools additionally
        # need >=1 healthy replica or every request would fail.
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
