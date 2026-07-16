"""Liveness, readiness, and metrics endpoints (design m7 D4)."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from kairyu.engine.backend import EngineBackend
# Import the pure name resolver from the module (NOT the attention package,
# whose __init__ pulls in torch_backend) so importing health.py stays torch-free.
from kairyu.engine.core.attention.selector import select_backend_name
from kairyu.engine.core.hw_profile import probe
from kairyu.entrypoints.server.metrics import ServerMetrics
from kairyu.orchestration.replica import ReplicaPool

# type(engine).__name__ -> engine-registry backend name. Kept local (not a class
# attr) so this endpoint needs no engine-class change and stays robust to tests
# that construct backends directly.
_ENGINE_LABELS = {
    "MockBackend": "mock",
    "KairyuBackend": "kairyu",
    "OpenAICompatBackend": "openai",
    "VLLMBackend": "vllm",
    "ZmqEngineBackend": "kairyu-proc",
    "ReplicaPool": "replica-pool",
}
# Engine backends that run attention locally in-process (so the resolved
# attention backend applies to them); remote/echo engines report null.
_LOCAL_ATTENTION_BACKENDS = frozenset({"kairyu", "kairyu-proc"})


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
        if request.scope.get("state", {}).get("is_admin"):
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

    @app.get("/backends")
    async def backends() -> dict:
        """Report the resolved attention backend, library versions, and the
        per-engine backend map (m13). Open endpoint (see middleware _OPEN_PATHS);
        disclosure level matches the existing public /readyz and /metrics.

        attention backend is a process-level decision (env override or probed hw
        profile — deterministic and shared), so it is resolved once here with
        ``select_backend_name(probe())`` rather than deep-walking each engine."""
        from importlib.metadata import PackageNotFoundError, version

        override = os.environ.get("KAIRYU_ATTENTION_BACKEND")
        try:
            profile = probe()
            attention = select_backend_name(profile)
            kernel_tier = profile.kernel_tier
        except Exception:  # introspection must never 500; fall back to torch
            attention, kernel_tier = "torch", "torch"

        def _pkg_version(name: str) -> str | None:
            try:
                return version(name)
            except PackageNotFoundError:
                return None

        versions = {"torch": _pkg_version("torch")}
        if attention == "flashinfer":  # only meaningful when it is the resolved kernel
            versions["flashinfer"] = _pkg_version("flashinfer")

        engine_list = []
        for name, engine in engines.items():
            label = _ENGINE_LABELS.get(type(engine).__name__, type(engine).__name__)
            engine_list.append(
                {
                    "model": name,
                    "engine_backend": label,
                    "attention_backend": (
                        attention if label in _LOCAL_ATTENTION_BACKENDS else None
                    ),
                }
            )

        return {
            "attention_backend": attention,
            "source": "env" if override else "hw_profile",
            "kernel_tier": kernel_tier,
            "versions": versions,
            "engines": engine_list,
        }

    if metrics is not None:

        @app.get("/metrics")
        async def metrics_endpoint() -> Response:
            body, content_type = metrics.render()
            return Response(content=body, media_type=content_type)
