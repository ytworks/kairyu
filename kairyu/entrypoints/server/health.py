"""Liveness, readiness, and metrics endpoints (design m7 D4)."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from kairyu.engine.backend import EngineBackend, EngineReadiness
from kairyu.engine.core.attention_selector import (
    AttentionBackendDecision,
    attention_backend_identity,
    select_backend_decision,
)
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

    def _engine_readiness() -> dict[str, EngineReadiness]:
        """Per-engine self-report; engines without the optional hook are skipped."""
        reported = {}
        for name, engine in engines.items():
            probe_readiness = getattr(engine, "readiness", None)
            if probe_readiness is None:
                continue
            try:
                reported[name] = probe_readiness()
            except Exception as error:  # introspection must never 500
                # class name only: this reaches an unauthenticated endpoint and
                # an exception message can carry a path or an upstream URL
                reported[name] = EngineReadiness(
                    False, f"readiness check raised {type(error).__name__}"
                )
        return reported

    @app.get("/health")
    async def health():
        """Liveness. A FATAL engine fault answers 503 so an orchestrator replaces
        the process — nothing in-process can undo dead tensor-parallel ranks, and
        without this the container stays up forever serving nothing."""
        fatal = {
            name: status.detail for name, status in _engine_readiness().items() if status.fatal
        }
        if fatal:
            return JSONResponse(status_code=503, content={"status": "fatal", "engines": fatal})
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
            return JSONResponse(status_code=503, content={"status": "draining"})
        # Engines are constructed by the time the app exists; pools additionally
        # need >=1 validated, non-ejected replica or every request would fail.
        # Declared remote readiness URLs therefore remain false here until the
        # startup prober succeeds; backend traffic is never implicit validation.
        degraded = {
            name: {
                "healthy": list(engine.healthy),
                "draining": list(engine.draining_by_id().values()),
                "eligible_ids": list(engine.eligible_ids),
            }
            for name, engine in engines.items()
            if isinstance(engine, ReplicaPool) and not engine.eligible_ids
        }
        if degraded:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unready",
                    "pools": degraded,
                },
            )
        # Local engines get a say too. "Constructed" is not "able to serve": a
        # KairyuBackend whose step loop or spawned TP ranks have died stays
        # constructed, so without this the endpoint reports ready while the node
        # cannot emit a token — which is exactly how a benchmark ran against a
        # dead 8-GPU deployment for 14 minutes before anyone noticed.
        failed = {
            name: status.detail for name, status in _engine_readiness().items() if not status.ready
        }
        if failed:
            return JSONResponse(status_code=503, content={"status": "unready", "engines": failed})
        return {"status": "ready"}

    @app.get("/backends")
    async def backends() -> dict:
        """Report the resolved attention backend, library versions, and the
        per-engine backend map (m13). Open endpoint (see middleware _OPEN_PATHS);
        disclosure level matches the existing public /readyz and /metrics.

        attention backend is a process-level decision (env override or probed hw
        profile — deterministic and shared), resolved once here with
        ``select_backend_decision(probe())`` rather than deep-walking each
        engine.

        Topology note: a pure gateway runs NO local attention — it forwards to
        replicas — so its own probe reports torch and its engines are all pools.
        To still surface the real kernel, each ReplicaPool engine asks ONE replica
        for its /backends and adopts that replica's attention backend (cached,
        best-effort, null on unreachable). `role` distinguishes the two cases."""
        from importlib.metadata import PackageNotFoundError, version

        try:
            profile = probe()
            kernel_tier = profile.kernel_tier
        except Exception:  # introspection must never 500
            profile, kernel_tier = None, "torch"
        try:
            decision = select_backend_decision(profile)
        except Exception as error:  # invalid env must not break introspection
            override = os.environ.get("KAIRYU_ATTENTION_BACKEND")
            decision = AttentionBackendDecision(
                requested=override or "auto",
                resolved="unavailable",
                source="env" if override else "hw_profile",
                components={},
                rationale=(f"attention selection is unavailable after {type(error).__name__}"),
                architecture={
                    "arch": profile.arch if profile is not None else "unknown",
                    "device_name": (profile.device_name if profile is not None else None),
                    "sm": profile.sm if profile is not None else None,
                    "kernel_tier": kernel_tier,
                },
            )
        configured_decision = decision

        # A real local engine retains the backend decision that was actually
        # constructed, including an ``auto`` dependency fallback. Prefer that
        # over re-evaluating env/profile state at inspection time. Engines with
        # no native model (mocks, remote gateways) have no such decision and
        # keep the configured process-level report above.
        actual_decisions = {
            name: actual
            for name, engine in engines.items()
            if isinstance(
                actual := getattr(engine, "attention_backend_decision", None),
                AttentionBackendDecision,
            )
        }
        if actual_decisions:
            identities = {
                attention_backend_identity(actual) for actual in actual_decisions.values()
            }
            if len(identities) == 1:
                decision = next(iter(actual_decisions.values()))
            else:
                decision = AttentionBackendDecision(
                    requested="mixed",
                    resolved="mixed",
                    source="engine",
                    components={},
                    rationale=(
                        "local engines constructed different attention "
                        "backend decisions; inspect the per-engine entries"
                    ),
                    architecture={
                        name: dict(actual.architecture) for name, actual in actual_decisions.items()
                    },
                )
        attention = decision.resolved

        def _kernel_tier_for(
            architecture: Mapping[str, object],
            fallback: str,
        ) -> str:
            direct = architecture.get("kernel_tier")
            if isinstance(direct, str) and direct:
                return direct
            nested = {
                tier
                for value in architecture.values()
                if isinstance(value, Mapping)
                and isinstance((tier := value.get("kernel_tier")), str)
                and tier
            }
            if len(nested) == 1:
                return nested.pop()
            return "mixed" if nested else fallback

        def _pkg_version(*names: str) -> str | None:
            # flashinfer ships under a few distribution names (flashinfer-python;
            # the AOT flashinfer-jit-cache) — try each so the version isn't null.
            for name in names:
                try:
                    return version(name)
                except PackageNotFoundError:
                    continue
            return None

        def _versions_for(
            components: Mapping[str, str],
        ) -> dict[str, str | None]:
            out: dict[str, str | None] = {"torch": _pkg_version("torch")}
            used = frozenset(components.values())
            if "flashinfer" in used:
                out["flashinfer"] = _pkg_version("flashinfer", "flashinfer-python")
            if "flashattention4" in used:
                out["flash-attn-4"] = _pkg_version("flash-attn-4")
            if "flashattention3" in used:
                out["flash_attn_3"] = _pkg_version("flash_attn_3")
            return out

        engine_list = []
        for name, engine in engines.items():
            label = _ENGINE_LABELS.get(type(engine).__name__, type(engine).__name__)
            actual = getattr(engine, "attention_backend_decision", None)
            engine_decision = (
                actual if isinstance(actual, AttentionBackendDecision) else configured_decision
            )
            entry: dict = {
                "model": name,
                "engine_backend": label,
                "attention_backend": (
                    engine_decision.resolved if label in _LOCAL_ATTENTION_BACKENDS else None
                ),
                "attention_components": (
                    dict(engine_decision.components) if label in _LOCAL_ATTENTION_BACKENDS else None
                ),
            }
            if label in _LOCAL_ATTENTION_BACKENDS:
                entry.update(
                    {
                        "requested_attention_backend": engine_decision.requested,
                        "selection_rationale": engine_decision.rationale,
                        "source": engine_decision.source,
                        "architecture": dict(engine_decision.architecture),
                        "decision_status": (
                            "actual"
                            if isinstance(actual, AttentionBackendDecision)
                            else "configured"
                        ),
                        "versions": _versions_for(engine_decision.components),
                    }
                )
            tensor_parallel_size = getattr(
                engine,
                "tensor_parallel_size",
                None,
            )
            if type(tensor_parallel_size) is int and tensor_parallel_size >= 1:
                entry["tensor_parallel_size"] = tensor_parallel_size
            if isinstance(engine, ReplicaPool):
                replica = await engine.probe_backends()
                if replica:  # adopt the replica's (real) attention backend
                    entry["attention_backend"] = replica.get("attention_backend")
                    replica_components = replica.get("attention_components")
                    entry["via_replica"] = {
                        "attention_backend": replica.get("attention_backend"),
                        "requested_attention_backend": replica.get("requested_attention_backend"),
                        "kernel_tier": replica.get("kernel_tier"),
                        "versions": replica.get("versions"),
                        "selection_rationale": replica.get("selection_rationale"),
                        "source": replica.get("source"),
                        "architecture": replica.get("architecture"),
                    }
                    if isinstance(replica_components, dict):
                        entry["attention_components"] = dict(replica_components)
                        entry["via_replica"]["attention_components"] = dict(replica_components)
                    replica_sizes = {
                        item.get("tensor_parallel_size")
                        for item in replica.get("engines", ())
                        if isinstance(item, dict) and type(item.get("tensor_parallel_size")) is int
                    }
                    if len(replica_sizes) == 1:
                        replica_size = replica_sizes.pop()
                        entry["tensor_parallel_size"] = replica_size
                        entry["via_replica"]["tensor_parallel_size"] = replica_size
            engine_list.append(entry)

        role = (
            "gateway"
            if engine_list and all(e["engine_backend"] == "replica-pool" for e in engine_list)
            else "engine-host"
        )

        return {
            "attention_backend": attention,
            "requested_attention_backend": decision.requested,
            "attention_components": dict(decision.components),
            "selection_rationale": decision.rationale,
            "source": decision.source,
            "architecture": dict(decision.architecture),
            "kernel_tier": _kernel_tier_for(
                decision.architecture,
                kernel_tier,
            ),
            "role": role,
            "versions": _versions_for(decision.components),
            "engines": engine_list,
        }

    if metrics is not None:

        @app.get("/metrics")
        async def metrics_endpoint() -> Response:
            body, content_type = metrics.render()
            return Response(content=body, media_type=content_type)
