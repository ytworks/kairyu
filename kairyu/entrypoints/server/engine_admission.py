"""Engine admission chain shared by the Chat and Messages engine paths.

A validated engine request passes these steps, in order, before dispatch:

1. **SLO lease**, for the interactive scheduling class only; the batch class
   skips it. The lease sheds the request, defers it (demoted to the batch
   class at the lowest scheduler priority, and shed instead when the backend
   cannot isolate deferred work) or admits it (its priority capped just above
   the deferred band). A demoted request is revalidated before prepare.
2. **Backend prepare**, timed as the ``backend_prepare`` phase.
3. **Admission upper bound** of the prepared request.
4. **Tenant token reservation** of that bound.
5. **Metrics**: the ``admission`` phase (bound plus reservation) and the
   request's scheduling class.

The chain never builds a response body: each surface renders the failures in
its own dialect through ``AdmissionErrors``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol

from kairyu.engine.backend import (
    UpstreamClientError,
    backend_admission_upper_bound_async,
    backend_supports_slo_defer,
    prepare_backend_request,
    validate_backend_request_before_prepare,
)
from kairyu.entrypoints.server.chat_errors import (
    ChatRequestError,
    chat_error_from_upstream_client_error,
)
from kairyu.entrypoints.server.middleware import _SLO_ADMISSION_LEASE_STATE_KEY

if TYPE_CHECKING:
    from fastapi import Request, Response

    from kairyu.engine.backend import AdmissionUpperBound
    from kairyu.entrypoints.server.chat_service import ValidatedChatRequest
    from kairyu.entrypoints.server.metrics import ServerMetrics
    from kairyu.entrypoints.server.slo import AdmissionController, AdmissionLease

_LOWEST_SCHEDULER_PRIORITY = 2**63 - 1
_SLO_INTERACTIVE_PRIORITY_CEILING = _LOWEST_SCHEDULER_PRIORITY - 1
_NS_PER_S = 1_000_000_000

# The preplacement phase a backend failure happened in.
AdmissionStage = Literal["backend_prepare", "admission"]


@dataclass(frozen=True)
class TenantRefusal:
    """A tenant token reservation that did not fit."""

    tenant: str
    reason: str


class AdmissionErrors(Protocol):
    """One surface's rendering of the admission chain's failures."""

    def slo_shed(self) -> Response:
        """The SLO lease shed the request, or deferred it to a backend that cannot defer."""

    def invalid(self, message: str) -> Response:
        """The demoted request or its admission bound is invalid."""

    def rejected(self, error: ValueError | ChatRequestError) -> Response:
        """Prepare rejected the request; an upstream 4xx arrives as ``ChatRequestError``."""

    def upstream_failed(self, stage: AdmissionStage, error: RuntimeError) -> Response:
        """The backend failed; called inside the ``except`` block, so it may log."""

    def tenant_limited(self, refusal: TenantRefusal) -> Response:
        """The tenant reservation did not fit."""


@dataclass(frozen=True)
class AdmissionSurface:
    """What differs between the surfaces that share the chain."""

    endpoint: str  # preplacement metrics label
    errors: AdmissionErrors
    # Chat-family endpoints time the admission phase of a refused reservation
    # too; Messages never has. Kept per surface so no metric changes.
    record_refused_admission_phase: bool = False


@dataclass(frozen=True)
class AdmittedEngineRequest:
    """The request to dispatch (possibly demoted by the SLO lease) and its lease."""

    validated: ValidatedChatRequest
    slo_lease: AdmissionLease | None


async def admit_engine_request(
    http_request: Request,
    validated: ValidatedChatRequest,
    *,
    surface: AdmissionSurface,
    scheduling_class: str,
) -> AdmittedEngineRequest | Response:
    """Run the admission chain; return the admitted request or the surface's error."""

    state = http_request.app.state
    metrics = getattr(state, "metrics", None)
    slo_admission = getattr(state, "slo_admission", None)
    admitted = AdmittedEngineRequest(validated=validated, slo_lease=None)
    if slo_admission is not None and scheduling_class == "interactive":
        leased = _begin_slo_lease(http_request, validated, slo_admission, surface.errors)
        if not isinstance(leased, AdmittedEngineRequest):
            return leased
        admitted = leased
    validated, slo_lease = admitted.validated, admitted.slo_lease
    failure = await _prepare(validated, surface, metrics)
    if failure is not None:
        return failure
    if (
        slo_lease is not None
        and slo_lease.decision.action == "defer"
        and not backend_supports_slo_defer(validated.engine, validated.generation_request)
    ):
        return _shed_deferred(http_request, slo_lease, surface.errors)
    failure = await _bound_and_reserve(http_request, validated, surface, metrics)
    if failure is not None:
        return failure
    return admitted


def reserve_tenant_tokens(
    http_request: Request,
    bound: AdmissionUpperBound,
) -> TenantRefusal | None:
    """Reserve ``bound`` against the request's tenant admission, if it has one."""

    admission = getattr(http_request.state, "tenant_admission", None)
    if admission is None:
        return None
    tenant = getattr(http_request.state, "tenant", None) or "default"
    admitted = admission.reserve_tokens(
        bound.tokens,
        refundable_on_exact_usage=bound.refundable_on_exact_usage,
    )
    metrics = getattr(http_request.app.state, "metrics", None)
    if metrics is not None:
        metrics.record_tenant_admission(
            tenant,
            source="http",
            admitted=admitted,
            reason=admission.reason,
        )
    if admitted:
        http_request.state.tenant_metric_admitted = True
        return None
    return TenantRefusal(tenant=tenant, reason=admission.reason)


def _begin_slo_lease(
    http_request: Request,
    validated: ValidatedChatRequest,
    slo_admission: AdmissionController,
    errors: AdmissionErrors,
) -> AdmittedEngineRequest | Response:
    ingress_ns = getattr(http_request.state, "placement_started_ns", None)
    elapsed_s = (
        max(0, time.perf_counter_ns() - ingress_ns) / _NS_PER_S if type(ingress_ns) is int else 0.0
    )
    lease = slo_admission.begin(elapsed_s=elapsed_s)
    if lease.decision.action == "shed":
        return errors.slo_shed()
    http_request.scope.setdefault("state", {})[_SLO_ADMISSION_LEASE_STATE_KEY] = lease
    generation_request = validated.generation_request
    if lease.decision.action == "defer":
        admission_request = replace(
            generation_request,
            priority=_LOWEST_SCHEDULER_PRIORITY,
            scheduling_class="batch",
        )
        if not backend_supports_slo_defer(validated.engine, admission_request):
            return _shed_deferred(http_request, lease, errors)
    elif generation_request.priority > _SLO_INTERACTIVE_PRIORITY_CEILING:
        admission_request = replace(
            generation_request,
            priority=_SLO_INTERACTIVE_PRIORITY_CEILING,
        )
    else:
        admission_request = generation_request
    if admission_request is not generation_request:
        try:
            validate_backend_request_before_prepare(validated.engine, admission_request)
        except ValueError as error:
            return errors.invalid(str(error))
        validated = replace(validated, generation_request=admission_request)
    return AdmittedEngineRequest(validated=validated, slo_lease=lease)


def _shed_deferred(
    http_request: Request,
    lease: AdmissionLease,
    errors: AdmissionErrors,
) -> Response:
    """Release a deferred lease the backend cannot honour and shed the request."""

    state = http_request.scope.setdefault("state", {})
    if state.get(_SLO_ADMISSION_LEASE_STATE_KEY) is lease:
        state.pop(_SLO_ADMISSION_LEASE_STATE_KEY)
    if lease.active:
        lease.completed()
    return errors.slo_shed()


async def _prepare(
    validated: ValidatedChatRequest,
    surface: AdmissionSurface,
    metrics: ServerMetrics | None,
) -> Response | None:
    started_ns = time.perf_counter_ns()
    try:
        await prepare_backend_request(validated.engine, validated.generation_request)
    except UpstreamClientError as error:
        return surface.errors.rejected(chat_error_from_upstream_client_error(error))
    except ValueError as error:
        return surface.errors.rejected(error)
    except RuntimeError as error:
        return surface.errors.upstream_failed("backend_prepare", error)
    finally:
        if metrics is not None:
            metrics.record_preplacement_phase(
                surface.endpoint,
                "backend_prepare",
                max(0, time.perf_counter_ns() - started_ns),
            )
    return None


async def _bound_and_reserve(
    http_request: Request,
    validated: ValidatedChatRequest,
    surface: AdmissionSurface,
    metrics: ServerMetrics | None,
) -> Response | None:
    started_ns = time.perf_counter_ns()
    try:
        bound = await backend_admission_upper_bound_async(
            validated.engine,
            validated.generation_request,
        )
    except ValueError as error:
        return surface.errors.invalid(str(error))
    except RuntimeError as error:
        return surface.errors.upstream_failed("admission", error)
    admission_ns = max(0, time.perf_counter_ns() - started_ns)
    reserve_started_ns = time.perf_counter_ns()
    refusal = reserve_tenant_tokens(http_request, bound)
    admission_ns += max(0, time.perf_counter_ns() - reserve_started_ns)
    if metrics is not None and (refusal is None or surface.record_refused_admission_phase):
        metrics.record_preplacement_phase(surface.endpoint, "admission", admission_ns)
    if refusal is not None:
        return surface.errors.tenant_limited(refusal)
    if metrics is not None:
        metrics.record_priority(validated.generation_request.scheduling_class, source="http")
    return None
