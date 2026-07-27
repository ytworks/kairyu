"""Orchestrator facade: route a query, then dispatch to an engine or the Conductor."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import asdict, dataclass

from kairyu.engine.backend import (
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    shutdown_all,
)
from kairyu.orchestration.budget import Budget, BudgetState
from kairyu.orchestration.conductor import (
    Conductor,
    ConductorStreamError,
    CostModel,
    RoleSpec,
    zero_cost,
)
from kairyu.orchestration.request import (
    OrchestrationRequest,
    default_orchestration_request,
)
from kairyu.orchestration.router import RouteDecision, Router, RuleRouter
from kairyu.orchestration.trace import (
    StructuredTrace,
    TraceBudget,
    TraceError,
    TraceEvent,
    TraceTiming,
    TraceUsage,
    WorkerTraceIdentity,
    utc_now_iso,
)
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams

_KEEPALIVE_INTERVAL_S = 15.0  # SSE keep-alive cadence for long multi-stage runs (M8)


_DEFAULT_ROLES = (
    RoleSpec(
        name="planner",
        worker="tier2",
        role_type="planner",
        prompt="[planner] Break the task into a short actionable plan.\nTask: {query}",
    ),
    RoleSpec(
        name="worker",
        worker="tier1",
        prompt="[worker] Execute this plan and produce the answer.\nPlan: {planner}\nTask: {query}",
        depends_on=("planner",),
    ),
    RoleSpec(
        name="verifier",
        worker="tier2",
        role_type="verifier",
        verifies="worker",
        prompt=(
            "[verifier] Check the answer for the task. Reply PASS or FAIL with reasons.\n"
            "Task: {query}\nAnswer: {worker}"
        ),
        depends_on=("worker",),
    ),
    RoleSpec(
        name="synthesizer",
        worker="tier2",
        role_type="synthesizer",
        prompt=(
            "[synthesizer] Produce the final polished answer.\n"
            "Task: {query}\nDraft: {worker}\nVerifier notes: {verifier}"
        ),
        depends_on=("worker", "verifier"),
    ),
)


@dataclass(frozen=True)
class OrchestratorResult:
    text: str
    route: RouteDecision
    trace: tuple[str, ...]
    completions: tuple[CompletionOutput, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    structured_trace: StructuredTrace | None = None


@dataclass(frozen=True)
class OrchestratorEvent:
    """Streaming event (m11 D1): status keep-alive, token delta, or final."""

    kind: str  # "status" | "delta" | "error" | "result"
    text: str = ""
    completions: tuple[CompletionOutput, ...] = ()
    result: OrchestratorResult | None = None
    error_type: str | None = None


class OrchestratorExecutionError(RuntimeError):
    """Backend failure plus the partial accounting/trace safe to return."""

    def __init__(
        self,
        cause: BaseException,
        result: OrchestratorResult,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.result = result


class _DirectExecutionError(RuntimeError):
    def __init__(self, cause: BaseException, event: TraceEvent) -> None:
        super().__init__(type(cause).__name__)
        self.cause = cause
        self.event = event


class PreviewNotSupportedError(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineDescriptor:
    backend_type: str
    model: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"backend_type": self.backend_type, "model": self.model}


class Orchestrator:
    def __init__(
        self,
        engines: Mapping[str, EngineBackend],
        router: Router | None = None,
        roles: tuple[RoleSpec, ...] | None = None,
        budget: Budget | None = None,
        shared_prefix: str = "",
        sampling_params: SamplingParams | None = None,
        cost_model: CostModel = zero_cost,
        moa_samples: int = 0,
        engine_descriptors: Mapping[str, EngineDescriptor] | None = None,
    ) -> None:
        if not engines:
            raise ValueError("Orchestrator requires at least one engine")
        self._engines = dict(engines)
        supplied_descriptors = dict(engine_descriptors or {})
        self._engine_descriptors = {
            name: supplied_descriptors.get(
                name,
                EngineDescriptor(backend_type=type(engine).__name__),
            )
            for name, engine in self._engines.items()
        }
        self._router = router or RuleRouter()
        self._roles = roles or _DEFAULT_ROLES
        self._budget = budget or Budget()
        self._shared_prefix = shared_prefix
        self._sampling_params = sampling_params or SamplingParams(max_tokens=1024)
        self._cost_model = cost_model
        # m11 A4: >0 routes multi_agent through MoA (the deep kairyu-auto-max tier)
        self._moa_samples = moa_samples

    def preview_route(self, prompt: str) -> RouteDecision:
        preview = getattr(self._router, "preview", None)
        if preview is None:
            raise PreviewNotSupportedError(
                f"router {type(self._router).__name__} does not support preview"
            )
        try:
            return preview(prompt)
        except NotImplementedError as error:
            raise PreviewNotSupportedError(str(error)) from error

    def _resolved_engine_descriptor(self, key: str) -> dict[str, object]:
        configured = key in self._engines
        effective = key if configured else next(iter(self._engines))
        return {
            "configured": configured,
            "engine": effective,
            "fallback": not configured,
        }

    def describe_routing(self) -> dict[str, object]:
        describe = getattr(self._router, "describe", None)
        router = (
            describe() if describe is not None else {"router_type": type(self._router).__name__}
        )
        role_workers = tuple(dict.fromkeys(role.worker for role in self._roles))
        if self._moa_samples > 0:
            multi_engines = [self._resolved_engine_descriptor(key) for key in ("tier1", "tier2")]
            multi_mode = "moa"
        else:
            multi_engines = [self._resolved_engine_descriptor(key) for key in role_workers]
            multi_mode = "roles"
        return {
            "router": router,
            "targets": ["tier1", "tier2", "multi_agent"],
            "configured_engines": {
                name: descriptor.as_dict() for name, descriptor in self._engine_descriptors.items()
            },
            "target_resolution": {
                "tier1": self._resolved_engine_descriptor("tier1"),
                "tier2": self._resolved_engine_descriptor("tier2"),
                "multi_agent": {
                    "mode": multi_mode,
                    "engines": multi_engines,
                },
            },
            "roles": [
                {
                    "name": role.name,
                    "worker": role.worker,
                    "role_type": role.role_type,
                    "depends_on": list(role.depends_on),
                    "verifies": role.verifies,
                }
                for role in self._roles
            ],
            "budget": asdict(self._budget),
            "moa_samples": self._moa_samples,
            "internal_max_tokens": self._sampling_params.max_tokens,
        }

    async def shutdown(self) -> None:
        await shutdown_all(self._engines.values(), "Orchestrator")

    def _resolve_engine_name(self, tier: str, notes: list[str]) -> str:
        if tier in self._engines:
            return tier
        fallback_name = next(iter(self._engines))
        notes.append(f"fallback: engine {tier!r} not configured, using {fallback_name!r}")
        return fallback_name

    def _resolve_engine(self, tier: str, notes: list[str]) -> EngineBackend:
        return self._engines[self._resolve_engine_name(tier, notes)]

    def _request(self, request: str | OrchestrationRequest) -> OrchestrationRequest:
        if isinstance(request, OrchestrationRequest):
            return request
        return default_orchestration_request(request, self._sampling_params)

    def _internal_sampling_params(
        self,
        call: OrchestrationRequest,
    ) -> SamplingParams:
        return call.internal_sampling_params(
            max_tokens_cap=self._sampling_params.max_tokens,
        )

    def _conductor_final_role(self) -> RoleSpec:
        units = [role for role in self._roles if role.role_type != "verifier"]
        dependents = {dependency for role in units for dependency in role.depends_on}
        terminal = [role for role in units if role.name not in dependents]
        synthesizers = [role for role in terminal if role.role_type == "synthesizer"]
        if not terminal:
            raise ValueError("orchestration requires at least one terminal role")
        return (synthesizers + terminal)[0]

    def _final_engine_keys(self, decision: RouteDecision | None) -> tuple[str, ...]:
        if decision is None:
            keys = {"tier1", "tier2", self._conductor_final_role().worker}
        elif decision.target == "multi_agent":
            keys = {"tier2" if self._moa_samples > 0 else self._conductor_final_role().worker}
        else:
            keys = {decision.target}
        fallback = next(iter(self._engines))
        return tuple(key if key in self._engines else fallback for key in sorted(keys))

    def _internal_engine_keys(
        self,
        decision: RouteDecision | None,
    ) -> tuple[str, ...]:
        if decision is not None and decision.target != "multi_agent":
            return ()
        if self._moa_samples > 0:
            keys = {"tier1"}
        else:
            final_role = self._conductor_final_role()
            keys = {
                role.worker
                for role in self._roles
                if role.name != final_role.name
            }
        fallback = next(iter(self._engines))
        return tuple(key if key in self._engines else fallback for key in sorted(keys))

    def _validate_call(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> None:
        if call.sampling_params.n <= 1:
            return
        final_role = self._conductor_final_role()
        if (
            (decision is None or decision.target == "multi_agent")
            and self._moa_samples == 0
            and any(
                role.role_type == "verifier" and role.verifies == final_role.name
                for role in self._roles
            )
        ):
            raise ValueError(
                "n > 1 is not supported when the final orchestration role has "
                "a post-generation verifier"
            )
        unsupported = [
            key
            for key in self._final_engine_keys(decision)
            if getattr(self._engines[key], "supports_n", True) is False
        ]
        if unsupported:
            raise ValueError(
                "n > 1 is not supported by final orchestration engine(s): " + ", ".join(unsupported)
            )

    def _validate_final_intent(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> None:
        failures: list[str] = []
        for key in self._final_engine_keys(decision):
            validate = getattr(self._engines[key], "validate_request", None)
            if validate is None:
                continue
            request = GenerationRequest(
                request_id=f"preflight-{key}",
                prompt=f"{self._shared_prefix}{call.prompt}",
                sampling_params=call.sampling_params,
                tools=call.tools,
                tool_choice=call.tool_choice,
                tools_in_prompt=call.tools_in_prompt,
            )
            try:
                validate(request)
            except ValueError as error:
                failures.append(f"{key}: {error}")
        if failures:
            raise ValueError(
                "final orchestration intent is unsupported: " + "; ".join(failures)
            )

    def _validate_internal_intent(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> None:
        failures: list[str] = []
        sampling_params = self._internal_sampling_params(call)
        for key in self._internal_engine_keys(decision):
            validate = getattr(self._engines[key], "validate_request", None)
            if validate is None:
                continue
            request = GenerationRequest(
                request_id=f"preflight-internal-{key}",
                prompt=f"{self._shared_prefix}{call.prompt}",
                sampling_params=sampling_params,
            )
            try:
                validate(request)
            except ValueError as error:
                failures.append(f"{key}: {error}")
        if failures:
            raise ValueError(
                "internal orchestration intent is unsupported: " + "; ".join(failures)
            )

    def validate_request(self, request: str | OrchestrationRequest) -> None:
        """Preflight capability checks without dispatching generation."""

        call = self._request(request)
        preview = getattr(self._router, "preview", None)
        if preview is None:
            decision = None
        else:
            try:
                decision = preview(call.prompt)
            except NotImplementedError:
                decision = None
        self._validate_call(call, decision)
        self._validate_final_intent(call, decision)
        self._validate_internal_intent(call, decision)

    def _conductor_workers(self, notes: list[str]) -> dict[str, EngineBackend]:
        needed = {role.worker for role in self._roles}
        return {name: self._resolve_engine(name, notes) for name in needed}

    def _conductor_worker_trace(self) -> dict[str, WorkerTraceIdentity]:
        fallback_name = next(iter(self._engines))
        needed = {role.worker for role in self._roles}
        identities = {}
        for worker in needed:
            engine = worker if worker in self._engines else fallback_name
            descriptor = self._engine_descriptors[engine]
            identities[worker] = WorkerTraceIdentity(
                engine=engine,
                model=descriptor.model,
            )
        return identities

    def _route_trace_event(
        self,
        decision: RouteDecision,
        *,
        started_at: str,
        completed_at: str,
    ) -> TraceEvent:
        return TraceEvent(
            node="router",
            kind="route",
            detail=f"{decision.target} ({decision.reason})",
            operation="routing",
            status="success",
            role="router",
            timing=TraceTiming(
                started_at=started_at,
                completed_at=completed_at,
            ),
            metadata={
                "target": decision.target,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
        )

    async def _run_direct(
        self, call: OrchestrationRequest, tier: str, notes: list[str]
    ) -> tuple[GenerationResult, tuple[int, int, int], TraceEvent]:
        queued_at = utc_now_iso()
        engine_name = self._resolve_engine_name(tier, notes)
        engine = self._engines[engine_name]
        descriptor = self._engine_descriptors[engine_name]
        request = GenerationRequest(
            request_id=f"direct-{uuid.uuid4().hex[:12]}",
            prompt=f"{self._shared_prefix}{call.prompt}",
            sampling_params=call.sampling_params,
            # no random per-request session (M2): a fresh uuid forces uniform HRW
            # placement and defeats prefix + least-outstanding routing. With no
            # hint the pool routes by shared-prefix overlap and load instead.
            cache_hint=None,
            tools=call.tools,
            tool_choice=call.tool_choice,
            tools_in_prompt=call.tools_in_prompt,
        )
        started_at = utc_now_iso()
        try:
            result = await engine.generate(request)
        except Exception as error:
            raise _DirectExecutionError(
                error,
                TraceEvent(
                    node=tier,
                    kind="failed",
                    operation="generation",
                    status="failed",
                    role="direct",
                    worker=tier,
                    engine=engine_name,
                    model=descriptor.model,
                    timing=TraceTiming(
                        queued_at=queued_at,
                        started_at=started_at,
                        completed_at=utc_now_iso(),
                    ),
                    error=TraceError(type=type(error).__name__),
                ),
            ) from error
        usage = (
            (
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                result.usage.cached_tokens,
            )
            if result.usage is not None
            else (0, 0, 0)
        )
        trace_usage = (
            TraceUsage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                cached_tokens=result.usage.cached_tokens,
            )
            if result.usage is not None
            else None
        )
        return (
            result,
            usage,
            TraceEvent(
                node=tier,
                kind="generated",
                operation="generation",
                status="success",
                role="direct",
                worker=tier,
                engine=engine_name,
                model=descriptor.model,
                timing=TraceTiming(
                    queued_at=queued_at,
                    started_at=started_at,
                    completed_at=utc_now_iso(),
                ),
                usage=trace_usage,
            ),
        )

    async def run(
        self,
        request: str | OrchestrationRequest,
    ) -> OrchestratorResult:
        call = self._request(request)
        query = call.prompt
        request_id = f"orch-{uuid.uuid4().hex[:16]}"
        trace_started_at = utc_now_iso()
        route_started_at = utc_now_iso()
        decision = self._router.route(query)
        self._validate_call(call, decision)
        route_completed_at = utc_now_iso()
        notes: list[str] = [f"route: {decision.target} ({decision.reason})"]
        trace_events = [
            self._route_trace_event(
                decision,
                started_at=route_started_at,
                completed_at=route_completed_at,
            )
        ]

        def result_with_trace(
            *,
            text: str,
            completions: tuple[CompletionOutput, ...] = (),
            prompt_tokens: int = 0,
            completion_tokens: int = 0,
            cached_tokens: int = 0,
        ) -> OrchestratorResult:
            return OrchestratorResult(
                text=text,
                route=decision,
                trace=tuple(notes),
                completions=completions,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                structured_trace=StructuredTrace(
                    request_id=request_id,
                    started_at=trace_started_at,
                    completed_at=utc_now_iso(),
                    events=tuple(trace_events),
                ),
            )

        if decision.target == "multi_agent":
            if self._moa_samples > 0:  # m11 A4: the deep tier's MoA route
                from kairyu.orchestration.moa import MoAExecutionError, run_moa

                moa_queued_at = utc_now_iso()
                budget_before = BudgetState(budget=self._budget)
                moa_steps = self._moa_samples + 1
                reservation = budget_before.try_reserve(
                    steps=moa_steps,
                    unknown_cost=True,
                )
                if reservation is None:
                    notes.append("moa: skipped:budget")
                    trace_events.append(
                        TraceEvent(
                            node="moa",
                            kind="skipped:budget",
                            operation="synthesis",
                            status="skipped",
                            role="moa",
                            budget=TraceBudget.between(
                                budget_before,
                                budget_before,
                            ),
                            metadata={"reason": "budget"},
                        )
                    )
                    return result_with_trace(text="")
                proposal_engine_name = self._resolve_engine_name("tier1", notes)
                synthesizer_engine_name = self._resolve_engine_name("tier2", notes)
                proposal_descriptor = self._engine_descriptors[proposal_engine_name]
                synthesizer_descriptor = self._engine_descriptors[synthesizer_engine_name]
                moa_started_at = utc_now_iso()
                moa = None
                try:
                    moa = await run_moa(
                        self._engines[proposal_engine_name],
                        query,
                        n_samples=self._moa_samples,
                        synthesizer=self._engines[synthesizer_engine_name],
                        sampling_params=self._internal_sampling_params(call),
                        final_sampling_params=call.sampling_params,
                        final_tools=call.tools,
                        final_tool_choice=call.tool_choice,
                        final_tools_in_prompt=call.tools_in_prompt,
                        shared_prefix=self._shared_prefix,
                    )
                    # M3: the deep MoA tier was invisible to the cost model / budget.
                    # Reconcile proposals + synthesis with the actual result cost;
                    # one admitted operation may visibly cross a result-priced cap.
                    moa_cost = self._cost_model(
                        GenerationRequest(
                            request_id="moa",
                            prompt=query,
                            sampling_params=call.sampling_params,
                            tools=call.tools,
                            tool_choice=call.tool_choice,
                            tools_in_prompt=call.tools_in_prompt,
                        ),
                        GenerationResult(
                            request_id="moa",
                            prompt=query,
                            completions=moa.completions
                            or (
                                CompletionOutput(
                                    index=0,
                                    text=moa.final_text,
                                    token_ids=(),
                                ),
                            ),
                            usage=GenerationUsage(
                                prompt_tokens=moa.usage[0],
                                completion_tokens=moa.usage[1],
                                cached_tokens=moa.cached_tokens,
                            ),
                        ),
                    )
                    budget_state = reservation.reconcile_success(
                        steps=moa_steps,
                        cost=moa_cost,
                        unknown_cost=True,
                    )
                except Exception as error:
                    released = reservation.release(
                        steps=moa_steps,
                        unknown_cost=True,
                    )
                    if not isinstance(error, MoAExecutionError):
                        raise
                    cause = error.cause
                    usage = error.usage
                    cached_tokens = error.cached_tokens
                    final_text = error.final_text
                    stage = error.stage
                    trace_events.append(
                        TraceEvent(
                            node="moa",
                            kind="failed",
                            operation="synthesis",
                            status="failed",
                            role="moa",
                            worker="tier1,tier2",
                            engine=",".join(
                                dict.fromkeys(
                                    (
                                        proposal_engine_name,
                                        synthesizer_engine_name,
                                    )
                                )
                            ),
                            model=",".join(
                                dict.fromkeys(
                                    model
                                    for model in (
                                        proposal_descriptor.model,
                                        synthesizer_descriptor.model,
                                    )
                                    if model is not None
                                )
                            )
                            or None,
                            timing=TraceTiming(
                                queued_at=moa_queued_at,
                                started_at=moa_started_at,
                                completed_at=utc_now_iso(),
                            ),
                            usage=TraceUsage(
                                prompt_tokens=usage[0],
                                completion_tokens=usage[1],
                                cached_tokens=cached_tokens,
                            ),
                            budget=TraceBudget.between(
                                budget_before,
                                released,
                            ),
                            metadata={"stage": stage},
                            error=TraceError(type=type(cause).__name__),
                        )
                    )
                    raise OrchestratorExecutionError(
                        cause,
                        result_with_trace(
                            text=final_text,
                            prompt_tokens=usage[0],
                            completion_tokens=usage[1],
                            cached_tokens=cached_tokens,
                        ),
                    ) from cause
                except BaseException:
                    reservation.release(steps=moa_steps, unknown_cost=True)
                    raise
                notes.append(
                    f"moa: {len(moa.proposals)} proposals synthesized (cost={moa_cost:.4f})"
                )
                if budget_state.is_exhausted:
                    notes.append("moa: budget exceeded")
                resolved_engines = tuple(
                    dict.fromkeys((proposal_engine_name, synthesizer_engine_name))
                )
                resolved_models = tuple(
                    dict.fromkeys(
                        model
                        for model in (
                            proposal_descriptor.model,
                            synthesizer_descriptor.model,
                        )
                        if model is not None
                    )
                )
                trace_events.append(
                    TraceEvent(
                        node="moa",
                        kind="synthesized",
                        operation="synthesis",
                        status="success",
                        role="moa",
                        worker="tier1,tier2",
                        engine=",".join(resolved_engines),
                        model=",".join(resolved_models) or None,
                        timing=TraceTiming(
                            queued_at=moa_queued_at,
                            started_at=moa_started_at,
                            completed_at=utc_now_iso(),
                        ),
                        usage=TraceUsage(
                            prompt_tokens=moa.usage[0],
                            completion_tokens=moa.usage[1],
                            cached_tokens=moa.cached_tokens,
                        ),
                        budget=TraceBudget.between(
                            budget_before,
                            budget_state,
                            steps_consumed=moa_steps,
                            cost_consumed_usd=moa_cost,
                        ),
                        metadata={
                            "proposals": len(moa.proposals),
                            "cost_usd": moa_cost,
                            "budget_exhausted": budget_state.is_exhausted,
                            "proposal_engine": proposal_engine_name,
                            "proposal_model": proposal_descriptor.model,
                            "synthesizer_engine": synthesizer_engine_name,
                            "synthesizer_model": synthesizer_descriptor.model,
                        },
                    )
                )
                return result_with_trace(
                    text=moa.final_text,
                    completions=moa.completions
                    or (
                        CompletionOutput(
                            index=0,
                            text=moa.final_text,
                            token_ids=(),
                            finish_reason="stop",
                        ),
                    ),
                    prompt_tokens=moa.usage[0],
                    completion_tokens=moa.usage[1],
                    cached_tokens=moa.cached_tokens,
                )
            conductor = Conductor(
                roles=self._roles,
                workers=self._conductor_workers(notes),
                shared_prefix=self._shared_prefix,
                sampling_params=self._internal_sampling_params(call),
                final_sampling_params=call.sampling_params,
                final_tools=call.tools,
                final_tool_choice=call.tool_choice,
                final_tools_in_prompt=call.tools_in_prompt,
                cost_model=self._cost_model,
                worker_trace=self._conductor_worker_trace(),
            )
            result = await conductor.run(query, budget=self._budget)
            notes.extend(f"{event.node}: {event.kind} {event.detail}" for event in result.trace)
            trace_events.extend(result.trace)
            return result_with_trace(
                text=result.final_text,
                completions=result.completions,
                prompt_tokens=result.usage[0],
                completion_tokens=result.usage[1],
                cached_tokens=result.cached_tokens,
            )
        try:
            direct_result, usage, direct_event = await self._run_direct(
                call,
                decision.target,
                notes,
            )
        except _DirectExecutionError as error:
            trace_events.append(error.event)
            raise OrchestratorExecutionError(
                error.cause,
                result_with_trace(text=""),
            ) from error.cause
        trace_events.append(direct_event)
        return result_with_trace(
            text=direct_result.text,
            completions=direct_result.completions,
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
            cached_tokens=usage[2],
        )

    async def run_chat(
        self,
        request: str | OrchestrationRequest,
        stream: bool = False,
        usage_observer: Callable[[GenerationUsage], None] | None = None,
    ):
        """The m11 D1 surface: pre-rendered prompt in (A4), events out.

        Non-stream: returns OrchestratorResult (same as run()). Stream:
        async-yields OrchestratorEvent — status keep-alives while pre-final
        stages run, token deltas pulled directly from every route's FINAL
        worker/synthesizer, then one result event.
        """
        if not stream:
            return await self.run(request)
        return self._run_chat_stream(
            self._request(request),
            usage_observer=usage_observer,
        )

    async def _with_initial_keepalives(self, stream) -> AsyncIterator[object | None]:
        """Emit timer sentinels until the first final-stage event is available.

        Only the first ``anext`` is task-backed so long pre-final work can be
        multiplexed with keep-alives.  Once the final backend produces its
        first event, all remaining deltas are pulled through in the caller's
        task without a per-token task or queue.
        """

        iterator = stream.__aiter__()
        first = asyncio.ensure_future(anext(iterator))
        try:
            while not first.done():
                done, _ = await asyncio.wait(
                    {first},
                    timeout=_KEEPALIVE_INTERVAL_S,
                )
                if not done:
                    yield None
            try:
                yield first.result()
            except StopAsyncIteration:
                return
            async for event in iterator:
                yield event
        finally:
            if not first.done():
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def _run_chat_stream(
        self,
        call: OrchestrationRequest,
        *,
        usage_observer: Callable[[GenerationUsage], None] | None,
    ):
        prompt = call.prompt
        request_id = f"orch-{uuid.uuid4().hex[:16]}"
        trace_started_at = utc_now_iso()
        route_started_at = utc_now_iso()
        decision = self._router.route(prompt)
        self._validate_call(call, decision)
        route_completed_at = utc_now_iso()
        notes = [f"route: {decision.target} ({decision.reason})"]
        trace_events = [
            self._route_trace_event(
                decision,
                started_at=route_started_at,
                completed_at=route_completed_at,
            )
        ]

        def result_with_trace(
            *,
            text: str,
            completions: tuple[CompletionOutput, ...] = (),
            prompt_tokens: int = 0,
            completion_tokens: int = 0,
            cached_tokens: int = 0,
        ) -> OrchestratorResult:
            return OrchestratorResult(
                text=text,
                route=decision,
                trace=tuple(notes),
                completions=completions,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                structured_trace=StructuredTrace(
                    request_id=request_id,
                    started_at=trace_started_at,
                    completed_at=utc_now_iso(),
                    events=tuple(trace_events),
                ),
            )

        if decision.target != "multi_agent":
            queued_at = utc_now_iso()
            engine_name = self._resolve_engine_name(decision.target, notes)
            engine = self._engines[engine_name]
            descriptor = self._engine_descriptors[engine_name]
            request = GenerationRequest(
                request_id=f"direct-{uuid.uuid4().hex[:12]}",
                prompt=f"{self._shared_prefix}{prompt}",
                sampling_params=call.sampling_params,
                cache_hint=None,  # M2: no random session — route by prefix + load
                tools=call.tools,
                tool_choice=call.tool_choice,
                tools_in_prompt=call.tools_in_prompt,
            )
            emitted = 0
            last = None
            latest_usage = None
            started_at = utc_now_iso()
            first_token_at = None
            previous_text = ""
            try:
                async for partial in engine.stream(request):
                    last = partial
                    if partial.usage is not None:
                        latest_usage = partial.usage
                        if usage_observer is not None:
                            usage_observer(latest_usage)
                    text = partial.text
                    if not text.startswith(previous_text):
                        raise RuntimeError("direct stream must emit cumulative, prefix-stable text")
                    previous_text = text
                    if first_token_at is None and partial.completions:
                        first_token_at = utc_now_iso()
                    yield OrchestratorEvent(
                        kind="delta",
                        text=text[emitted:],
                        completions=partial.completions,
                    )
                    emitted = len(text)
                if last is None:
                    raise RuntimeError("direct stream produced no result")
            except Exception as error:
                trace_usage = (
                    TraceUsage(
                        prompt_tokens=latest_usage.prompt_tokens,
                        completion_tokens=latest_usage.completion_tokens,
                        cached_tokens=latest_usage.cached_tokens,
                    )
                    if latest_usage is not None
                    else None
                )
                trace_events.append(
                    TraceEvent(
                        node=decision.target,
                        kind="failed",
                        operation="generation",
                        status="failed",
                        role="direct",
                        worker=decision.target,
                        engine=engine_name,
                        model=descriptor.model,
                        timing=TraceTiming(
                            queued_at=queued_at,
                            started_at=started_at,
                            first_token_at=first_token_at,
                            completed_at=utc_now_iso(),
                        ),
                        usage=trace_usage,
                        metadata={"streamed": True},
                        error=TraceError(type=type(error).__name__),
                    )
                )
                usage = latest_usage or GenerationUsage()
                partial_result = result_with_trace(
                    text=previous_text,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cached_tokens=usage.cached_tokens,
                )
                yield OrchestratorEvent(
                    kind="error",
                    result=partial_result,
                    error_type=type(error).__name__,
                )
                return
            trace_usage = (
                TraceUsage(
                    prompt_tokens=latest_usage.prompt_tokens,
                    completion_tokens=latest_usage.completion_tokens,
                    cached_tokens=latest_usage.cached_tokens,
                )
                if latest_usage is not None
                else None
            )
            trace_events.append(
                TraceEvent(
                    node=decision.target,
                    kind="generated",
                    operation="generation",
                    status="success",
                    role="direct",
                    worker=decision.target,
                    engine=engine_name,
                    model=descriptor.model,
                    timing=TraceTiming(
                        queued_at=queued_at,
                        started_at=started_at,
                        first_token_at=first_token_at,
                        completed_at=utc_now_iso(),
                    ),
                    usage=trace_usage,
                    metadata={"streamed": True},
                )
            )
            usage = (
                (
                    latest_usage.prompt_tokens,
                    latest_usage.completion_tokens,
                    latest_usage.cached_tokens,
                )
                if latest_usage is not None
                else (0, 0, 0)
            )
            yield OrchestratorEvent(
                kind="result",
                result=result_with_trace(
                    text=last.text if last is not None else "",
                    completions=last.completions if last is not None else (),
                    prompt_tokens=usage[0],
                    completion_tokens=usage[1],
                    cached_tokens=usage[2],
                ),
            )
            return

        # Multi-stage routes emit comments while the pre-final DAG/proposals and
        # first final token are pending.  Final deltas then pull through from
        # Conductor/MoA without a queue bridge.
        yield OrchestratorEvent(kind="status", text=f"routing: {decision.target}")
        if self._moa_samples > 0:
            from kairyu.orchestration.moa import MoAExecutionError, stream_moa

            moa_queued_at = utc_now_iso()
            budget_before = BudgetState(budget=self._budget)
            moa_steps = self._moa_samples + 1
            reservation = budget_before.try_reserve(
                steps=moa_steps,
                unknown_cost=True,
            )
            if reservation is None:
                notes.append("moa: skipped:budget")
                trace_events.append(
                    TraceEvent(
                        node="moa",
                        kind="skipped:budget",
                        operation="synthesis",
                        status="skipped",
                        role="moa",
                        budget=TraceBudget.between(
                            budget_before,
                            budget_before,
                        ),
                        metadata={"reason": "budget"},
                    )
                )
                yield OrchestratorEvent(
                    kind="result",
                    result=result_with_trace(text=""),
                )
                return

            proposal_engine_name = self._resolve_engine_name("tier1", notes)
            synthesizer_engine_name = self._resolve_engine_name("tier2", notes)
            proposal_descriptor = self._engine_descriptors[proposal_engine_name]
            synthesizer_descriptor = self._engine_descriptors[synthesizer_engine_name]
            moa_started_at = utc_now_iso()
            moa_result = None
            latest_usage = GenerationUsage()
            first_token_at = None

            def observe_moa_usage(usage: GenerationUsage) -> None:
                nonlocal latest_usage
                latest_usage = usage
                if usage_observer is not None:
                    usage_observer(usage)

            try:
                moa_stream = stream_moa(
                    self._engines[proposal_engine_name],
                    prompt,
                    n_samples=self._moa_samples,
                    synthesizer=self._engines[synthesizer_engine_name],
                    sampling_params=self._internal_sampling_params(call),
                    final_sampling_params=call.sampling_params,
                    final_tools=call.tools,
                    final_tool_choice=call.tool_choice,
                    final_tools_in_prompt=call.tools_in_prompt,
                    shared_prefix=self._shared_prefix,
                    usage_observer=observe_moa_usage,
                )
                async for event in self._with_initial_keepalives(moa_stream):
                    if event is None:
                        yield OrchestratorEvent(kind="status", text="working")
                    elif event.kind == "delta":
                        if first_token_at is None:
                            first_token_at = utc_now_iso()
                        yield OrchestratorEvent(
                            kind="delta",
                            text=event.text,
                            completions=event.completions,
                        )
                    else:
                        moa_result = event.result
                if moa_result is None:
                    raise RuntimeError("MoA stream did not produce a final result")
                moa_cost = self._cost_model(
                    GenerationRequest(
                        request_id="moa",
                        prompt=prompt,
                        sampling_params=call.sampling_params,
                        tools=call.tools,
                        tool_choice=call.tool_choice,
                        tools_in_prompt=call.tools_in_prompt,
                    ),
                    GenerationResult(
                        request_id="moa",
                        prompt=prompt,
                        completions=moa_result.completions
                        or (
                            CompletionOutput(
                                index=0,
                                text=moa_result.final_text,
                                token_ids=(),
                            ),
                        ),
                        usage=GenerationUsage(
                            prompt_tokens=moa_result.usage[0],
                            completion_tokens=moa_result.usage[1],
                            cached_tokens=moa_result.cached_tokens,
                        ),
                    ),
                )
                budget_state = reservation.reconcile_success(
                    steps=moa_steps,
                    cost=moa_cost,
                    unknown_cost=True,
                )
            except Exception as error:
                released = reservation.release(
                    steps=moa_steps,
                    unknown_cost=True,
                )
                if isinstance(error, MoAExecutionError):
                    cause = error.cause
                    usage = error.usage
                    cached_tokens = error.cached_tokens
                    partial_text = error.final_text
                    stage = error.stage
                else:
                    cause = error
                    usage = (
                        moa_result.usage
                        if moa_result is not None
                        else (
                            latest_usage.prompt_tokens,
                            latest_usage.completion_tokens,
                        )
                    )
                    cached_tokens = (
                        moa_result.cached_tokens
                        if moa_result is not None
                        else latest_usage.cached_tokens
                    )
                    partial_text = moa_result.final_text if moa_result is not None else ""
                    stage = "accounting"
                trace_events.append(
                    TraceEvent(
                        node="moa",
                        kind="failed",
                        operation="synthesis",
                        status="failed",
                        role="moa",
                        worker="tier1,tier2",
                        engine=",".join(
                            dict.fromkeys(
                                (
                                    proposal_engine_name,
                                    synthesizer_engine_name,
                                )
                            )
                        ),
                        model=",".join(
                            dict.fromkeys(
                                model
                                for model in (
                                    proposal_descriptor.model,
                                    synthesizer_descriptor.model,
                                )
                                if model is not None
                            )
                        )
                        or None,
                        timing=TraceTiming(
                            queued_at=moa_queued_at,
                            started_at=moa_started_at,
                            first_token_at=first_token_at,
                            completed_at=utc_now_iso(),
                        ),
                        usage=TraceUsage(
                            prompt_tokens=usage[0],
                            completion_tokens=usage[1],
                            cached_tokens=cached_tokens,
                        ),
                        budget=TraceBudget.between(
                            budget_before,
                            released,
                        ),
                        metadata={"stage": stage, "streamed": True},
                        error=TraceError(type=type(cause).__name__),
                    )
                )
                yield OrchestratorEvent(
                    kind="error",
                    result=result_with_trace(
                        text=partial_text,
                        prompt_tokens=usage[0],
                        completion_tokens=usage[1],
                        cached_tokens=cached_tokens,
                    ),
                    error_type=type(cause).__name__,
                )
                return
            except BaseException:
                reservation.release(steps=moa_steps, unknown_cost=True)
                raise

            notes.append(
                f"moa: {len(moa_result.proposals)} proposals synthesized (cost={moa_cost:.4f})"
            )
            if budget_state.is_exhausted:
                notes.append("moa: budget exceeded")
            resolved_engines = tuple(dict.fromkeys((proposal_engine_name, synthesizer_engine_name)))
            resolved_models = tuple(
                dict.fromkeys(
                    model
                    for model in (
                        proposal_descriptor.model,
                        synthesizer_descriptor.model,
                    )
                    if model is not None
                )
            )
            trace_events.append(
                TraceEvent(
                    node="moa",
                    kind="synthesized",
                    operation="synthesis",
                    status="success",
                    role="moa",
                    worker="tier1,tier2",
                    engine=",".join(resolved_engines),
                    model=",".join(resolved_models) or None,
                    timing=TraceTiming(
                        queued_at=moa_queued_at,
                        started_at=moa_started_at,
                        first_token_at=first_token_at,
                        completed_at=utc_now_iso(),
                    ),
                    usage=TraceUsage(
                        prompt_tokens=moa_result.usage[0],
                        completion_tokens=moa_result.usage[1],
                        cached_tokens=moa_result.cached_tokens,
                    ),
                    budget=TraceBudget.between(
                        budget_before,
                        budget_state,
                        steps_consumed=moa_steps,
                        cost_consumed_usd=moa_cost,
                    ),
                    metadata={
                        "proposals": len(moa_result.proposals),
                        "cost_usd": moa_cost,
                        "budget_exhausted": budget_state.is_exhausted,
                        "proposal_engine": proposal_engine_name,
                        "proposal_model": proposal_descriptor.model,
                        "synthesizer_engine": synthesizer_engine_name,
                        "synthesizer_model": synthesizer_descriptor.model,
                        "streamed": True,
                    },
                )
            )
            yield OrchestratorEvent(
                kind="result",
                result=result_with_trace(
                    text=moa_result.final_text,
                    completions=moa_result.completions
                    or (
                        CompletionOutput(
                            index=0,
                            text=moa_result.final_text,
                            token_ids=(),
                            finish_reason="stop",
                        ),
                    ),
                    prompt_tokens=moa_result.usage[0],
                    completion_tokens=moa_result.usage[1],
                    cached_tokens=moa_result.cached_tokens,
                ),
            )
            return

        conductor = Conductor(
            roles=self._roles,
            workers=self._conductor_workers(notes),
            shared_prefix=self._shared_prefix,
            sampling_params=self._internal_sampling_params(call),
            final_sampling_params=call.sampling_params,
            final_tools=call.tools,
            final_tool_choice=call.tool_choice,
            final_tools_in_prompt=call.tools_in_prompt,
            cost_model=self._cost_model,
            worker_trace=self._conductor_worker_trace(),
            usage_observer=usage_observer,
        )
        conductor_result = None
        try:
            async for event in self._with_initial_keepalives(
                conductor.stream(prompt, budget=self._budget)
            ):
                if event is None:
                    yield OrchestratorEvent(kind="status", text="working")
                elif event.kind == "delta":
                    yield OrchestratorEvent(
                        kind="delta",
                        text=event.text,
                        completions=event.completions,
                    )
                else:
                    conductor_result = event.result
        except ConductorStreamError as error:
            conductor_result = error.result
            notes.extend(
                f"{event.node}: {event.kind} {event.detail}" for event in conductor_result.trace
            )
            trace_events.extend(conductor_result.trace)
            yield OrchestratorEvent(
                kind="error",
                result=result_with_trace(
                    text=conductor_result.final_text,
                    completions=conductor_result.completions,
                    prompt_tokens=conductor_result.usage[0],
                    completion_tokens=conductor_result.usage[1],
                    cached_tokens=conductor_result.cached_tokens,
                ),
                error_type=type(error.cause).__name__,
            )
            return
        if conductor_result is None:
            raise RuntimeError("Conductor stream did not produce a final result")
        notes.extend(
            f"{event.node}: {event.kind} {event.detail}" for event in conductor_result.trace
        )
        trace_events.extend(conductor_result.trace)
        yield OrchestratorEvent(
            kind="result",
            result=result_with_trace(
                text=conductor_result.final_text,
                completions=conductor_result.completions,
                prompt_tokens=conductor_result.usage[0],
                completion_tokens=conductor_result.usage[1],
                cached_tokens=conductor_result.cached_tokens,
            ),
        )

    def run_sync(self, query: str) -> OrchestratorResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(query))
        raise RuntimeError(
            "run_sync() cannot be called from a running event loop; await run() instead"
        )
