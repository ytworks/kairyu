"""Orchestrator facade: route a query, then dispatch to an engine or the Conductor."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from kairyu.engine.backend import (
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    shutdown_all,
)
from kairyu.orchestration.budget import Budget, BudgetState
from kairyu.orchestration.conductor import Conductor, CostModel, RoleSpec, zero_cost
from kairyu.orchestration.router import RouteDecision, Router, RuleRouter
from kairyu.orchestration.trace import (
    StructuredTrace,
    TraceBudget,
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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    structured_trace: StructuredTrace | None = None


@dataclass(frozen=True)
class OrchestratorEvent:
    """Streaming event (m11 D1): status keep-alive, token delta, or final."""

    kind: str  # "status" | "delta" | "result"
    text: str = ""
    result: OrchestratorResult | None = None


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
            describe()
            if describe is not None
            else {"router_type": type(self._router).__name__}
        )
        role_workers = tuple(dict.fromkeys(role.worker for role in self._roles))
        if self._moa_samples > 0:
            multi_engines = [
                self._resolved_engine_descriptor(key) for key in ("tier1", "tier2")
            ]
            multi_mode = "moa"
        else:
            multi_engines = [
                self._resolved_engine_descriptor(key) for key in role_workers
            ]
            multi_mode = "roles"
        return {
            "router": router,
            "targets": ["tier1", "tier2", "multi_agent"],
            "configured_engines": {
                name: descriptor.as_dict()
                for name, descriptor in self._engine_descriptors.items()
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
        self, query: str, tier: str, notes: list[str]
    ) -> tuple[str, tuple[int, int], TraceEvent]:
        queued_at = utc_now_iso()
        engine_name = self._resolve_engine_name(tier, notes)
        engine = self._engines[engine_name]
        descriptor = self._engine_descriptors[engine_name]
        request = GenerationRequest(
            request_id=f"direct-{uuid.uuid4().hex[:12]}",
            prompt=f"{self._shared_prefix}{query}",
            sampling_params=self._sampling_params,
            # no random per-request session (M2): a fresh uuid forces uniform HRW
            # placement and defeats prefix + least-outstanding routing. With no
            # hint the pool routes by shared-prefix overlap and load instead.
            cache_hint=None,
        )
        started_at = utc_now_iso()
        result = await engine.generate(request)
        usage = (
            (result.usage.prompt_tokens, result.usage.completion_tokens)
            if result.usage is not None
            else (0, 0)
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
            result.text,
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

    async def run(self, query: str) -> OrchestratorResult:
        request_id = f"orch-{uuid.uuid4().hex[:16]}"
        trace_started_at = utc_now_iso()
        route_started_at = utc_now_iso()
        decision = self._router.route(query)
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
            prompt_tokens: int = 0,
            completion_tokens: int = 0,
        ) -> OrchestratorResult:
            return OrchestratorResult(
                text=text,
                route=decision,
                trace=tuple(notes),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                structured_trace=StructuredTrace(
                    request_id=request_id,
                    started_at=trace_started_at,
                    completed_at=utc_now_iso(),
                    events=tuple(trace_events),
                ),
            )

        if decision.target == "multi_agent":
            if self._moa_samples > 0:  # m11 A4: the deep tier's MoA route
                from kairyu.orchestration.moa import run_moa

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
                synthesizer_descriptor = self._engine_descriptors[
                    synthesizer_engine_name
                ]
                moa_started_at = utc_now_iso()
                try:
                    moa = await run_moa(
                        self._engines[proposal_engine_name],
                        query,
                        n_samples=self._moa_samples,
                        synthesizer=self._engines[synthesizer_engine_name],
                        shared_prefix=self._shared_prefix,
                    )
                    # M3: the deep MoA tier was invisible to the cost model / budget.
                    # Reconcile proposals + synthesis with the actual result cost;
                    # one admitted operation may visibly cross a result-priced cap.
                    moa_cost = self._cost_model(
                        GenerationRequest(
                            request_id="moa", prompt=query,
                            sampling_params=self._sampling_params,
                        ),
                        GenerationResult(
                            request_id="moa", prompt=query,
                            completions=(
                                CompletionOutput(
                                    index=0, text=moa.final_text, token_ids=()
                                ),
                            ),
                            usage=GenerationUsage(
                                prompt_tokens=moa.usage[0],
                                completion_tokens=moa.usage[1],
                            ),
                        ),
                    )
                    budget_state = reservation.reconcile_success(
                        steps=moa_steps,
                        cost=moa_cost,
                        unknown_cost=True,
                    )
                except BaseException:
                    reservation.release(steps=moa_steps, unknown_cost=True)
                    raise
                notes.append(
                    f"moa: {len(moa.proposals)} proposals synthesized "
                    f"(cost={moa_cost:.4f})"
                )
                if budget_state.is_exhausted:
                    notes.append("moa: budget exceeded")
                resolved_engines = tuple(
                    dict.fromkeys(
                        (proposal_engine_name, synthesizer_engine_name)
                    )
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
                    prompt_tokens=moa.usage[0],
                    completion_tokens=moa.usage[1],
                )
            conductor = Conductor(
                roles=self._roles,
                workers=self._conductor_workers(notes),
                shared_prefix=self._shared_prefix,
                sampling_params=self._sampling_params,
                cost_model=self._cost_model,
                worker_trace=self._conductor_worker_trace(),
            )
            result = await conductor.run(query, budget=self._budget)
            notes.extend(f"{event.node}: {event.kind} {event.detail}" for event in result.trace)
            trace_events.extend(result.trace)
            return result_with_trace(
                text=result.final_text,
                prompt_tokens=result.usage[0],
                completion_tokens=result.usage[1],
            )
        text, usage, direct_event = await self._run_direct(
            query,
            decision.target,
            notes,
        )
        trace_events.append(direct_event)
        return result_with_trace(
            text=text,
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
        )

    async def run_chat(self, prompt: str, stream: bool = False):
        """The m11 D1 surface: pre-rendered prompt in (A4), events out.

        Non-stream: returns OrchestratorResult (same as run()). Stream:
        async-yields OrchestratorEvent — status keep-alives while pre-final
        stages run, token deltas for the FINAL text (direct route streams
        live; conductor/MoA finals are buffered per A5 — refine regeneration
        would invalidate streamed deltas), then one result event.
        """
        if not stream:
            return await self.run(prompt)
        return self._run_chat_stream(prompt)

    async def _run_chat_stream(self, prompt: str):
        decision = self._router.route(prompt)
        notes = [f"route: {decision.target} ({decision.reason})"]
        if decision.target != "multi_agent":
            engine = self._resolve_engine(decision.target, notes)
            request = GenerationRequest(
                request_id=f"direct-{uuid.uuid4().hex[:12]}",
                prompt=f"{self._shared_prefix}{prompt}",
                sampling_params=self._sampling_params,
                cache_hint=None,  # M2: no random session — route by prefix + load
            )
            emitted = 0
            last = None
            latest_usage = None
            async for partial in engine.stream(request):
                last = partial
                if partial.usage is not None:
                    latest_usage = partial.usage
                text = partial.text
                if len(text) > emitted:
                    yield OrchestratorEvent(kind="delta", text=text[emitted:])
                    emitted = len(text)
            usage = (
                (latest_usage.prompt_tokens, latest_usage.completion_tokens)
                if latest_usage is not None
                else (0, 0)
            )
            yield OrchestratorEvent(
                kind="result",
                result=OrchestratorResult(
                    text=last.text if last is not None else "",
                    route=decision,
                    trace=tuple(notes),
                    prompt_tokens=usage[0],
                    completion_tokens=usage[1],
                ),
            )
            return
        # multi-stage routes: PERIODIC keep-alives while the (possibly minutes-
        # long) run executes, then the buffered final (A5). Without the periodic
        # emit a proxy/LB idle timeout would sever the SSE connection (M8).
        yield OrchestratorEvent(kind="status", text=f"routing: {decision.target}")
        run_task = asyncio.ensure_future(self.run(prompt))
        try:
            while not run_task.done():
                done, _ = await asyncio.wait({run_task}, timeout=_KEEPALIVE_INTERVAL_S)
                if not done:
                    yield OrchestratorEvent(kind="status", text="working")
            result = run_task.result()
        finally:
            if not run_task.done():
                run_task.cancel()
        yield OrchestratorEvent(kind="delta", text=result.text)
        yield OrchestratorEvent(kind="result", result=result)

    def run_sync(self, query: str) -> OrchestratorResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(query))
        raise RuntimeError(
            "run_sync() cannot be called from a running event loop; await run() instead"
        )
