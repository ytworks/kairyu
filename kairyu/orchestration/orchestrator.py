"""Orchestrator facade: route a query, then dispatch to an engine or the Conductor."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

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


@dataclass(frozen=True)
class OrchestratorEvent:
    """Streaming event (m11 D1): status keep-alive, token delta, or final."""

    kind: str  # "status" | "delta" | "result"
    text: str = ""
    result: OrchestratorResult | None = None


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
    ) -> None:
        if not engines:
            raise ValueError("Orchestrator requires at least one engine")
        self._engines = dict(engines)
        self._router = router or RuleRouter()
        self._roles = roles or _DEFAULT_ROLES
        self._budget = budget or Budget()
        self._shared_prefix = shared_prefix
        self._sampling_params = sampling_params or SamplingParams(max_tokens=1024)
        self._cost_model = cost_model
        # m11 A4: >0 routes multi_agent through MoA (the deep kairyu-auto-max tier)
        self._moa_samples = moa_samples

    async def shutdown(self) -> None:
        await shutdown_all(self._engines.values(), "Orchestrator")

    def _resolve_engine(self, tier: str, notes: list[str]) -> EngineBackend:
        engine = self._engines.get(tier)
        if engine is not None:
            return engine
        fallback_name = next(iter(self._engines))
        notes.append(f"fallback: engine {tier!r} not configured, using {fallback_name!r}")
        return self._engines[fallback_name]

    def _conductor_workers(self, notes: list[str]) -> dict[str, EngineBackend]:
        needed = {role.worker for role in self._roles}
        return {name: self._resolve_engine(name, notes) for name in needed}

    async def _run_direct(
        self, query: str, tier: str, notes: list[str]
    ) -> tuple[str, tuple[int, int]]:
        engine = self._resolve_engine(tier, notes)
        request = GenerationRequest(
            request_id=f"direct-{uuid.uuid4().hex[:12]}",
            prompt=f"{self._shared_prefix}{query}",
            sampling_params=self._sampling_params,
            # no random per-request session (M2): a fresh uuid forces uniform HRW
            # placement and defeats prefix + least-outstanding routing. With no
            # hint the pool routes by shared-prefix overlap and load instead.
            cache_hint=None,
        )
        result = await engine.generate(request)
        usage = (
            (result.usage.prompt_tokens, result.usage.completion_tokens)
            if result.usage is not None
            else (0, 0)
        )
        return result.text, usage

    async def run(self, query: str) -> OrchestratorResult:
        decision = self._router.route(query)
        notes: list[str] = [f"route: {decision.target} ({decision.reason})"]
        if decision.target == "multi_agent":
            if self._moa_samples > 0:  # m11 A4: the deep tier's MoA route
                from kairyu.orchestration.moa import run_moa

                moa = await run_moa(
                    self._resolve_engine("tier1", notes),
                    query,
                    n_samples=self._moa_samples,
                    synthesizer=self._resolve_engine("tier2", notes),
                    shared_prefix=self._shared_prefix,
                )
                # M3: the deep MoA tier was invisible to the cost model / budget.
                # Charge it (steps = proposals + synthesis) and surface whether it
                # exceeded max_cost_usd in the trace (budget overrun is queryable,
                # not a raise — matching the Budget philosophy).
                moa_cost = self._cost_model(
                    GenerationRequest(
                        request_id="moa", prompt=query,
                        sampling_params=self._sampling_params,
                    ),
                    GenerationResult(
                        request_id="moa", prompt=query,
                        completions=(
                            CompletionOutput(index=0, text=moa.final_text, token_ids=()),
                        ),
                        usage=GenerationUsage(
                            prompt_tokens=moa.usage[0], completion_tokens=moa.usage[1]
                        ),
                    ),
                )
                budget_state = BudgetState(budget=self._budget).charge(
                    steps=self._moa_samples + 1, cost=moa_cost
                )
                notes.append(
                    f"moa: {len(moa.proposals)} proposals synthesized "
                    f"(cost={moa_cost:.4f})"
                )
                if budget_state.is_exhausted:
                    notes.append("moa: budget exceeded")
                return OrchestratorResult(
                    text=moa.final_text,
                    route=decision,
                    trace=tuple(notes),
                    prompt_tokens=moa.usage[0],
                    completion_tokens=moa.usage[1],
                )
            conductor = Conductor(
                roles=self._roles,
                workers=self._conductor_workers(notes),
                shared_prefix=self._shared_prefix,
                sampling_params=self._sampling_params,
                cost_model=self._cost_model,
            )
            result = await conductor.run(query, budget=self._budget)
            notes.extend(f"{event.node}: {event.kind} {event.detail}" for event in result.trace)
            return OrchestratorResult(
                text=result.final_text,
                route=decision,
                trace=tuple(notes),
                prompt_tokens=result.usage[0],
                completion_tokens=result.usage[1],
            )
        text, usage = await self._run_direct(query, decision.target, notes)
        return OrchestratorResult(
            text=text,
            route=decision,
            trace=tuple(notes),
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
            async for partial in engine.stream(request):
                last = partial
                text = partial.text
                if len(text) > emitted:
                    yield OrchestratorEvent(kind="delta", text=text[emitted:])
                    emitted = len(text)
            usage = (
                (last.usage.prompt_tokens, last.usage.completion_tokens)
                if last is not None and last.usage is not None
                else (0, 0)
            )  # usage read from the LAST partial (m11 A1 contract)
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
