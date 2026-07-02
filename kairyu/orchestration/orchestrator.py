"""Orchestrator facade: route a query, then dispatch to an engine or the Conductor."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from kairyu.engine.backend import CacheHint, EngineBackend, GenerationRequest
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.conductor import Conductor, CostModel, RoleSpec, zero_cost
from kairyu.orchestration.router import RouteDecision, Router, RuleRouter
from kairyu.sampling_params import SamplingParams

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

    async def _run_direct(self, query: str, tier: str, notes: list[str]) -> str:
        engine = self._resolve_engine(tier, notes)
        request = GenerationRequest(
            request_id=f"direct-{uuid.uuid4().hex[:12]}",
            prompt=f"{self._shared_prefix}{query}",
            sampling_params=self._sampling_params,
            cache_hint=CacheHint(session_id=uuid.uuid4().hex[:12]),
        )
        return (await engine.generate(request)).text

    async def run(self, query: str) -> OrchestratorResult:
        decision = self._router.route(query)
        notes: list[str] = [f"route: {decision.target} ({decision.reason})"]
        if decision.target == "multi_agent":
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
                text=result.final_text, route=decision, trace=tuple(notes)
            )
        text = await self._run_direct(query, decision.target, notes)
        return OrchestratorResult(text=text, route=decision, trace=tuple(notes))

    def run_sync(self, query: str) -> OrchestratorResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(query))
        raise RuntimeError(
            "run_sync() cannot be called from a running event loop; await run() instead"
        )
