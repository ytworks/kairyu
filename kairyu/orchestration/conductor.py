"""Conductor: declarative role DAG executed with asyncio waves (design doc D4).

Roles (planner/worker/verifier/synthesizer/custom) form a DAG; a verifier node
gates its target with a bounded refine loop. All prompts are rendered as
``shared_prefix + role_suffix`` so multi-step calls share a KV-cacheable prefix
(design doc D5).
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from kairyu.engine.backend import CacheHint, EngineBackend, GenerationRequest, GenerationResult
from kairyu.orchestration.budget import Budget, BudgetState
from kairyu.sampling_params import SamplingParams

_PASS_PREFIX = "PASS"

CostModel = Callable[[GenerationRequest, GenerationResult], float]


def zero_cost(request: GenerationRequest, result: GenerationResult) -> float:
    return 0.0


def chars_cost_model(usd_per_1k_chars: float) -> CostModel:
    """Approximate cost from prompt+completion character volume."""

    def estimate(request: GenerationRequest, result: GenerationResult) -> float:
        chars = len(request.prompt) + sum(len(c.text) for c in result.completions)
        return chars / 1000 * usd_per_1k_chars

    return estimate


@dataclass(frozen=True)
class RoleSpec:
    name: str
    worker: str
    prompt: str
    role_type: str = "worker"
    depends_on: tuple[str, ...] = ()
    verifies: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "depends_on", tuple(self.depends_on))


@dataclass(frozen=True)
class TraceEvent:
    node: str
    kind: str
    detail: str = ""


@dataclass(frozen=True)
class ConductorResult:
    final_text: str
    outputs: dict[str, str]
    budget_state: BudgetState
    trace: tuple[TraceEvent, ...]


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


@dataclass
class _RunState:
    """Mutable accumulator local to one run(); public results are frozen."""

    budget: BudgetState
    outputs: dict[str, str] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)
    completion_order: list[str] = field(default_factory=list)


def _is_pass(verdict_text: str) -> bool:
    first_line = verdict_text.strip().splitlines()[0] if verdict_text.strip() else ""
    return first_line.upper().startswith(_PASS_PREFIX)


class Conductor:
    def __init__(
        self,
        roles: tuple[RoleSpec, ...],
        workers: Mapping[str, EngineBackend],
        shared_prefix: str = "",
        sampling_params: SamplingParams | None = None,
        cost_model: CostModel = zero_cost,
    ) -> None:
        self._roles = tuple(roles)
        self._workers = dict(workers)
        self._shared_prefix = shared_prefix
        self._sampling_params = sampling_params or SamplingParams(max_tokens=1024)
        self._cost_model = cost_model
        self._by_name = {role.name: role for role in self._roles}
        self._verifier_for = {
            role.verifies: role for role in self._roles if role.role_type == "verifier"
        }
        self._units = tuple(role for role in self._roles if role.role_type != "verifier")
        self._unit_deps = {unit.name: self._remapped_deps(unit) for unit in self._units}
        self._validate()

    def _remapped_deps(self, unit: RoleSpec) -> frozenset[str]:
        """Dependencies at unit granularity: a dep on a verifier maps to its target."""
        deps = set()
        for dep in unit.depends_on:
            dep_role = self._by_name.get(dep)
            if dep_role is not None and dep_role.role_type == "verifier" and dep_role.verifies:
                deps.add(dep_role.verifies)
            else:
                deps.add(dep)
        return frozenset(deps - {unit.name})

    def _validate(self) -> None:
        if len(self._by_name) != len(self._roles):
            raise ValueError("duplicate role names")
        for role in self._roles:
            if role.worker not in self._workers:
                raise ValueError(f"role {role.name!r} references unknown worker {role.worker!r}")
            for dep in role.depends_on:
                if dep not in self._by_name:
                    raise ValueError(f"role {role.name!r} has unknown dependency {dep!r}")
            if role.role_type == "verifier":
                if role.verifies is None or role.verifies not in self._by_name:
                    raise ValueError(f"verifier {role.name!r} must set verifies=<existing role>")
                if role.verifies not in role.depends_on:
                    raise ValueError(
                        f"verifier {role.name!r} must depend on its target {role.verifies!r}"
                    )
        self._check_acyclic()

    def _check_acyclic(self) -> None:
        remaining = {name: set(deps) for name, deps in self._unit_deps.items()}
        while remaining:
            ready = [name for name, deps in remaining.items() if not deps]
            if not ready:
                raise ValueError(f"role DAG contains a cycle among: {sorted(remaining)}")
            for name in ready:
                del remaining[name]
            for deps in remaining.values():
                deps.difference_update(ready)

    def _render(self, template: str, query: str, outputs: Mapping[str, str]) -> str:
        body = template.format_map(_SafeDict(query=query, **outputs))
        return f"{self._shared_prefix}{body}"

    def _cache_hint(self, session: str) -> CacheHint:
        fingerprint = hashlib.sha256(self._shared_prefix.encode()).hexdigest()[:16]
        return CacheHint(session_id=session, prefix_fingerprint=fingerprint)

    async def _generate(
        self, run: _RunState, session: str, node: str, worker: str, prompt: str, attempt: int
    ) -> str:
        backend = self._workers[worker]
        request = GenerationRequest(
            request_id=f"{session}-{node}-{attempt}",
            prompt=prompt,
            sampling_params=self._sampling_params,
            cache_hint=self._cache_hint(session),
        )
        result = await backend.generate(request)
        run.budget = run.budget.charge(cost=self._cost_model(request, result))
        return result.text

    async def _run_unit(self, run: _RunState, session: str, query: str, spec: RoleSpec) -> None:
        if run.budget.is_exhausted:
            run.trace.append(TraceEvent(spec.name, "skipped:budget"))
            return
        base_prompt = self._render(spec.prompt, query, run.outputs)
        verifier = self._verifier_for.get(spec.name)
        prompt = base_prompt
        depth = 0
        while True:
            text = await self._generate(run, session, spec.name, spec.worker, prompt, depth)
            run.outputs[spec.name] = text
            run.trace.append(TraceEvent(spec.name, "generated", f"attempt={depth}"))
            if verifier is None:
                break
            verifier_prompt = self._render(verifier.prompt, query, run.outputs)
            verdict = await self._generate(
                run, session, verifier.name, verifier.worker, verifier_prompt, depth
            )
            run.outputs[verifier.name] = verdict
            passed = _is_pass(verdict)
            run.trace.append(
                TraceEvent(verifier.name, "verified", f"attempt={depth} pass={passed}")
            )
            if passed or not run.budget.can_refine(depth):
                break
            depth += 1
            prompt = (
                f"{base_prompt}\n\nPrevious attempt:\n{text}\n\n"
                f"Verifier feedback:\n{verdict}\n\nRevise the answer addressing the feedback."
            )
        run.completion_order.append(spec.name)

    def _final_text(self, run: _RunState) -> str:
        dependents: set[str] = set()
        for deps in self._unit_deps.values():
            dependents.update(deps)
        terminal = [unit for unit in self._units if unit.name not in dependents]
        synthesizers = [unit for unit in terminal if unit.role_type == "synthesizer"]
        for unit in synthesizers + terminal:
            if unit.name in run.outputs:
                return run.outputs[unit.name]
        if run.completion_order:
            return run.outputs[run.completion_order[-1]]
        return ""

    async def run(self, query: str, budget: Budget | None = None) -> ConductorResult:
        run = _RunState(budget=BudgetState(budget=budget or Budget()))
        session = uuid.uuid4().hex[:12]
        pending = {name: set(deps) for name, deps in self._unit_deps.items()}
        while pending:
            ready = [name for name, deps in pending.items() if not deps]
            await asyncio.gather(
                *(self._run_unit(run, session, query, self._by_name[name]) for name in ready)
            )
            for name in ready:
                del pending[name]
            for deps in pending.values():
                deps.difference_update(ready)
        return ConductorResult(
            final_text=self._final_text(run),
            outputs=dict(run.outputs),
            budget_state=run.budget,
            trace=tuple(run.trace),
        )
