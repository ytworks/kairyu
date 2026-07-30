"""Conductor: declarative role DAG executed with asyncio waves (design doc D4).

Roles (planner/worker/verifier/synthesizer/custom) form a DAG; a verifier node
gates its target with a bounded refine loop. All prompts are rendered as
``shared_prefix + role_suffix`` so multi-step calls share a KV-cacheable prefix
(design doc D5).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field

from kairyu.engine.backend import (
    CacheHint,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
)
from kairyu.engine.prompt import prompt_kind, prompt_text
from kairyu.orchestration.budget import Budget, BudgetState
from kairyu.orchestration.prefix_index import prefix_root_fingerprint
from kairyu.orchestration.trace import (
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

_PASS_PREFIX = "PASS"

CostModel = Callable[[GenerationRequest, GenerationResult], float]


def zero_cost(request: GenerationRequest, result: GenerationResult) -> float:
    return 0.0


def chars_cost_model(usd_per_1k_chars: float) -> CostModel:
    """Approximate cost from prompt+completion character volume."""

    def estimate(request: GenerationRequest, result: GenerationResult) -> float:
        if prompt_kind(request.prompt) != "text":
            raise ValueError("character cost estimation supports text prompts only")
        text = prompt_text(request.prompt)
        assert text is not None
        chars = len(text) + sum(len(c.text) for c in result.completions)
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
class ConductorResult:
    final_text: str
    completions: tuple[CompletionOutput, ...]
    outputs: dict[str, str]
    budget_state: BudgetState
    trace: tuple[TraceEvent, ...]
    usage: tuple[int, int] = (0, 0)  # (prompt, completion) summed over units (m11 A1)
    cached_tokens: int = 0


@dataclass(frozen=True)
class ConductorEvent:
    """A pull-through final-stage stream event.

    The Conductor owns the final worker iterator so no task/queue bridge sits
    between the backend and the Orchestrator.  Pre-final DAG work remains an
    implementation detail; callers receive only committed final-answer deltas
    followed by the complete accounting result.
    """

    kind: str  # "delta" | "result"
    text: str = ""
    completions: tuple[CompletionOutput, ...] = ()
    result: ConductorResult | None = None


class ConductorStreamError(RuntimeError):
    """Carry privacy-safe partial state when the final backend stream fails."""

    def __init__(
        self,
        cause: BaseException,
        result: ConductorResult,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.result = result


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
    usage: list[int] = field(default_factory=lambda: [0, 0])
    cached_tokens: int = 0
    final_completions: tuple[CompletionOutput, ...] = ()


class _BudgetRefused(Exception):
    """Internal control flow for a generation that could not reserve budget."""


class _ObservedGenerationError(Exception):
    """A backend/cost failure already represented in the structured trace."""


@dataclass(frozen=True)
class _GenerationObservation:
    text: str
    completions: tuple[CompletionOutput, ...]
    timing: TraceTiming
    usage: TraceUsage | None
    budget: TraceBudget


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
        final_sampling_params: SamplingParams | None = None,
        final_tools: tuple[Mapping[str, object], ...] = (),
        final_tool_choice: str | Mapping[str, object] | None = None,
        final_tools_in_prompt: bool = False,
        cost_model: CostModel = zero_cost,
        worker_trace: Mapping[str, WorkerTraceIdentity] | None = None,
        usage_observer: Callable[[GenerationUsage], None] | None = None,
    ) -> None:
        self._roles = tuple(roles)
        self._workers = dict(workers)
        self._shared_prefix = shared_prefix
        self._prefix_fingerprint = prefix_root_fingerprint(shared_prefix)
        self._sampling_params = sampling_params or SamplingParams(max_tokens=1024)
        self._final_sampling_params = final_sampling_params or self._sampling_params
        self._final_tools = tuple(final_tools)
        self._final_tool_choice = final_tool_choice
        self._final_tools_in_prompt = final_tools_in_prompt
        self._cost_model = cost_model
        self._usage_observer = usage_observer
        supplied_trace = dict(worker_trace or {})
        self._worker_trace = {
            worker: supplied_trace.get(
                worker,
                WorkerTraceIdentity(engine=worker),
            )
            for worker in self._workers
        }
        self._by_name = {role.name: role for role in self._roles}
        self._verifier_for = {
            role.verifies: role for role in self._roles if role.role_type == "verifier"
        }
        self._units = tuple(role for role in self._roles if role.role_type != "verifier")
        self._unit_deps = {unit.name: self._remapped_deps(unit) for unit in self._units}
        self._validate()

    def _selected_final_unit(self) -> RoleSpec:
        terminal = self._terminal_units()
        if not terminal:
            raise ValueError("orchestration requires at least one generation role")
        synthesizers = [unit for unit in terminal if unit.role_type == "synthesizer"]
        return (synthesizers + terminal)[0]

    def _request_intent(
        self,
        spec: RoleSpec,
    ) -> tuple[
        SamplingParams,
        tuple[Mapping[str, object], ...],
        str | Mapping[str, object] | None,
        bool,
    ]:
        if spec.name == self._selected_final_unit().name:
            return (
                self._final_sampling_params,
                self._final_tools,
                self._final_tool_choice,
                self._final_tools_in_prompt,
            )
        return self._sampling_params, (), None, False

    def _observe_usage(
        self,
        run: _RunState,
        pending: GenerationUsage | None = None,
    ) -> None:
        if self._usage_observer is None:
            return
        self._usage_observer(
            GenerationUsage(
                prompt_tokens=run.usage[0] + (pending.prompt_tokens if pending is not None else 0),
                completion_tokens=run.usage[1]
                + (pending.completion_tokens if pending is not None else 0),
                cached_tokens=run.cached_tokens
                + (pending.cached_tokens if pending is not None else 0),
            )
        )

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
                # a verifier runs INLINE right after its target, so any OTHER
                # dependency must also be the target's dependency (else it may not
                # have run yet and _SafeDict would render it as "" → a silent
                # wrong PASS/FAIL). Catch the misconfiguration loudly (M1).
                target = self._by_name[role.verifies]
                available = set(target.depends_on) | {role.verifies}
                for dep in role.depends_on:
                    if dep != role.verifies and dep not in available:
                        raise ValueError(
                            f"verifier {role.name!r} depends on {dep!r}, which is not "
                            f"available when it runs inline after {role.verifies!r}; "
                            f"add {dep!r} to {role.verifies!r}'s depends_on"
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
        return CacheHint(
            session_id=session,
            prefix_fingerprint=self._prefix_fingerprint,
        )

    def _trace_event(
        self,
        spec: RoleSpec,
        kind: str,
        *,
        operation: str,
        status: str,
        attempt: int,
        detail: str = "",
        timing: TraceTiming | None = None,
        usage: TraceUsage | None = None,
        budget: TraceBudget | None = None,
        metadata: dict | None = None,
        error: TraceError | None = None,
    ) -> TraceEvent:
        identity = self._worker_trace[spec.worker]
        return TraceEvent(
            node=spec.name,
            kind=kind,
            detail=detail,
            operation=operation,
            status=status,
            attempt=attempt,
            role=spec.role_type,
            worker=spec.worker,
            engine=identity.engine,
            model=identity.model,
            timing=timing,
            usage=usage,
            budget=budget,
            metadata=dict(metadata or {}),
            error=error,
        )

    async def _generate(
        self,
        run: _RunState,
        session: str,
        node: str,
        worker: str,
        prompt: str,
        attempt: int,
        *,
        spec: RoleSpec | None = None,
        operation: str = "generation",
    ) -> _GenerationObservation:
        event_spec = spec or RoleSpec(name=node, worker=worker, prompt="")
        backend = self._workers[worker]
        sampling_params, tools, tool_choice, tools_in_prompt = self._request_intent(event_spec)
        request = GenerationRequest(
            request_id=f"{session}-{node}-{attempt}",
            prompt=prompt,
            sampling_params=sampling_params,
            cache_hint=self._cache_hint(session),
            tools=tools,
            tool_choice=tool_choice,
            tools_in_prompt=tools_in_prompt,
        )
        queued_at = utc_now_iso()
        budget_before = run.budget
        unknown_cost = run.budget.budget.max_cost_usd is not None
        reserved = run.budget.try_reserve(unknown_cost=unknown_cost)
        if reserved is None:
            raise _BudgetRefused
        run.budget = reserved
        started_at = utc_now_iso()
        try:
            result = await backend.generate(request)
            actual_cost = self._cost_model(request, result)
            reconciled = run.budget.reconcile_success(
                cost=actual_cost,
                unknown_cost=unknown_cost,
            )
        except Exception as error:
            run.budget = run.budget.release(unknown_cost=unknown_cost)
            run.trace.append(
                self._trace_event(
                    event_spec,
                    "failed",
                    operation=operation,
                    status="failed",
                    attempt=attempt,
                    detail=type(error).__name__,
                    timing=TraceTiming(
                        queued_at=queued_at,
                        started_at=started_at,
                        completed_at=utc_now_iso(),
                    ),
                    budget=TraceBudget.between(budget_before, run.budget),
                    error=TraceError(type=type(error).__name__),
                )
            )
            raise _ObservedGenerationError from error
        except BaseException:
            run.budget = run.budget.release(unknown_cost=unknown_cost)
            raise
        run.budget = reconciled
        trace_usage = None
        if result.usage is not None:  # m11 A1: usage was dropped here
            run.usage[0] += result.usage.prompt_tokens
            run.usage[1] += result.usage.completion_tokens
            run.cached_tokens += result.usage.cached_tokens
            self._observe_usage(run)
            trace_usage = TraceUsage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                cached_tokens=result.usage.cached_tokens,
            )
        return _GenerationObservation(
            text=result.text,
            completions=result.completions,
            timing=TraceTiming(
                queued_at=queued_at,
                started_at=started_at,
                completed_at=utc_now_iso(),
            ),
            usage=trace_usage,
            budget=TraceBudget.between(
                budget_before,
                run.budget,
                steps_consumed=1,
                cost_consumed_usd=actual_cost,
            ),
        )

    async def _run_unit(self, run: _RunState, session: str, query: str, spec: RoleSpec) -> None:
        if run.budget.is_exhausted:
            run.trace.append(
                self._trace_event(
                    spec,
                    "skipped:budget",
                    operation="generation",
                    status="skipped",
                    attempt=0,
                    budget=TraceBudget.between(run.budget, run.budget),
                    metadata={"reason": "budget"},
                )
            )
            return
        base_prompt = self._render(spec.prompt, query, run.outputs)
        verifier = self._verifier_for.get(spec.name)
        prompt = base_prompt
        depth = 0
        while True:
            try:
                observed = await self._generate(
                    run,
                    session,
                    spec.name,
                    spec.worker,
                    prompt,
                    depth,
                    spec=spec,
                    operation="generation",
                )
            except _BudgetRefused:
                run.trace.append(
                    self._trace_event(
                        spec,
                        "skipped:budget",
                        operation="generation",
                        status="skipped",
                        attempt=depth,
                        budget=TraceBudget.between(run.budget, run.budget),
                        metadata={"reason": "budget"},
                    )
                )
                if spec.name not in run.outputs:
                    return
                break
            text = observed.text
            run.outputs[spec.name] = text
            if spec.name == self._selected_final_unit().name:
                run.final_completions = observed.completions
            run.trace.append(
                self._trace_event(
                    spec,
                    "generated",
                    operation="generation",
                    status="success",
                    attempt=depth,
                    detail=f"attempt={depth}",
                    timing=observed.timing,
                    usage=observed.usage,
                    budget=observed.budget,
                )
            )
            if verifier is None:
                break
            verifier_prompt = self._render(verifier.prompt, query, run.outputs)
            try:
                verifier_observed = await self._generate(
                    run,
                    session,
                    verifier.name,
                    verifier.worker,
                    verifier_prompt,
                    depth,
                    spec=verifier,
                    operation="verification",
                )
            except _BudgetRefused:
                run.trace.append(
                    self._trace_event(
                        verifier,
                        "skipped:budget",
                        operation="verification",
                        status="skipped",
                        attempt=depth,
                        budget=TraceBudget.between(run.budget, run.budget),
                        metadata={"reason": "budget"},
                    )
                )
                break
            verdict = verifier_observed.text
            run.outputs[verifier.name] = verdict
            passed = _is_pass(verdict)
            run.trace.append(
                self._trace_event(
                    verifier,
                    "verified",
                    operation="verification",
                    status="success",
                    attempt=depth,
                    detail=f"attempt={depth} pass={passed}",
                    timing=verifier_observed.timing,
                    usage=verifier_observed.usage,
                    budget=verifier_observed.budget,
                    metadata={"pass": passed},
                )
            )
            if passed or not run.budget.can_refine(depth):
                break
            depth += 1
            prompt = (
                f"{base_prompt}\n\nPrevious attempt:\n{text}\n\n"
                f"Verifier feedback:\n{verdict}\n\nRevise the answer addressing the feedback."
            )
        run.completion_order.append(spec.name)

    async def _run_unit_safe(
        self, run: _RunState, session: str, query: str, spec: RoleSpec
    ) -> None:
        # A transient backend failure on one unit must not destroy the whole
        # multi-agent run: record it and let the Conductor return the best
        # result produced so far (O4). Sibling units keep their completed work.
        try:
            await self._run_unit(run, session, query, spec)
        except _ObservedGenerationError:
            pass
        except Exception as error:
            run.trace.append(
                self._trace_event(
                    spec,
                    "failed",
                    operation="generation",
                    status="failed",
                    attempt=0,
                    detail=type(error).__name__,
                    error=TraceError(type=type(error).__name__),
                )
            )

    def _final_text(self, run: _RunState) -> str:
        terminal = self._terminal_units()
        synthesizers = [unit for unit in terminal if unit.role_type == "synthesizer"]
        for unit in synthesizers + terminal:
            if unit.name in run.outputs:
                return run.outputs[unit.name]
        if run.completion_order:
            return run.outputs[run.completion_order[-1]]
        return ""

    def _terminal_units(self) -> list[RoleSpec]:
        dependents: set[str] = set()
        for deps in self._unit_deps.values():
            dependents.update(deps)
        return [unit for unit in self._units if unit.name not in dependents]

    def _stream_final_unit(self) -> RoleSpec:
        final = self._selected_final_unit()
        if final.name in self._verifier_for:
            raise ValueError(
                "the final streamed role cannot have a post-generation verifier; "
                "verify a draft before an unverified final synthesizer"
            )
        return final

    async def _run_pending(
        self,
        run: _RunState,
        session: str,
        query: str,
        *,
        exclude: frozenset[str] = frozenset(),
    ) -> None:
        pending = {name: set(deps) for name, deps in self._unit_deps.items() if name not in exclude}
        while pending:
            ready = [name for name, deps in pending.items() if not deps]
            await asyncio.gather(
                *(self._run_unit_safe(run, session, query, self._by_name[name]) for name in ready)
            )
            for name in ready:
                del pending[name]
            for deps in pending.values():
                deps.difference_update(ready)

    async def _stream_unit(
        self,
        run: _RunState,
        session: str,
        query: str,
        spec: RoleSpec,
    ) -> AsyncIterator[ConductorEvent]:
        if run.budget.is_exhausted:
            run.trace.append(
                self._trace_event(
                    spec,
                    "skipped:budget",
                    operation="generation",
                    status="skipped",
                    attempt=0,
                    budget=TraceBudget.between(run.budget, run.budget),
                    metadata={"reason": "budget"},
                )
            )
            return

        prompt = self._render(spec.prompt, query, run.outputs)
        backend = self._workers[spec.worker]
        sampling_params, tools, tool_choice, tools_in_prompt = self._request_intent(spec)
        request = GenerationRequest(
            request_id=f"{session}-{spec.name}-0",
            prompt=prompt,
            sampling_params=sampling_params,
            cache_hint=self._cache_hint(session),
            tools=tools,
            tool_choice=tool_choice,
            tools_in_prompt=tools_in_prompt,
        )
        queued_at = utc_now_iso()
        budget_before = run.budget
        unknown_cost = run.budget.budget.max_cost_usd is not None
        reserved = run.budget.try_reserve(unknown_cost=unknown_cost)
        if reserved is None:
            run.trace.append(
                self._trace_event(
                    spec,
                    "skipped:budget",
                    operation="generation",
                    status="skipped",
                    attempt=0,
                    budget=TraceBudget.between(run.budget, run.budget),
                    metadata={"reason": "budget"},
                )
            )
            return
        run.budget = reserved

        started_at = utc_now_iso()
        emitted = 0
        last_result: GenerationResult | None = None
        latest_usage = None
        first_token_at = None
        try:
            async for partial in backend.stream(request):
                last_result = partial
                if partial.usage is not None:
                    latest_usage = partial.usage
                    self._observe_usage(run, latest_usage)
                text = partial.text
                if not text.startswith(run.outputs.get(spec.name, "")):
                    raise RuntimeError(
                        "final worker stream must emit cumulative, prefix-stable text"
                    )
                run.outputs[spec.name] = text
                if first_token_at is None and partial.completions:
                    first_token_at = utc_now_iso()
                yield ConductorEvent(
                    kind="delta",
                    text=text[emitted:],
                    completions=partial.completions,
                )
                emitted = len(text)
            if last_result is None:
                raise RuntimeError("final worker stream produced no result")
            actual_cost = self._cost_model(request, last_result)
            reconciled = run.budget.reconcile_success(
                cost=actual_cost,
                unknown_cost=unknown_cost,
            )
        except Exception as error:
            run.budget = run.budget.release(unknown_cost=unknown_cost)
            trace_usage = None
            if latest_usage is not None:
                run.usage[0] += latest_usage.prompt_tokens
                run.usage[1] += latest_usage.completion_tokens
                run.cached_tokens += latest_usage.cached_tokens
                self._observe_usage(run)
                trace_usage = TraceUsage(
                    prompt_tokens=latest_usage.prompt_tokens,
                    completion_tokens=latest_usage.completion_tokens,
                    cached_tokens=latest_usage.cached_tokens,
                )
            run.trace.append(
                self._trace_event(
                    spec,
                    "failed",
                    operation="generation",
                    status="failed",
                    attempt=0,
                    detail=type(error).__name__,
                    timing=TraceTiming(
                        queued_at=queued_at,
                        started_at=started_at,
                        first_token_at=first_token_at,
                        completed_at=utc_now_iso(),
                    ),
                    usage=trace_usage,
                    budget=TraceBudget.between(budget_before, run.budget),
                    error=TraceError(type=type(error).__name__),
                )
            )
            raise
        except BaseException:
            run.budget = run.budget.release(unknown_cost=unknown_cost)
            raise

        run.budget = reconciled
        trace_usage = None
        if latest_usage is not None:
            run.usage[0] += latest_usage.prompt_tokens
            run.usage[1] += latest_usage.completion_tokens
            run.cached_tokens += latest_usage.cached_tokens
            self._observe_usage(run)
            trace_usage = TraceUsage(
                prompt_tokens=latest_usage.prompt_tokens,
                completion_tokens=latest_usage.completion_tokens,
                cached_tokens=latest_usage.cached_tokens,
            )
        run.trace.append(
            self._trace_event(
                spec,
                "generated",
                operation="generation",
                status="success",
                attempt=0,
                detail="attempt=0 streamed=true",
                timing=TraceTiming(
                    queued_at=queued_at,
                    started_at=started_at,
                    first_token_at=first_token_at,
                    completed_at=utc_now_iso(),
                ),
                usage=trace_usage,
                budget=TraceBudget.between(
                    budget_before,
                    run.budget,
                    steps_consumed=1,
                    cost_consumed_usd=actual_cost,
                ),
                metadata={"streamed": True},
            )
        )
        run.completion_order.append(spec.name)
        run.final_completions = last_result.completions

    async def run(self, query: str, budget: Budget | None = None) -> ConductorResult:
        run = _RunState(budget=BudgetState(budget=budget or Budget()))
        session = uuid.uuid4().hex[:12]
        await self._run_pending(run, session, query)
        return ConductorResult(
            final_text=self._final_text(run),
            completions=run.final_completions,
            outputs=dict(run.outputs),
            budget_state=run.budget,
            trace=tuple(run.trace),
            usage=tuple(run.usage),
            cached_tokens=run.cached_tokens,
        )

    async def stream(
        self, query: str, budget: Budget | None = None
    ) -> AsyncIterator[ConductorEvent]:
        """Run pre-final DAG waves, then pull final backend deltas directly.

        A verifier attached to the final role would make early deltas
        provisional and therefore impossible to retract in OpenAI SSE.  Such a
        DAG is rejected explicitly; the supported shape verifies drafts before
        an unverified final worker/synthesizer.
        """

        run = _RunState(budget=BudgetState(budget=budget or Budget()))
        session = uuid.uuid4().hex[:12]
        final = self._stream_final_unit()
        await self._run_pending(
            run,
            session,
            query,
            exclude=frozenset({final.name}),
        )
        emitted_text = ""
        try:
            async for event in self._stream_unit(run, session, query, final):
                if event.kind == "delta":
                    emitted_text += event.text
                yield event
        except Exception as error:
            result = ConductorResult(
                final_text=self._final_text(run),
                completions=run.final_completions,
                outputs=dict(run.outputs),
                budget_state=run.budget,
                trace=tuple(run.trace),
                usage=tuple(run.usage),
                cached_tokens=run.cached_tokens,
            )
            raise ConductorStreamError(error, result) from error
        final_text = self._final_text(run)
        if final_text != emitted_text:
            if not final_text.startswith(emitted_text):
                raise RuntimeError("final Conductor result does not extend its streamed deltas")
            yield ConductorEvent(
                kind="delta",
                text=final_text[len(emitted_text) :],
            )
        yield ConductorEvent(
            kind="result",
            result=ConductorResult(
                final_text=final_text,
                completions=run.final_completions,
                outputs=dict(run.outputs),
                budget_state=run.budget,
                trace=tuple(run.trace),
                usage=tuple(run.usage),
                cached_tokens=run.cached_tokens,
            ),
        )
