"""Orchestrator facade: route a query, then dispatch to an engine or the Conductor."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace

from kairyu.async_thread import run_prompt_work, run_serialized_prompt_work
from kairyu.engine.backend import (
    UNLIMITED_OUTPUT_ADMISSION_TOKENS,
    AdmissionUpperBound,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    backend_admission_upper_bound,
    backend_max_model_len,
    backend_supports_chat_template_kwargs,
    backend_supports_prompt_kind,
    prepare_backend_request,
    shutdown_all,
    validate_backend_request_before_prepare,
)
from kairyu.engine.backend import (
    admission_upper_bound as generation_admission_upper_bound,
)
from kairyu.engine.prompt import (
    MultimodalPrompt,
    TemplatedPrompt,
    derive_multimodal_prompt,
)
from kairyu.models.generation import GenerationDefaults
from kairyu.orchestration.budget import Budget, BudgetState
from kairyu.orchestration.conductor import (
    Conductor,
    ConductorStreamError,
    CostModel,
    RoleSpec,
    zero_cost,
)
from kairyu.orchestration.execution import ExecutionBackend, ExecutorDescriptor
from kairyu.orchestration.moa import _build_moa_setup, _MoASetup
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


def _public_multistage_completions(
    completions: tuple[CompletionOutput, ...],
    reasoning_content: str | None = None,
) -> tuple[CompletionOutput, ...]:
    """Keep final text while replacing final-stage reasoning with policy output."""

    return tuple(
        replace(
            completion,
            reasoning_content=reasoning_content,
            reasoning_delta=None,
            reasoning_offset=None,
        )
        for completion in completions
    )


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
    reasoning_content: str | None = None
    # Backend-reported tokens of the client-visible request/completion
    # (issue #496). ``prompt_tokens``/``completion_tokens`` above stay the
    # cumulative internal-call totals (m11 #196); the wire layer maps THESE
    # onto standard OpenAI usage and falls back to the m9 approximation when
    # None. Direct routes report both (one public call); multi-stage routes
    # leave the prompt None because no backend call tokenizes exactly the
    # caller's request.
    public_prompt_tokens: int | None = None
    public_completion_tokens: int | None = None


@dataclass(frozen=True)
class OrchestratorEvent:
    """Streaming event (m11 D1): status keep-alive, token delta, or final."""

    kind: str  # "status" | "reasoning" | "delta" | "error" | "result"
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


@dataclass(frozen=True)
class _IntentRequest:
    engine_key: str
    request: GenerationRequest
    role_name: str | None = None


@dataclass
class _PreparedDispatchState:
    """One alias-shared single-use capability for prepared dispatch."""

    final_requests: tuple[_IntentRequest, ...]
    initial_requests: tuple[_IntentRequest, ...] = ()
    conductor_session: str | None = None
    moa_setup: _MoASetup | None = None
    consumed: bool = False

    def __deepcopy__(self, _memo):
        return self


@dataclass(frozen=True)
class _PreparedOrchestration:
    """Request-local preflight plan that may be consumed by direct dispatch."""

    source_request: str | OrchestrationRequest
    owner_token: object
    call: OrchestrationRequest
    decision: RouteDecision | None
    dispatch: _PreparedDispatchState

    @property
    def consumed(self) -> bool:
        return self.dispatch.consumed

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self


@dataclass(frozen=True)
class _OrchestrationPreparationPlan:
    call: OrchestrationRequest
    decision: RouteDecision
    final_requests: tuple[_IntentRequest, ...]
    internal_requests: tuple[_IntentRequest, ...]
    initial_requests: tuple[_IntentRequest, ...]
    conductor_session: str | None
    moa_setup: _MoASetup | None


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
        owned_engines: Iterable[EngineBackend] | None = None,
        expose_intermediate_outputs: bool = False,
        execution_workers: Mapping[str, ExecutionBackend] | None = None,
        executor_descriptors: Mapping[str, ExecutorDescriptor] | None = None,
    ) -> None:
        if not engines:
            raise ValueError("Orchestrator requires at least one engine")
        if isinstance(shared_prefix, TemplatedPrompt):
            raise ValueError(
                "orchestration shared_prefix cannot be a tokenizer-owned pre-rendered chat prompt"
            )
        self._engines = dict(engines)
        self._owned_engines = tuple(
            {id(engine): engine for engine in (
                self._engines.values() if owned_engines is None else owned_engines
            )}.values()
        )
        self._owner_token = object()
        supplied_descriptors = dict(engine_descriptors or {})
        self._engine_descriptors = {
            name: supplied_descriptors.get(
                name,
                EngineDescriptor(backend_type=type(engine).__name__),
            )
            for name, engine in self._engines.items()
        }
        self._router = router or RuleRouter()
        self._router_gate = asyncio.Lock()
        self._roles = roles or _DEFAULT_ROLES
        self._budget = budget or Budget()
        self._shared_prefix = shared_prefix
        self._sampling_params = sampling_params or SamplingParams(max_tokens=1024)
        self._cost_model = cost_model
        # m11 A4: >0 routes multi_agent through MoA (the deep kairyu-auto-max tier)
        self._moa_samples = moa_samples
        self._expose_intermediate_outputs = expose_intermediate_outputs
        # Serving-layer hook for aggregate per-stage latency attribution
        # (issue #495); set after construction because the deploy builder,
        # not the HTTP app that owns metrics, constructs orchestrators.
        self._stage_observer: Callable[[TraceEvent], None] | None = None
        self._execution_workers = dict(execution_workers or {})
        supplied_executor_descriptors = dict(executor_descriptors or {})
        self._executor_descriptors = {
            name: supplied_executor_descriptors.get(
                name,
                ExecutorDescriptor(backend_type=type(backend).__name__),
            )
            for name, backend in self._execution_workers.items()
        }
        if self._moa_samples == 0:
            # Fail fast on an invalid role DAG (head shape, executor bindings,
            # final-unit overrides) at construction instead of per request.
            Conductor(
                roles=self._roles,
                workers=self._conductor_workers([]),
                shared_prefix=self._shared_prefix,
                sampling_params=self._sampling_params,
                execution_workers=self._execution_workers,
            )

    def generation_defaults_snapshot(
        self,
    ) -> dict[str, GenerationDefaults | None]:
        """Return immutable worker policies without exposing backend ownership."""

        return {
            name: (
                defaults
                if isinstance(
                    defaults := getattr(engine, "generation_defaults", None),
                    GenerationDefaults,
                )
                else None
            )
            for name, engine in self._engines.items()
        }

    async def startup(self) -> None:
        """Eagerly start owned process resources before serving auto models."""

        for engine in self._owned_engines:
            startup = getattr(engine, "startup", None)
            if callable(startup):
                await startup()

    def preview_route(self, prompt: str) -> RouteDecision:
        if isinstance(prompt, TemplatedPrompt):
            raise ValueError(
                "orchestration cannot safely derive prompts from an already rendered chat template"
            )
        preview = getattr(self._router, "preview", None)
        if preview is None:
            raise PreviewNotSupportedError(
                f"router {type(self._router).__name__} does not support preview"
            )
        try:
            return preview(prompt)
        except NotImplementedError as error:
            raise PreviewNotSupportedError(str(error)) from error

    async def preview_route_async(self, prompt: str) -> RouteDecision:
        """Run request-sized preview work without blocking the serving loop."""

        return await run_serialized_prompt_work(
            self._router_gate,
            None,
            self.preview_route,
            prompt,
        )

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
        role_workers = tuple(
            dict.fromkeys(role.worker for role in self._generation_roles())
        )
        if self._moa_samples > 0:
            multi_engines = [self._resolved_engine_descriptor(key) for key in ("tier1", "tier2")]
            multi_mode = "moa"
        else:
            multi_engines = [self._resolved_engine_descriptor(key) for key in role_workers]
            multi_mode = "roles"

        def role_payload(role: RoleSpec) -> dict[str, object]:
            payload: dict[str, object] = {
                "name": role.name,
                "worker": role.worker,
                "role_type": role.role_type,
                "depends_on": list(role.depends_on),
                "verifies": role.verifies,
            }
            if role.sampling is not None:
                payload["sampling"] = {
                    key: value
                    for key, value in asdict(role.sampling).items()
                    if value not in (None, ())
                }
            if role.executor is not None:
                payload["executor"] = {
                    "code_from": list(role.executor.code_from),
                    "tests_from": list(role.executor.tests_from),
                    "mode": role.executor.mode,
                }
            return payload

        head = self._conductor_head_role()
        return {
            "router": router,
            "targets": ["tier1", "tier2", "multi_agent"],
            "configured_engines": {
                name: descriptor.as_dict() for name, descriptor in self._engine_descriptors.items()
            },
            "configured_executors": {
                name: descriptor.as_dict()
                for name, descriptor in self._executor_descriptors.items()
            },
            "target_resolution": {
                "tier1": self._resolved_engine_descriptor("tier1"),
                "tier2": self._resolved_engine_descriptor("tier2"),
                "multi_agent": {
                    "mode": multi_mode,
                    "engines": multi_engines,
                },
            },
            "roles": [role_payload(role) for role in self._roles],
            "stream_head": head.name if head is not None else None,
            "budget": asdict(self._budget),
            "moa_samples": self._moa_samples,
            "internal_max_tokens": self._sampling_params.max_tokens,
            "expose_intermediate_outputs": self._expose_intermediate_outputs,
        }

    async def shutdown(self) -> None:
        await shutdown_all(self._owned_engines, "Orchestrator")

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
            call = request
        else:
            call = default_orchestration_request(request, self._sampling_params)
        if isinstance(call.prompt, TemplatedPrompt):
            raise ValueError(
                "orchestration cannot safely derive prompts from an already rendered chat template"
            )
        return call

    def _route(self, prompt: str) -> RouteDecision:
        """Invoke a potentially stateful synchronous router serially off-loop."""

        return self._router.route(prompt)

    async def _route_async(self, prompt: str) -> RouteDecision:
        """Queue before the executor so a stateful router occupies one lane."""

        return await run_serialized_prompt_work(
            self._router_gate,
            None,
            self._route,
            prompt,
        )

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
        terminal = [
            role
            for role in units
            if role.name not in dependents and role.role_type not in {"head", "executor"}
        ]
        synthesizers = [role for role in terminal if role.role_type == "synthesizer"]
        if not terminal:
            raise ValueError("orchestration requires at least one terminal role")
        return (synthesizers + terminal)[0]

    def _conductor_head_role(self) -> RoleSpec | None:
        return next((role for role in self._roles if role.role_type == "head"), None)

    def _final_engine_keys(self, decision: RouteDecision | None) -> tuple[str, ...]:
        if decision is None:
            keys = {"tier1", "tier2", self._conductor_final_role().worker}
        elif decision.target == "multi_agent":
            keys = {"tier2" if self._moa_samples > 0 else self._conductor_final_role().worker}
        else:
            keys = {decision.target}
        fallback = next(iter(self._engines))
        return tuple(
            dict.fromkeys(key if key in self._engines else fallback for key in sorted(keys))
        )

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
                for role in self._generation_roles()
                if role.name != final_role.name
            }
        fallback = next(iter(self._engines))
        return tuple(
            dict.fromkeys(key if key in self._engines else fallback for key in sorted(keys))
        )

    def _validate_call(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> None:
        if call.multimodal_prompt is not None:
            if self._moa_samples > 0 and (
                decision is None or decision.target == "multi_agent"
            ):
                raise ValueError("multimodal orchestration does not support MoA sampling")
            if decision is not None and decision.target != "multi_agent":
                keys = self._final_engine_keys(decision)
                if not all(
                    backend_supports_prompt_kind(self._engines[key], "multimodal")
                    for key in keys
                ):
                    raise ValueError(
                        f"orchestration route {decision.target!r} does not support image inputs"
                    )
                media_keys = keys
            else:
                fallback = next(iter(self._engines))
                role_keys = {
                    role.worker if role.worker in self._engines else fallback
                    for role in self._generation_roles()
                }
                if not any(
                    backend_supports_prompt_kind(self._engines[key], "multimodal")
                    for key in role_keys
                ):
                    raise ValueError(
                        "multimodal orchestration requires at least one image-capable worker"
                    )
                media_keys = tuple(
                    key
                    for key in role_keys
                    if backend_supports_prompt_kind(
                        self._engines[key],
                        "multimodal",
                    )
                )
            if call.chat_template_kwargs is not None and not all(
                backend_supports_chat_template_kwargs(
                    self._engines[key],
                    frozenset(call.chat_template_kwargs),
                )
                for key in media_keys
            ):
                raise ValueError(
                    "image-capable orchestration workers do not all support "
                    "chat_template_kwargs"
                )
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

    def _final_intent_requests(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> tuple[_IntentRequest, ...]:
        requests: list[_IntentRequest] = []
        for key in self._final_engine_keys(decision):
            prompt = self._engine_prompt(
                key,
                call,
                f"{self._shared_prefix}{call.prompt}",
            )
            requests.append(
                _IntentRequest(
                    key,
                    GenerationRequest(
                        request_id=f"preflight-{key}",
                        prompt=prompt,
                        sampling_params=call.sampling_params,
                        tools=call.tools,
                        tool_choice=call.tool_choice,
                        tools_in_prompt=call.tools_in_prompt,
                        parallel_tool_calls=call.parallel_tool_calls,
                        tool_call_protocol=call.tool_call_protocol,
                        reasoning_effort=call.reasoning_effort,
                        chat_template_kwargs=(
                            call.chat_template_kwargs
                            if isinstance(prompt, MultimodalPrompt)
                            else None
                        ),
                    ),
                )
            )
        return tuple(requests)

    def _direct_generation_request(
        self,
        call: OrchestrationRequest,
        request_id: str,
        engine_key: str,
    ) -> GenerationRequest:
        """Build a direct request without copying its prompt on the loop."""

        prompt = self._engine_prompt(
            engine_key,
            call,
            f"{self._shared_prefix}{call.prompt}",
        )
        return GenerationRequest(
            request_id=request_id,
            prompt=prompt,
            sampling_params=call.sampling_params,
            # No random session: route by shared-prefix overlap and load.
            cache_hint=None,
            tools=call.tools,
            tool_choice=call.tool_choice,
            tools_in_prompt=call.tools_in_prompt,
            parallel_tool_calls=call.parallel_tool_calls,
            tool_call_protocol=call.tool_call_protocol,
            reasoning_effort=call.reasoning_effort,
            chat_template_kwargs=(
                call.chat_template_kwargs
                if isinstance(prompt, MultimodalPrompt)
                else None
            ),
        )

    def _engine_prompt(
        self,
        engine_key: str,
        call: OrchestrationRequest,
        text: str,
    ):
        if call.multimodal_prompt is None or not backend_supports_prompt_kind(
            self._engines[engine_key],
            "multimodal",
        ):
            return text
        return derive_multimodal_prompt(call.multimodal_prompt, text)

    def _internal_intent_requests(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> tuple[_IntentRequest, ...]:
        sampling_params = self._internal_sampling_params(call)
        if self._moa_samples > 0 and (decision is None or decision.target == "multi_agent"):
            setup = _build_moa_setup(
                call.prompt,
                self._moa_samples,
                sampling_params,
                call.sampling_params,
                self._shared_prefix,
                session="validation-moa",
            )
            key = self._internal_engine_keys(decision)[0]
            return tuple(_IntentRequest(key, request) for request in setup.proposal_requests)
        requests: list[_IntentRequest] = []
        for key in self._internal_engine_keys(decision):
            prompt = self._engine_prompt(
                key,
                call,
                f"{self._shared_prefix}{call.prompt}",
            )
            requests.append(
                _IntentRequest(
                    key,
                    GenerationRequest(
                        request_id=f"preflight-internal-{key}",
                        prompt=prompt,
                        sampling_params=sampling_params,
                        reasoning_effort=call.reasoning_effort,
                        chat_template_kwargs=(
                            call.chat_template_kwargs
                            if isinstance(prompt, MultimodalPrompt)
                            else None
                        ),
                    ),
                )
            )
        return tuple(requests)

    def _initial_role_intent_requests(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
        *,
        request_id_suffix: str,
    ) -> tuple[_IntentRequest, ...]:
        """Return exact first-wave Conductor prompts for role orchestration."""

        if self._moa_samples > 0 or (decision is not None and decision.target != "multi_agent"):
            return ()
        conductor = self._new_conductor(call, [])
        fallback = next(iter(self._engines))
        return tuple(
            _IntentRequest(
                spec.worker if spec.worker in self._engines else fallback,
                request,
                spec.name,
            )
            for spec, request in conductor.initial_requests(
                call.prompt,
                request_id_suffix=request_id_suffix,
            )
        )

    @staticmethod
    def _intent_failure_message(kind: str, failures: list[str]) -> str:
        return f"{kind} orchestration intent is unsupported: " + "; ".join(failures)

    def _validate_intent_requests(
        self,
        kind: str,
        requests: tuple[_IntentRequest, ...],
        *,
        before_prepare: bool = False,
    ) -> None:
        failures: list[str] = []
        for intent in requests:
            try:
                engine = self._engines[intent.engine_key]
                if before_prepare:
                    validate_backend_request_before_prepare(
                        engine,
                        intent.request,
                    )
                else:
                    validate = getattr(engine, "validate_request", None)
                    if validate is None:
                        continue
                    validate(intent.request)
            except ValueError as error:
                failures.append(f"{intent.engine_key}: {error}")
        if failures:
            raise ValueError(self._intent_failure_message(kind, failures))

    def _validate_final_intent(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> None:
        self._validate_intent_requests(
            "final",
            self._final_intent_requests(call, decision),
        )

    def _validate_internal_intent(
        self,
        call: OrchestrationRequest,
        decision: RouteDecision | None,
    ) -> None:
        self._validate_intent_requests(
            "internal",
            self._internal_intent_requests(call, decision),
        )

    def _preflight_call(
        self,
        request: str | OrchestrationRequest,
    ) -> tuple[OrchestrationRequest, RouteDecision | None]:
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
        return call, decision

    def validate_request(self, request: str | OrchestrationRequest) -> None:
        """Preflight capability checks without dispatching generation."""

        call, decision = self._preflight_call(request)
        self._validate_final_intent(call, decision)
        self._validate_internal_intent(call, decision)
        self._validate_intent_requests(
            "initial",
            self._initial_role_intent_requests(
                call,
                decision,
                request_id_suffix="validation",
            ),
        )

    def validate_request_before_prepare(
        self,
        request: str | OrchestrationRequest,
    ) -> None:
        """Keep only bounded orchestration shape checks on the event loop."""

        self._request(request)

    def _build_preparation_plan(
        self,
        call: OrchestrationRequest,
        suffix: str,
        decision: RouteDecision,
    ) -> _OrchestrationPreparationPlan:
        """Route, render, and validate request-sized intents off-loop."""

        self._validate_call(call, decision)
        final_requests = self._final_intent_requests(call, decision)
        if decision.target != "multi_agent":
            final_requests = tuple(
                _IntentRequest(
                    intent.engine_key,
                    replace(
                        intent.request,
                        request_id=f"direct-{suffix}-{intent.engine_key}",
                    ),
                )
                for intent in final_requests
            )
        else:
            final_requests = tuple(
                _IntentRequest(
                    intent.engine_key,
                    replace(
                        intent.request,
                        request_id=f"{intent.request.request_id}-{suffix}",
                    ),
                )
                for intent in final_requests
            )
        moa_setup = None
        if decision.target == "multi_agent" and self._moa_samples > 0:
            moa_setup = _build_moa_setup(
                call.prompt,
                self._moa_samples,
                self._internal_sampling_params(call),
                call.sampling_params,
                self._shared_prefix,
                session=f"preflight-moa-{suffix}",
            )
            proposal_key = self._internal_engine_keys(decision)[0]
            internal_requests = tuple(
                _IntentRequest(proposal_key, request) for request in moa_setup.proposal_requests
            )
        else:
            internal_requests = tuple(
                _IntentRequest(
                    intent.engine_key,
                    replace(
                        intent.request,
                        request_id=f"{intent.request.request_id}-{suffix}",
                    ),
                )
                for intent in self._internal_intent_requests(call, decision)
            )
        initial_requests = self._initial_role_intent_requests(
            call,
            decision,
            request_id_suffix=suffix,
        )
        return _OrchestrationPreparationPlan(
            call=call,
            decision=decision,
            final_requests=final_requests,
            internal_requests=internal_requests,
            initial_requests=initial_requests,
            conductor_session=(Conductor.preflight_session(suffix) if initial_requests else None),
            moa_setup=moa_setup,
        )

    async def prepare_request(
        self,
        request: str | OrchestrationRequest,
    ) -> _PreparedOrchestration:
        """Run semantic and backend-owned prompt preflight before dispatch."""

        call = self._request(request)
        suffix = uuid.uuid4().hex[:12]
        decision = await self._route_async(call.prompt)
        plan = await run_prompt_work(
            self._build_preparation_plan,
            call,
            suffix,
            decision,
        )
        # Backend state and ReplicaPool membership are event-loop owned.  The
        # request-sized strings are already built, so these bounded fast hooks
        # stay atomic with respect to membership mutation.
        self._validate_intent_requests(
            "final",
            plan.final_requests,
            before_prepare=True,
        )
        self._validate_intent_requests(
            "internal",
            plan.internal_requests,
            before_prepare=True,
        )
        self._validate_intent_requests(
            "initial",
            plan.initial_requests,
            before_prepare=True,
        )

        failure_groups: list[str] = []
        for kind, intents in (
            ("final", plan.final_requests),
            ("internal", plan.internal_requests),
            ("initial", plan.initial_requests),
        ):
            failures: list[str] = []
            for intent in intents:
                try:
                    await prepare_backend_request(
                        self._engines[intent.engine_key],
                        intent.request,
                    )
                except ValueError as error:
                    failures.append(f"{intent.engine_key}: {error}")
            if failures:
                failure_groups.append(self._intent_failure_message(kind, failures))
        if failure_groups:
            raise ValueError("; ".join(failure_groups))
        return _PreparedOrchestration(
            source_request=request,
            owner_token=self._owner_token,
            call=plan.call,
            decision=plan.decision,
            dispatch=_PreparedDispatchState(
                final_requests=plan.final_requests,
                initial_requests=plan.initial_requests,
                conductor_session=plan.conductor_session,
                moa_setup=plan.moa_setup,
            ),
        )

    def _call_with_prepared_handle(
        self,
        request: str | OrchestrationRequest,
        prepared: _PreparedOrchestration | None,
    ) -> OrchestrationRequest:
        if (
            prepared is not None
            and prepared.owner_token is self._owner_token
            and (
                request is prepared.call
                or request is prepared.source_request
                or (
                    type(request) is str
                    and type(prepared.source_request) is str
                    and request == prepared.source_request
                )
            )
        ):
            return prepared.call
        return self._request(request)

    def _take_prepared_decision(
        self,
        prepared: _PreparedOrchestration | None,
        call: OrchestrationRequest,
    ) -> RouteDecision | None:
        """Consume the route decision bound by exact-call async preflight."""

        if (
            prepared is None
            or prepared.owner_token is not self._owner_token
            or prepared.call is not call
            or prepared.dispatch.consumed
        ):
            return None
        prepared.dispatch.consumed = True
        assert prepared.decision is not None
        if prepared.decision.target == "multi_agent":
            # Multi-agent prompts evolve after each stage, so synthetic final
            # preparation is validation-only and cannot be dispatched exactly.
            prepared.dispatch.final_requests = ()
        return prepared.decision

    def _prepared_direct_request(
        self,
        prepared: _PreparedOrchestration | None,
        call: OrchestrationRequest,
        decision: RouteDecision,
        engine_name: str,
    ) -> GenerationRequest | None:
        """Reuse exact direct preflight work when preview and route agree."""

        if (
            prepared is None
            or prepared.owner_token is not self._owner_token
            or prepared.call is not call
            or prepared.decision is None
            or prepared.decision.target != decision.target
            or decision.target == "multi_agent"
        ):
            return None
        request = next(
            (
                intent.request
                for intent in prepared.dispatch.final_requests
                if intent.engine_key == engine_name
            ),
            None,
        )
        # The request ID is unique but backend active-ID contracts still make
        # the handle single-use. A repeated caller safely falls back to fresh
        # preparation instead of racing the same GenerationRequest.
        prepared.dispatch.final_requests = ()
        return request

    def _take_prepared_initial_roles(
        self,
        prepared: _PreparedOrchestration | None,
        call: OrchestrationRequest,
        decision: RouteDecision,
    ) -> tuple[str | None, dict[str, GenerationRequest] | None]:
        """Consume exact dependency-free role requests bound by preflight."""

        if (
            prepared is None
            or prepared.owner_token is not self._owner_token
            or prepared.call is not call
            or not prepared.dispatch.consumed
            or prepared.decision is None
            or prepared.decision.target != decision.target
            or decision.target != "multi_agent"
            or not prepared.dispatch.initial_requests
            or prepared.dispatch.conductor_session is None
        ):
            return None, None
        requests = {
            intent.role_name: intent.request
            for intent in prepared.dispatch.initial_requests
            if intent.role_name is not None
        }
        session = prepared.dispatch.conductor_session
        prepared.dispatch.initial_requests = ()
        prepared.dispatch.conductor_session = None
        return session, requests

    def _take_prepared_moa_setup(
        self,
        prepared: _PreparedOrchestration | None,
        call: OrchestrationRequest,
        decision: RouteDecision,
    ) -> _MoASetup | None:
        """Consume the exact proposal request set prepared for one MoA run."""

        if (
            prepared is None
            or prepared.owner_token is not self._owner_token
            or prepared.call is not call
            or not prepared.dispatch.consumed
            or prepared.decision is None
            or prepared.decision.target != decision.target
            or decision.target != "multi_agent"
        ):
            return None
        setup = prepared.dispatch.moa_setup
        prepared.dispatch.moa_setup = None
        return setup

    def set_stage_observer(
        self,
        observer: Callable[[TraceEvent], None] | None,
    ) -> None:
        """Install the per-stage trace-event hook used for latency metrics."""

        self._stage_observer = observer

    @property
    def max_model_len(self) -> int | None:
        """Largest engine context this orchestration can serve publicly."""

        lengths = [
            length
            for engine in self._engines.values()
            if (length := backend_max_model_len(engine)) is not None
        ]
        return max(lengths, default=None)

    def admission_upper_bound(
        self,
        request: str | OrchestrationRequest,
    ) -> AdmissionUpperBound:
        """Worst-route compute ceiling for one public AUTO request.

        The bound belongs here rather than in the HTTP adapter because this
        object owns route fan-out, refine depth, MoA samples, and internal
        generation limits.  It includes every possible step plus triangular
        re-ingestion of preceding generated output into later prompts.
        """

        call = self._request(request)
        internal = self._internal_sampling_params(call)
        if internal.max_tokens is None:
            raise ValueError("AUTO tenant admission requires a finite internal limit")
        # An omitted public limit follows the OpenAI chat contract (bounded by
        # the serving model's context); reserve the largest engine context —
        # or the documented constant — instead of rejecting the request.
        public_output_fallback = (
            max(
                (
                    getattr(engine, "max_model_len", None) or 0
                    for engine in self._engines.values()
                ),
                default=0,
            )
            or UNLIMITED_OUTPUT_ADMISSION_TOKENS
        )
        steps = max(1, self._budget.max_steps, self._moa_samples + 1)
        candidates = max(
            call.sampling_params.n,
            call.sampling_params.best_of or call.sampling_params.n,
        )
        largest_role_prompt = max(
            self._roles,
            key=lambda role: len(role.prompt.encode("utf-8")),
            default=None,
        )
        role_prompt = largest_role_prompt.prompt if largest_role_prompt else ""
        role_bytes = len(role_prompt.encode("utf-8"))
        supplied_bytes = len(f"{self._shared_prefix}{call.prompt}".encode())
        stage_prompt = max(1, supplied_bytes + role_bytes + 256)
        internal_output = internal.max_tokens
        head = self._conductor_head_role()
        if head is not None and head.sampling is not None and head.sampling.max_tokens:
            # The head streams public output whose role cap may exceed the
            # internal ceiling; keep the bound conservative.
            internal_output = max(internal_output, head.sampling.max_tokens)
        private_steps = max(0, steps - 1)
        private_work = (
            private_steps * (stage_prompt + internal_output)
            + internal_output * private_steps * (private_steps - 1) // 2
        )
        final_request_bound = generation_admission_upper_bound(
            GenerationRequest(
                request_id="auto-admission",
                prompt=f"{self._shared_prefix}{call.prompt}{role_prompt}",
                sampling_params=call.sampling_params,
                tools=call.tools,
                tool_choice=call.tool_choice,
                tools_in_prompt=call.tools_in_prompt,
                parallel_tool_calls=call.parallel_tool_calls,
                tool_call_protocol=call.tool_call_protocol,
                reasoning_effort=call.reasoning_effort,
            ),
            fallback_output_tokens=public_output_fallback,
        )
        media_stage_bounds = [
            backend_admission_upper_bound(
                self._engines[intent.engine_key],
                intent.request,
            ).tokens
            for intent in self._internal_intent_requests(call, None)
            if call.multimodal_prompt is not None
            and backend_supports_prompt_kind(
                self._engines[intent.engine_key],
                "multimodal",
            )
        ]
        if media_stage_bounds:
            private_work = max(
                private_work,
                private_steps * max(media_stage_bounds),
            )
        # The last stage can ingest every preceding private output. Candidate
        # fan-out duplicates that prefill just as it duplicates the supplied
        # prompt, tools, response schema, and decode ceiling.
        final_reingestion = candidates * internal_output * private_steps
        return AdmissionUpperBound(
            tokens=private_work + final_request_bound.tokens + final_reingestion,
            refundable_on_exact_usage=False,
        )

    async def admission_upper_bound_async(
        self,
        request: str | OrchestrationRequest,
    ) -> AdmissionUpperBound:
        """Serialize tool/response intent on the bounded prompt CPU lane."""

        return await run_prompt_work(self.admission_upper_bound, request)

    def _generation_roles(self) -> tuple[RoleSpec, ...]:
        return tuple(role for role in self._roles if role.role_type != "executor")

    def _conductor_workers(self, notes: list[str]) -> dict[str, EngineBackend]:
        needed = {role.worker for role in self._generation_roles()}
        return {name: self._resolve_engine(name, notes) for name in needed}

    @staticmethod
    def _conversation_affinity_key(prompt: object) -> str | None:
        """Stable KV-affinity key for consecutive turns of one conversation.

        Agent loops resend the same conversation with new turns appended, so
        hashing a fixed-length head (long enough to pass the constant L2
        preamble and reach conversation-specific text) keeps every turn on
        the replica that already holds the warm prefix, instead of the
        per-request random session defeating KV reuse (issue #495).
        """

        if not isinstance(prompt, str) or not prompt:
            return None
        return "conv-" + hashlib.sha256(prompt[:2048].encode()).hexdigest()[:16]

    def _new_conductor(
        self,
        call: OrchestrationRequest,
        notes: list[str],
        *,
        usage_observer: Callable[[GenerationUsage], None] | None = None,
    ) -> Conductor:
        """Construct the one exact role execution contract used by preflight/run."""

        return Conductor(
            roles=self._roles,
            workers=self._conductor_workers(notes),
            shared_prefix=self._shared_prefix,
            sampling_params=self._internal_sampling_params(call),
            final_sampling_params=call.sampling_params,
            final_tools=call.tools,
            final_tool_choice=call.tool_choice,
            final_tools_in_prompt=call.tools_in_prompt,
            final_structured_format_in_prompt=call.structured_format_in_prompt,
            final_parallel_tool_calls=call.parallel_tool_calls,
            final_tool_call_protocol=call.tool_call_protocol,
            cost_model=self._cost_model,
            worker_trace=self._conductor_worker_trace(),
            usage_observer=usage_observer,
            stage_observer=self._stage_observer,
            affinity_key=self._conversation_affinity_key(call.prompt),
            expose_intermediate_outputs=self._expose_intermediate_outputs,
            multimodal_prompt=call.multimodal_prompt,
            chat_template_kwargs=call.chat_template_kwargs,
            execution_workers=self._execution_workers,
        )

    def _conductor_worker_trace(self) -> dict[str, WorkerTraceIdentity]:
        fallback_name = next(iter(self._engines))
        needed = {role.worker for role in self._generation_roles()}
        identities = {}
        for worker in needed:
            engine = worker if worker in self._engines else fallback_name
            descriptor = self._engine_descriptors[engine]
            identities[worker] = WorkerTraceIdentity(
                engine=engine,
                model=descriptor.model,
            )
        for role in self._roles:
            if role.role_type == "executor" and role.worker not in identities:
                identities[role.worker] = WorkerTraceIdentity(engine=role.worker)
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
        self,
        call: OrchestrationRequest,
        tier: str,
        notes: list[str],
        *,
        prepared_request: GenerationRequest | None = None,
    ) -> tuple[GenerationResult, tuple[int, int, int], TraceEvent]:
        queued_at = utc_now_iso()
        engine_name = self._resolve_engine_name(tier, notes)
        engine = self._engines[engine_name]
        descriptor = self._engine_descriptors[engine_name]
        request = prepared_request
        if request is None:
            request = await run_prompt_work(
                self._direct_generation_request,
                call,
                f"direct-{uuid.uuid4().hex[:12]}",
                engine_name,
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
        *,
        prepared: _PreparedOrchestration | None = None,
    ) -> OrchestratorResult:
        call = self._call_with_prepared_handle(request, prepared)
        query = call.prompt
        request_id = f"orch-{uuid.uuid4().hex[:16]}"
        trace_started_at = utc_now_iso()
        route_started_at = utc_now_iso()
        from kairyu.telemetry import traced_span

        with traced_span(
            "kairyu.route",
            {"kairyu.request_id": request_id},
        ) as route_span:
            decision = self._take_prepared_decision(prepared, call)
            if decision is None:
                decision = await self._route_async(query)
            self._validate_call(call, decision)
            if route_span is not None:
                route_span.set_attribute("kairyu.route.target", decision.target)
                route_span.set_attribute("kairyu.route.confidence", decision.confidence)
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
            reasoning_content: str | None = None,
            public_prompt_tokens: int | None = None,
            public_completion_tokens: int | None = None,
        ) -> OrchestratorResult:
            return OrchestratorResult(
                text=text,
                route=decision,
                trace=tuple(notes),
                completions=completions,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                reasoning_content=reasoning_content,
                public_prompt_tokens=public_prompt_tokens,
                public_completion_tokens=public_completion_tokens,
                structured_trace=StructuredTrace(
                    request_id=request_id,
                    started_at=trace_started_at,
                    completed_at=utc_now_iso(),
                    events=tuple(trace_events),
                ),
            )

        if decision.target == "multi_agent":
            if self._moa_samples > 0:  # m11 A4: the deep tier's MoA route
                from kairyu.orchestration.moa import (
                    MoAExecutionError,
                    _reuse_prepared_moa_setup,
                    run_moa,
                )

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
                prepared_moa = self._take_prepared_moa_setup(
                    prepared,
                    call,
                    decision,
                )
                try:
                    with _reuse_prepared_moa_setup(prepared_moa):
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
                            final_parallel_tool_calls=call.parallel_tool_calls,
                            final_tool_call_protocol=call.tool_call_protocol,
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
                            parallel_tool_calls=call.parallel_tool_calls,
                            tool_call_protocol=call.tool_call_protocol,
                            reasoning_effort=call.reasoning_effort,
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
                    completions=_public_multistage_completions(
                        moa.completions
                        or (
                            CompletionOutput(
                                index=0,
                                text=moa.final_text,
                                token_ids=(),
                                finish_reason="stop",
                            ),
                        )
                    ),
                    prompt_tokens=moa.usage[0],
                    completion_tokens=moa.usage[1],
                    cached_tokens=moa.cached_tokens,
                )
            conductor = self._new_conductor(call, notes)
            conductor_session, initial_requests = self._take_prepared_initial_roles(
                prepared, call, decision
            )
            result = await conductor.run(
                query,
                budget=self._budget,
                session=conductor_session,
                prepared_initial_requests=initial_requests,
            )
            notes.extend(f"{event.node}: {event.kind} {event.detail}" for event in result.trace)
            trace_events.extend(result.trace)
            if not result.final_unit_ok and not result.final_text:
                # The selected final unit failed or produced no public text
                # after its retry; publishing an internal stage or an empty
                # "stop" would be a silent lie to the caller (issue #496).
                raise OrchestratorExecutionError(
                    RuntimeError("orchestration final unit produced no public output"),
                    result_with_trace(
                        text="",
                        prompt_tokens=result.usage[0],
                        completion_tokens=result.usage[1],
                        cached_tokens=result.cached_tokens,
                        reasoning_content=result.reasoning_content,
                    ),
                )
            return result_with_trace(
                text=result.final_text,
                completions=_public_multistage_completions(
                    result.completions,
                    result.reasoning_content,
                ),
                prompt_tokens=result.usage[0],
                completion_tokens=result.usage[1],
                cached_tokens=result.cached_tokens,
                reasoning_content=result.reasoning_content,
                public_completion_tokens=result.public_completion_tokens,
            )
        try:
            engine_name = (
                decision.target if decision.target in self._engines else next(iter(self._engines))
            )
            direct_result, usage, direct_event = await self._run_direct(
                call,
                decision.target,
                notes,
                prepared_request=self._prepared_direct_request(
                    prepared,
                    call,
                    decision,
                    engine_name,
                ),
            )
        except _DirectExecutionError as error:
            trace_events.append(error.event)
            if (
                decision.target == "tier1"
                and "tier2" in self._engines
                and (
                    call.multimodal_prompt is None
                    or backend_supports_prompt_kind(
                        self._engines["tier2"],
                        "multimodal",
                    )
                )
            ):
                notes.append("failover: tier1 failed; retrying once on tier2")
                try:
                    direct_result, usage, direct_event = await self._run_direct(
                        call,
                        "tier2",
                        notes,
                    )
                except _DirectExecutionError as failover_error:
                    trace_events.append(failover_error.event)
                    raise OrchestratorExecutionError(
                        failover_error.cause,
                        result_with_trace(text=""),
                    ) from failover_error.cause
            else:
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
            # A direct route is one public call: its reported usage IS the
            # client-visible view (0 = unreported → m9 fallback).
            public_prompt_tokens=usage[0] or None,
            public_completion_tokens=usage[1] or None,
        )

    async def run_chat(
        self,
        request: str | OrchestrationRequest,
        stream: bool = False,
        usage_observer: Callable[[GenerationUsage], None] | None = None,
        *,
        prepared: _PreparedOrchestration | None = None,
    ):
        """The m11 D1 surface: pre-rendered prompt in (A4), events out.

        Non-stream: returns OrchestratorResult (same as run()). Stream:
        async-yields OrchestratorEvent — status keep-alives while pre-final
        stages run, token deltas pulled directly from every route's FINAL
        worker/synthesizer, then one result event.
        """
        if not stream:
            return await self.run(request, prepared=prepared)
        return self._run_chat_stream(
            self._call_with_prepared_handle(request, prepared),
            usage_observer=usage_observer,
            prepared=prepared,
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

    async def _with_keepalives(self, stream) -> AsyncIterator[object | None]:
        """Emit timer sentinels whenever the next event is slow to arrive.

        The head/incremental-reasoning Conductor stream produces events
        throughout the DAG's lifetime with potentially long gaps between
        stages, so every ``anext`` is task-backed (EO-D8).  The per-event task
        cost is microseconds against a multi-millisecond token cadence; the
        latency-critical direct route keeps its pull-through loop.
        """

        iterator = stream.__aiter__()
        try:
            while True:
                upcoming = asyncio.ensure_future(anext(iterator))
                try:
                    while not upcoming.done():
                        done, _ = await asyncio.wait(
                            {upcoming},
                            timeout=_KEEPALIVE_INTERVAL_S,
                        )
                        if not done:
                            yield None
                    try:
                        yield upcoming.result()
                    except StopAsyncIteration:
                        return
                finally:
                    if not upcoming.done():
                        upcoming.cancel()
                        await asyncio.gather(upcoming, return_exceptions=True)
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def _run_chat_stream(
        self,
        call: OrchestrationRequest,
        *,
        usage_observer: Callable[[GenerationUsage], None] | None,
        prepared: _PreparedOrchestration | None,
    ):
        prompt = call.prompt
        request_id = f"orch-{uuid.uuid4().hex[:16]}"
        trace_started_at = utc_now_iso()
        route_started_at = utc_now_iso()
        from kairyu.telemetry import traced_span

        with traced_span(
            "kairyu.route",
            {"kairyu.request_id": request_id},
        ) as route_span:
            decision = self._take_prepared_decision(prepared, call)
            if decision is None:
                decision = await self._route_async(prompt)
            self._validate_call(call, decision)
            if route_span is not None:
                route_span.set_attribute("kairyu.route.target", decision.target)
                route_span.set_attribute("kairyu.route.confidence", decision.confidence)
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
            reasoning_content: str | None = None,
            public_prompt_tokens: int | None = None,
            public_completion_tokens: int | None = None,
        ) -> OrchestratorResult:
            return OrchestratorResult(
                text=text,
                route=decision,
                trace=tuple(notes),
                completions=completions,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                reasoning_content=reasoning_content,
                public_prompt_tokens=public_prompt_tokens,
                public_completion_tokens=public_completion_tokens,
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
            request = self._prepared_direct_request(
                prepared,
                call,
                decision,
                engine_name,
            )
            if request is None:
                request = await run_prompt_work(
                    self._direct_generation_request,
                    call,
                    f"direct-{uuid.uuid4().hex[:12]}",
                    engine_name,
                )
            last = None
            latest_usage = None
            started_at = utc_now_iso()
            first_token_at = None
            emitted = 0
            text_parts: list[str] = []
            try:
                async for partial in engine.stream(request):
                    last = partial
                    if partial.usage is not None:
                        latest_usage = partial.usage
                        if usage_observer is not None:
                            usage_observer(latest_usage)
                    completion = partial.completions[0] if partial.completions else None
                    delta, emitted = (
                        completion.delta_after(emitted)
                        if completion is not None
                        else ("", emitted)
                    )
                    text_parts.append(delta)
                    if first_token_at is None and partial.completions:
                        first_token_at = utc_now_iso()
                    yield OrchestratorEvent(
                        kind="delta",
                        text=delta,
                        completions=partial.completions,
                    )
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
                if (
                    decision.target == "tier1"
                    and "tier2" in self._engines
                    and not "".join(text_parts)
                    and (
                        call.multimodal_prompt is None
                        or backend_supports_prompt_kind(
                            self._engines["tier2"],
                            "multimodal",
                        )
                    )
                ):
                    notes.append("failover: tier1 failed before output; retrying once on tier2")
                    failover_queued_at = utc_now_iso()
                    failover_engine = self._engines["tier2"]
                    failover_descriptor = self._engine_descriptors["tier2"]
                    failover_request = await run_prompt_work(
                        self._direct_generation_request,
                        call,
                        f"direct-{uuid.uuid4().hex[:12]}",
                        "tier2",
                    )
                    failover_last = None
                    failover_usage = None
                    failover_started_at = utc_now_iso()
                    failover_first_token_at = None
                    failover_emitted = 0
                    failover_text_parts: list[str] = []
                    try:
                        async for partial in failover_engine.stream(failover_request):
                            failover_last = partial
                            if partial.usage is not None:
                                failover_usage = partial.usage
                                if usage_observer is not None:
                                    usage_observer(failover_usage)
                            completion = (
                                partial.completions[0] if partial.completions else None
                            )
                            delta, failover_emitted = (
                                completion.delta_after(failover_emitted)
                                if completion is not None
                                else ("", failover_emitted)
                            )
                            failover_text_parts.append(delta)
                            if failover_first_token_at is None and partial.completions:
                                failover_first_token_at = utc_now_iso()
                            yield OrchestratorEvent(
                                kind="delta",
                                text=delta,
                                completions=partial.completions,
                            )
                        if failover_last is None:
                            raise RuntimeError("tier2 failover stream produced no result")
                    except Exception as failover_error:
                        failover_trace_usage = (
                            TraceUsage(
                                prompt_tokens=failover_usage.prompt_tokens,
                                completion_tokens=failover_usage.completion_tokens,
                                cached_tokens=failover_usage.cached_tokens,
                            )
                            if failover_usage is not None
                            else None
                        )
                        trace_events.append(
                            TraceEvent(
                                node="tier2",
                                kind="failed",
                                operation="generation",
                                status="failed",
                                role="direct",
                                worker="tier2",
                                engine="tier2",
                                model=failover_descriptor.model,
                                timing=TraceTiming(
                                    queued_at=failover_queued_at,
                                    started_at=failover_started_at,
                                    first_token_at=failover_first_token_at,
                                    completed_at=utc_now_iso(),
                                ),
                                usage=failover_trace_usage,
                                metadata={"streamed": True, "failover": True},
                                error=TraceError(type=type(failover_error).__name__),
                            )
                        )
                        usage = failover_usage or latest_usage or GenerationUsage()
                        partial_result = result_with_trace(
                            text="".join(failover_text_parts),
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            cached_tokens=usage.cached_tokens,
                        )
                        yield OrchestratorEvent(
                            kind="error",
                            result=partial_result,
                            error_type=type(failover_error).__name__,
                        )
                        return
                    failover_trace_usage = (
                        TraceUsage(
                            prompt_tokens=failover_usage.prompt_tokens,
                            completion_tokens=failover_usage.completion_tokens,
                            cached_tokens=failover_usage.cached_tokens,
                        )
                        if failover_usage is not None
                        else None
                    )
                    trace_events.append(
                        TraceEvent(
                            node="tier2",
                            kind="generated",
                            operation="generation",
                            status="success",
                            role="direct",
                            worker="tier2",
                            engine="tier2",
                            model=failover_descriptor.model,
                            timing=TraceTiming(
                                queued_at=failover_queued_at,
                                started_at=failover_started_at,
                                first_token_at=failover_first_token_at,
                                completed_at=utc_now_iso(),
                            ),
                            usage=failover_trace_usage,
                            metadata={"streamed": True, "failover": True},
                        )
                    )
                    usage = failover_usage or GenerationUsage()
                    yield OrchestratorEvent(
                        kind="result",
                        result=result_with_trace(
                            text=(
                                failover_last.text
                                or "".join(failover_text_parts)
                            ),
                            completions=failover_last.completions,
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            cached_tokens=usage.cached_tokens,
                            public_prompt_tokens=usage.prompt_tokens or None,
                            public_completion_tokens=usage.completion_tokens or None,
                        ),
                    )
                    return
                usage = latest_usage or GenerationUsage()
                partial_result = result_with_trace(
                    text="".join(text_parts),
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
                    text=(
                        (last.text or "".join(text_parts))
                        if last is not None
                        else ""
                    ),
                    completions=last.completions if last is not None else (),
                    prompt_tokens=usage[0],
                    completion_tokens=usage[1],
                    cached_tokens=usage[2],
                    public_prompt_tokens=usage[0] or None,
                    public_completion_tokens=usage[1] or None,
                ),
            )
            return

        # Multi-stage routes emit comments while the pre-final DAG/proposals and
        # first final token are pending.  Final deltas then pull through from
        # Conductor/MoA without a queue bridge.
        yield OrchestratorEvent(kind="status", text=f"routing: {decision.target}")
        if self._moa_samples > 0:
            from kairyu.orchestration.moa import (
                MoAExecutionError,
                _reuse_prepared_moa_setup,
                stream_moa,
            )

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
            prepared_moa = self._take_prepared_moa_setup(
                prepared,
                call,
                decision,
            )

            def observe_moa_usage(usage: GenerationUsage) -> None:
                nonlocal latest_usage
                latest_usage = usage
                if usage_observer is not None:
                    usage_observer(usage)

            try:
                with _reuse_prepared_moa_setup(prepared_moa):
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
                        final_parallel_tool_calls=call.parallel_tool_calls,
                        final_tool_call_protocol=call.tool_call_protocol,
                        shared_prefix=self._shared_prefix,
                        usage_observer=observe_moa_usage,
                    )
                async for event in self._with_initial_keepalives(moa_stream):
                    if event is None:
                        yield OrchestratorEvent(kind="status", text="working")
                    elif event.kind == "delta":
                        if not event.text:
                            continue
                        if first_token_at is None:
                            first_token_at = utc_now_iso()
                        yield OrchestratorEvent(
                            kind="delta",
                            text=event.text,
                            completions=_public_multistage_completions(
                                event.completions
                            ),
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
                        parallel_tool_calls=call.parallel_tool_calls,
                        tool_call_protocol=call.tool_call_protocol,
                        reasoning_effort=call.reasoning_effort,
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
                    completions=_public_multistage_completions(
                        moa_result.completions
                        or (
                            CompletionOutput(
                                index=0,
                                text=moa_result.final_text,
                                token_ids=(),
                                finish_reason="stop",
                            ),
                        )
                    ),
                    prompt_tokens=moa_result.usage[0],
                    completion_tokens=moa_result.usage[1],
                    cached_tokens=moa_result.cached_tokens,
                ),
            )
            return

        conductor = self._new_conductor(
            call,
            notes,
            usage_observer=usage_observer,
        )
        conductor_session, initial_requests = self._take_prepared_initial_roles(
            prepared,
            call,
            decision,
        )
        conductor_result = None
        try:
            async for event in self._with_keepalives(
                conductor.stream(
                    prompt,
                    budget=self._budget,
                    session=conductor_session,
                    prepared_initial_requests=initial_requests,
                )
            ):
                if event is None:
                    yield OrchestratorEvent(kind="status", text="working")
                elif event.kind == "reasoning":
                    if event.text:
                        yield OrchestratorEvent(kind="reasoning", text=event.text)
                elif event.kind == "delta":
                    if not event.text:
                        continue
                    yield OrchestratorEvent(
                        kind="delta",
                        text=event.text,
                        completions=_public_multistage_completions(
                            event.completions
                        ),
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
                    completions=_public_multistage_completions(
                        conductor_result.completions,
                        conductor_result.reasoning_content,
                    ),
                    prompt_tokens=conductor_result.usage[0],
                    completion_tokens=conductor_result.usage[1],
                    cached_tokens=conductor_result.cached_tokens,
                    reasoning_content=conductor_result.reasoning_content,
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
        if not conductor_result.final_unit_ok and not conductor_result.final_text:
            # Same contract as the unary route (issue #496): an empty final
            # unit becomes an explicit error event, never a silent empty stop.
            yield OrchestratorEvent(
                kind="error",
                result=result_with_trace(
                    text="",
                    prompt_tokens=conductor_result.usage[0],
                    completion_tokens=conductor_result.usage[1],
                    cached_tokens=conductor_result.cached_tokens,
                    reasoning_content=conductor_result.reasoning_content,
                ),
                error_type="EmptyFinalOutput",
            )
            return
        yield OrchestratorEvent(
            kind="result",
            result=result_with_trace(
                text=conductor_result.final_text,
                completions=_public_multistage_completions(
                    conductor_result.completions,
                    conductor_result.reasoning_content,
                ),
                prompt_tokens=conductor_result.usage[0],
                completion_tokens=conductor_result.usage[1],
                cached_tokens=conductor_result.cached_tokens,
                reasoning_content=conductor_result.reasoning_content,
                public_completion_tokens=conductor_result.public_completion_tokens,
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
