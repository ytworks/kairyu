"""Conductor: declarative role DAG executed with asyncio waves (design doc D4).

Roles (planner/worker/verifier/synthesizer/custom) form a DAG; a verifier node
gates its target with a bounded refine loop. All prompts are rendered as
``shared_prefix + role_suffix`` so multi-step calls share a KV-cacheable prefix
(design doc D5).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace

from kairyu.async_thread import run_prompt_work
from kairyu.engine.backend import (
    CacheHint,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    backend_supports_chat_template_kwargs,
    backend_supports_prompt_kind,
)
from kairyu.engine.prompt import (
    MultimodalPrompt,
    TemplatedPrompt,
    derive_multimodal_prompt,
    prompt_kind,
    prompt_text,
)
from kairyu.orchestration.budget import Budget, BudgetState
from kairyu.orchestration.execution import (
    NOT_APPLICABLE_SENTINEL,
    ExecutionBackend,
    ExecutionLimits,
    ExecutionReport,
    ExecutionRequest,
    extract_python_block,
    render_matrix_report,
    render_single_report,
    skipped_report,
    unavailable_report,
)
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
from kairyu.sampling_params import PARALLEL_TOOL_CALLS_EXTRA_ARG, SamplingParams

_PASS_PREFIX = "PASS"

# EO-D7 amendment (issue #495): a publisher continuing a committed opening may
# answer exactly this sentinel to decline continuation; the conductor strips
# it from the public answer instead of publishing deliberation-shaped text.
NO_CONTINUATION_SENTINEL = "NO_CONTINUATION"

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
class RoleSamplingOverrides:
    """Static per-role sampling policy (EO-D9).

    Internal roles stay capped by the orchestration-wide internal ceiling; the
    selected final unit carries the caller's public intent and rejects
    overrides at construction.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed_offset: int | None = None
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop", tuple(self.stop))


@dataclass(frozen=True)
class ExecutorRoleConfig:
    """Inputs and limits for a sandbox execution role (ECO-D3)."""

    code_from: tuple[str, ...]
    tests_from: tuple[str, ...]
    mode: str = "single"  # "single" | "matrix"
    limits: ExecutionLimits = ExecutionLimits()

    def __post_init__(self) -> None:
        object.__setattr__(self, "code_from", tuple(self.code_from))
        object.__setattr__(self, "tests_from", tuple(self.tests_from))
        if not self.code_from or not self.tests_from:
            raise ValueError("executor roles require code_from and tests_from")
        if self.mode not in {"single", "matrix"}:
            raise ValueError("executor mode must be 'single' or 'matrix'")


@dataclass(frozen=True)
class RoleSpec:
    name: str
    worker: str
    prompt: str
    role_type: str = "worker"
    depends_on: tuple[str, ...] = ()
    verifies: str | None = None
    sampling: RoleSamplingOverrides | None = None
    executor: ExecutorRoleConfig | None = None
    # Rendered after the prompt body on EVERY attempt, including verifier
    # refinements (which append their feedback to the body first). Used for
    # worker-native scaffolding such as a chat-format assistant marker that
    # must terminate the prompt.
    prompt_suffix: str = ""
    # Alternate body for the selected final unit when the head is absent or
    # disabled for the call: write the complete answer, not a continuation
    # of a committed opening that does not exist (issue #496).
    prompt_headless: str = ""
    # The rendered scaffold already closed the model's private-reasoning
    # span: reasoning-classified output alongside empty public text is the
    # answer itself and is reclaimed as the role's output (issue #496).
    reasoning_closed: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.prompt, TemplatedPrompt):
            raise ValueError(
                "Conductor role templates cannot be tokenizer-owned pre-rendered "
                "chat prompts; provide a plain derivation template"
            )
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        if (self.role_type == "executor") != (self.executor is not None):
            raise ValueError(
                f"role {self.name!r}: role_type 'executor' and executor config "
                "must be declared together"
            )
        if self.executor is not None and self.sampling is not None:
            raise ValueError(f"executor role {self.name!r} cannot declare sampling")
        if self.executor is not None and (self.prompt_headless or self.reasoning_closed):
            raise ValueError(
                f"executor role {self.name!r} cannot declare prompt_headless "
                "or reasoning_closed"
            )


@dataclass(frozen=True)
class ConductorResult:
    final_text: str
    completions: tuple[CompletionOutput, ...]
    outputs: dict[str, str]
    budget_state: BudgetState
    trace: tuple[TraceEvent, ...]
    usage: tuple[int, int] = (0, 0)  # (prompt, completion) summed over units (m11 A1)
    cached_tokens: int = 0
    reasoning_content: str | None = None
    # Backend-reported tokens of the public answer (head + selected final
    # unit); None when any active public stage lacked backend usage. Standard
    # OpenAI usage derives from this, never from the cumulative sums above.
    public_completion_tokens: int | None = None
    # False when the selected final unit failed or produced no public text
    # (issue #496): the caller must see an explicit error, never an internal
    # stage's text or an empty "stop".
    final_unit_ok: bool = True


@dataclass(frozen=True)
class IntermediateOutput:
    """One explicitly publishable completed L2/L1 stage output."""

    node: str
    role: str
    attempt: int
    worker: str
    engine: str
    model: str | None
    text: str
    model_reasoning: str | None = None

    def as_markdown(self) -> str:
        model = self.model or "unspecified"
        lines = [
            f"### {self.node} — attempt {self.attempt + 1}",
            "",
            f"- L2 role: `{self.role}`",
            f"- L1 worker: `{self.worker}`",
            f"- Engine: `{self.engine}`",
            f"- Model: `{model}`",
        ]
        if self.model_reasoning:
            lines.extend(("", "#### Model reasoning", "", self.model_reasoning))
        lines.extend(("", "#### Stage output", "", self.text))
        return "\n".join(lines)


@dataclass(frozen=True)
class ConductorEvent:
    """A pull-through final-stage stream event.

    The Conductor owns the final worker iterator so no task/queue bridge sits
    between the backend and the Orchestrator.  Pre-final DAG work remains an
    implementation detail; callers receive only committed final-answer deltas
    followed by the complete accounting result.
    """

    kind: str  # "reasoning" | "delta" | "result"
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


class _ContinuationDeduper:
    """Drop only an exact verbatim repetition of the committed head prefix.

    While the continuation's opening bytes (leading whitespace ignored) still
    match the committed text they are buffered; a full match is dropped once,
    and the first divergence (or end of stream) flushes the buffer verbatim.
    This is byte-identical prefix dedup, never rewriting (EO-D7).
    """

    def __init__(self, committed: str) -> None:
        self._committed = committed
        self._buffer = ""
        self._matching = bool(committed)
        # With a committed opening the publisher may decline to continue by
        # answering exactly NO_CONTINUATION (EO-D7 amendment, issue #495):
        # buffer while the output is still a prefix of the sentinel and
        # suppress it only on an exact sentinel-only stream.
        self._sentinel_active = bool(committed)
        self._sentinel_buffer = ""

    def feed(self, delta: str) -> str:
        if not self._matching:
            return self._sentinel_feed(delta)
        candidate = self._buffer + delta
        stripped = candidate.lstrip()
        if stripped.startswith(self._committed):
            self._matching = False
            self._buffer = ""
            return self._sentinel_feed(stripped[len(self._committed) :])
        if self._committed.startswith(stripped):
            self._buffer = candidate
            return ""
        self._matching = False
        self._buffer = ""
        return self._sentinel_feed(candidate)

    def _sentinel_feed(self, out: str) -> str:
        if not self._sentinel_active or not out:
            return out
        candidate = self._sentinel_buffer + out
        stripped = candidate.strip()
        if not stripped or NO_CONTINUATION_SENTINEL.startswith(stripped):
            self._sentinel_buffer = candidate
            return ""
        # Divergence (including sentinel followed by more text) flushes
        # verbatim: content is never lost, only an exact sentinel is dropped.
        self._sentinel_active = False
        self._sentinel_buffer = ""
        return candidate

    def flush(self) -> str:
        buffered, self._buffer = self._buffer, ""
        self._matching = False
        withheld, self._sentinel_buffer = self._sentinel_buffer, ""
        combined = withheld + buffered
        if self._sentinel_active:
            self._sentinel_active = False
            if combined.strip() == NO_CONTINUATION_SENTINEL:
                return ""
        return combined


def dedup_public_continuation(committed: str, continuation: str) -> str:
    """Whole-text equivalent of streaming ``_ContinuationDeduper`` semantics."""

    if not committed:
        return continuation
    stripped = continuation.lstrip()
    if stripped.startswith(committed):
        stripped = stripped[len(committed) :]
    else:
        stripped = continuation
    if stripped.strip() == NO_CONTINUATION_SENTINEL:
        # The publisher declined to continue a complete committed opening
        # (EO-D7 amendment, issue #495).
        return ""
    return stripped


def _completion_tokens_for_public_budget(
    usage: GenerationUsage | None,
    completions: tuple[CompletionOutput, ...],
    text: str,
    *,
    unreported_upper_bound: int | None,
) -> int:
    """Account for the head without ever under-reserving caller output.

    Backend usage is authoritative, followed by exact returned token IDs. If
    neither is available, a finite head dispatch must be charged at its full
    upper bound: a whitespace estimate can undercount CJK or other unbroken
    text and let head + continuation exceed the caller limit. The final text
    approximation remains useful only when no finite dispatch bound exists;
    billing still uses backend-reported usage (m9).
    """

    if usage is not None:
        return usage.completion_tokens
    token_count = sum(len(completion.token_ids) for completion in completions)
    if token_count:
        return token_count
    if unreported_upper_bound is not None:
        return unreported_upper_bound
    return len(text.split())


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
    prepared_initial_requests: dict[str, GenerationRequest] = field(
        default_factory=dict
    )
    intermediate_outputs: list[IntermediateOutput] = field(default_factory=list)
    # Public-answer accounting (issue #496): backend-reported completion
    # tokens of the head and the selected final unit, and whether the
    # caller's public limit was consumed before the final unit could run.
    head_completion_tokens: int | None = None
    head_usage_reported: bool = False
    final_unit_completion_tokens: int | None = None
    public_budget_exhausted: bool = False
    final_unit_unusable: bool = False


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


def _verdict_is_inconclusive(verdict_text: str) -> bool:
    """True when no PASS/FAIL was declared (e.g. the role's private
    deliberation consumed its whole token budget before a public verdict)."""

    first_line = verdict_text.strip().splitlines()[0] if verdict_text.strip() else ""
    upper = first_line.upper()
    return not (upper.startswith(_PASS_PREFIX) or upper.startswith("FAIL"))


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
        final_structured_format_in_prompt: bool = False,
        cost_model: CostModel = zero_cost,
        worker_trace: Mapping[str, WorkerTraceIdentity] | None = None,
        usage_observer: Callable[[GenerationUsage], None] | None = None,
        stage_observer: Callable[[TraceEvent], None] | None = None,
        affinity_key: str | None = None,
        final_parallel_tool_calls: bool | None = None,
        final_tool_call_protocol: str = "generic",
        expose_intermediate_outputs: bool = False,
        multimodal_prompt: MultimodalPrompt | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
        execution_workers: Mapping[str, ExecutionBackend] | None = None,
    ) -> None:
        if isinstance(shared_prefix, TemplatedPrompt):
            raise ValueError(
                "Conductor shared_prefix cannot be a tokenizer-owned pre-rendered "
                "chat prompt"
            )
        self._roles = tuple(roles)
        self._workers = dict(workers)
        self._shared_prefix = shared_prefix
        self._prefix_fingerprint = prefix_root_fingerprint(shared_prefix)
        public_sampling = sampling_params or SamplingParams(max_tokens=1024)
        self._final_sampling_params = final_sampling_params or public_sampling
        internal_extra_args = dict(public_sampling.extra_args)
        internal_extra_args.pop(PARALLEL_TOOL_CALLS_EXTRA_ARG, None)
        self._sampling_params = public_sampling.clone(extra_args=internal_extra_args)
        self._final_tools = tuple(final_tools)
        self._final_tool_choice = final_tool_choice
        self._final_tools_in_prompt = final_tools_in_prompt
        self._final_structured_format_in_prompt = final_structured_format_in_prompt
        self._final_parallel_tool_calls = final_parallel_tool_calls
        self._final_tool_call_protocol = final_tool_call_protocol
        self._cost_model = cost_model
        self._usage_observer = usage_observer
        self._stage_observer = stage_observer
        self._affinity_key = affinity_key
        self._expose_intermediate_outputs = expose_intermediate_outputs
        self._multimodal_prompt = multimodal_prompt
        self._chat_template_kwargs = (
            None if chat_template_kwargs is None else dict(chat_template_kwargs)
        )
        self._execution_workers = dict(execution_workers or {})
        supplied_trace = dict(worker_trace or {})
        self._worker_trace = {
            worker: supplied_trace.get(
                worker,
                WorkerTraceIdentity(engine=worker),
            )
            for worker in (*self._workers, *self._execution_workers)
        }
        self._by_name = {role.name: role for role in self._roles}
        self._verifier_for = {
            role.verifies: role for role in self._roles if role.role_type == "verifier"
        }
        self._head = next(
            (role for role in self._roles if role.role_type == "head"),
            None,
        )
        # An executor role named in a verifier's depends_on runs INLINE inside
        # the verify/refine loop (fresh execution evidence for every attempt)
        # instead of as an independent DAG unit.
        self._inline_executors: dict[str, tuple[RoleSpec, ...]] = {}
        self._inline_executor_target: dict[str, str] = {}
        for verifier in self._verifier_for.values():
            bound = tuple(
                self._by_name[dep]
                for dep in verifier.depends_on
                if dep in self._by_name and self._by_name[dep].role_type == "executor"
            )
            if bound and verifier.verifies:
                self._inline_executors[verifier.verifies] = bound
                for spec in bound:
                    self._inline_executor_target[spec.name] = verifier.verifies
        self._units = tuple(
            role
            for role in self._roles
            if role.role_type != "verifier"
            and role.name not in self._inline_executor_target
        )
        self._unit_deps = {unit.name: self._remapped_deps(unit) for unit in self._units}
        self._validate()

    def _selected_final_unit(self) -> RoleSpec:
        terminal = [
            unit
            for unit in self._terminal_units()
            if unit.role_type not in {"head", "executor"}
        ]
        if not terminal:
            raise ValueError("orchestration requires at least one generation role")
        synthesizers = [unit for unit in terminal if unit.role_type == "synthesizer"]
        return (synthesizers + terminal)[0]

    def _head_enabled(self) -> bool:
        """A caller intent incompatible with a split public stream disables the
        head for this call instead of rejecting the request (EO-D7)."""

        if self._head is None:
            return False
        if self._final_tools or self._final_tools_in_prompt:
            return False
        if self._final_structured_format_in_prompt:
            # A plain-text demand for a machine-parsed format (issue #495):
            # the head's mandatory prose opening would corrupt the reply the
            # same way tools/response_format would.
            return False
        params = self._final_sampling_params
        if params.n != 1 or params.best_of not in (None, 1):
            return False
        if params.logprobs is not None:
            return False
        if params.extra_args.get("response_format") is not None:
            return False
        return True

    def _head_sampling_params(self) -> SamplingParams:
        """The head derives from the caller's PUBLIC sampling so its style
        matches the answer, with caller intent carriers stripped (#208)."""

        assert self._head is not None
        extra_args = dict(self._final_sampling_params.extra_args)
        extra_args.pop("response_format", None)
        extra_args.pop(PARALLEL_TOOL_CALLS_EXTRA_ARG, None)
        # The head opens the public answer, so the caller's public limit caps
        # it exactly like the continuation (issue #496): the role override may
        # only shrink below that cap, never exceed it.
        public_cap = self._final_sampling_params.max_tokens
        base_max = self._sampling_params.max_tokens
        if public_cap is not None:
            base_max = public_cap if base_max is None else min(base_max, public_cap)
        params = self._final_sampling_params.clone(
            n=1,
            best_of=None,
            logprobs=None,
            max_tokens=base_max,
            extra_args=extra_args,
        )
        return self._apply_overrides(params, self._head.sampling, cap=public_cap)

    def _role_sampling_params(self, spec: RoleSpec) -> SamplingParams:
        return self._apply_overrides(
            self._sampling_params,
            spec.sampling,
            cap=self._sampling_params.max_tokens,
        )

    @staticmethod
    def _apply_overrides(
        params: SamplingParams,
        overrides: RoleSamplingOverrides | None,
        *,
        cap: int | None,
    ) -> SamplingParams:
        if overrides is None:
            return params
        updates: dict[str, object] = {}
        if overrides.temperature is not None:
            updates["temperature"] = overrides.temperature
        if overrides.top_p is not None:
            updates["top_p"] = overrides.top_p
        if overrides.max_tokens is not None:
            updates["max_tokens"] = (
                overrides.max_tokens if cap is None else min(overrides.max_tokens, cap)
            )
        if overrides.seed_offset is not None:
            # Mirrors MoA proposal seed derivation for diversity under a fixed
            # caller seed.
            updates["seed"] = (
                overrides.seed_offset
                if params.seed is None
                else params.seed + overrides.seed_offset
            )
        if overrides.stop:
            base_stop = params.stop
            if base_stop is None:
                combined: tuple[str, ...] = overrides.stop
            elif isinstance(base_stop, str):
                combined = (base_stop, *overrides.stop)
            else:
                combined = (*base_stop, *overrides.stop)
            updates["stop"] = combined
        return params.clone(**updates) if updates else params

    def _request_intent(
        self,
        spec: RoleSpec,
        run: _RunState | None = None,
    ) -> tuple[
        SamplingParams,
        tuple[Mapping[str, object], ...],
        str | Mapping[str, object] | None,
        bool,
        bool | None,
        str,
    ]:
        if self._head is not None and spec.name == self._head.name:
            return (self._head_sampling_params(), (), None, False, None, "generic")
        if spec.name == self._selected_final_unit().name:
            return (
                self._final_unit_sampling_params(run),
                self._final_tools,
                self._final_tool_choice,
                self._final_tools_in_prompt,
                self._final_parallel_tool_calls,
                self._final_tool_call_protocol,
            )
        return self._role_sampling_params(spec), (), None, False, None, "generic"

    def _final_unit_sampling_params(self, run: _RunState | None) -> SamplingParams:
        """The caller's public intent, minus tokens the head already spent.

        The public answer is head + continuation, so a finite caller limit is
        one budget across the seam (issue #496). The head's completed usage is
        known before the final unit dispatches because the final unit depends
        on the head (EO-D7).
        """

        params = self._final_sampling_params
        if run is None or params.max_tokens is None:
            return params
        spent = run.head_completion_tokens
        if not spent:
            return params
        return params.clone(max_tokens=max(1, params.max_tokens - spent))

    def _public_budget_remaining(self, run: _RunState) -> int | None:
        """Tokens the final unit may still generate; ``None`` when unlimited."""

        limit = self._final_sampling_params.max_tokens
        if limit is None:
            return None
        return limit - (run.head_completion_tokens or 0)

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
        """Dependencies at unit granularity: a dep on a verifier (or on an
        executor bound inline to a verifier) maps to the verifier's target."""
        deps = set()
        for dep in unit.depends_on:
            dep_role = self._by_name.get(dep)
            if dep_role is not None and dep_role.role_type == "verifier" and dep_role.verifies:
                deps.add(dep_role.verifies)
            elif dep in self._inline_executor_target:
                deps.add(self._inline_executor_target[dep])
            else:
                deps.add(dep)
        return frozenset(deps - {unit.name})

    def _transitive_unit_closure(self, name: str) -> set[str]:
        """Unit names guaranteed complete before unit ``name`` generates."""

        closure: set[str] = set()
        frontier = set(self._unit_deps.get(name, frozenset()))
        while frontier:
            dep = frontier.pop()
            if dep in closure:
                continue
            closure.add(dep)
            frontier.update(self._unit_deps.get(dep, frozenset()))
        return closure

    def _validate(self) -> None:
        if len(self._by_name) != len(self._roles):
            raise ValueError("duplicate role names")
        for role in self._roles:
            if role.role_type == "executor":
                if role.worker not in self._execution_workers:
                    raise ValueError(
                        f"executor role {role.name!r} references unknown execution "
                        f"worker {role.worker!r}"
                    )
            elif role.worker not in self._workers:
                raise ValueError(f"role {role.name!r} references unknown worker {role.worker!r}")
            for dep in role.depends_on:
                if dep not in self._by_name:
                    raise ValueError(f"role {role.name!r} has unknown dependency {dep!r}")
            if role.executor is not None:
                for source in (*role.executor.code_from, *role.executor.tests_from):
                    source_role = self._by_name.get(source)
                    if source_role is None or source_role.role_type in {
                        "executor",
                        "verifier",
                    }:
                        raise ValueError(
                            f"executor role {role.name!r} input {source!r} must be "
                            "an existing generation role"
                        )
                    if source not in role.depends_on:
                        raise ValueError(
                            f"executor role {role.name!r} must depend on its input "
                            f"{source!r}"
                        )
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
                # An executor dependency is re-run inline before each verdict;
                # its own inputs must be complete before the target generates.
                target = self._by_name[role.verifies]
                available = set(target.depends_on) | {role.verifies}
                for dep in role.depends_on:
                    dep_role = self._by_name[dep]
                    if dep_role.role_type == "executor":
                        continue
                    if dep != role.verifies and dep not in available:
                        raise ValueError(
                            f"verifier {role.name!r} depends on {dep!r}, which is not "
                            f"available when it runs inline after {role.verifies!r}; "
                            f"add {dep!r} to {role.verifies!r}'s depends_on"
                        )
        self._validate_head()
        self._check_acyclic()
        self._validate_inline_executors()
        if self._units:
            final = self._selected_final_unit()
            if final.sampling is not None:
                raise ValueError(
                    f"the final unit {final.name!r} carries the caller's public "
                    "intent and cannot declare sampling overrides"
                )

    def _validate_head(self) -> None:
        heads = [role for role in self._roles if role.role_type == "head"]
        if len(heads) > 1:
            raise ValueError("at most one role may declare role_type 'head'")
        if not heads:
            return
        head = heads[0]
        if head.depends_on:
            raise ValueError(f"head role {head.name!r} cannot declare dependencies")
        if head.name in self._verifier_for:
            raise ValueError(f"head role {head.name!r} cannot have a verifier")
        if len(self._units) < 2:
            raise ValueError("a head role requires at least one downstream unit")

    def _validate_inline_executors(self) -> None:
        # After _check_acyclic so the closure walk terminates.
        for target_name, executors in self._inline_executors.items():
            available = self._transitive_unit_closure(target_name) | {target_name}
            for spec in executors:
                assert spec.executor is not None
                for source in (*spec.executor.code_from, *spec.executor.tests_from):
                    resolved = self._inline_executor_target.get(source, source)
                    if resolved != target_name and resolved not in available:
                        raise ValueError(
                            f"inline executor {spec.name!r} input {source!r} is not "
                            f"complete before its verifier target {target_name!r}"
                        )
        if self._head is not None and self._units:
            final = self._selected_final_unit()
            if self._head.name not in self._transitive_unit_closure(final.name):
                raise ValueError(
                    f"the final unit {final.name!r} must (transitively) depend on "
                    f"the head role {self._head.name!r} to continue its committed "
                    "public prefix"
                )

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

    def _rendered_role_prompt(
        self,
        spec: RoleSpec,
        query: str,
        outputs: Mapping[str, str],
    ) -> str:
        body = spec.prompt
        if (
            spec.prompt_headless
            and spec.name == self._selected_final_unit().name
            and (self._head is None or not self._head_enabled())
        ):
            # No committed opening exists on this call: the publisher writes
            # the complete answer instead of continuing one (issue #496).
            body = spec.prompt_headless
        return f"{self._render(body, query, outputs)}{spec.prompt_suffix}"

    @staticmethod
    def _prompt_body(spec: RoleSpec, rendered: str) -> str:
        """Strip the role suffix so refinements append before it."""

        if spec.prompt_suffix and rendered.endswith(spec.prompt_suffix):
            return rendered[: len(rendered) - len(spec.prompt_suffix)]
        return rendered

    def _worker_prompt(self, spec: RoleSpec, text: str):
        if self._multimodal_prompt is None or not backend_supports_prompt_kind(
            self._workers[spec.worker],
            "multimodal",
        ):
            return text
        return derive_multimodal_prompt(self._multimodal_prompt, text)

    def _worker_chat_template_kwargs(
        self,
        spec: RoleSpec,
        prompt: object,
    ) -> Mapping[str, object] | None:
        if not isinstance(prompt, MultimodalPrompt):
            return None
        if self._chat_template_kwargs is not None and not backend_supports_chat_template_kwargs(
            self._workers[spec.worker],
            frozenset(self._chat_template_kwargs),
        ):
            raise ValueError(
                f"worker {spec.worker!r} does not support chat_template_kwargs"
            )
        return self._chat_template_kwargs

    def _cache_hint(self, session: str, *, multimodal: bool = False) -> CacheHint:
        # A conversation-derived affinity key keeps consecutive agent turns on
        # the replica holding the warm KV prefix (issue #495); the per-request
        # session only namespaces request ids, so it stays unique.
        return CacheHint(
            session_id=self._affinity_key or session,
            prefix_fingerprint="" if multimodal else self._prefix_fingerprint,
        )

    @staticmethod
    def preflight_session(request_id_suffix: str) -> str:
        """Return the execution session bound to an initial-request plan."""

        return f"preflight-{request_id_suffix}"

    def initial_requests(
        self,
        query: str,
        *,
        request_id_suffix: str,
    ) -> tuple[tuple[RoleSpec, GenerationRequest], ...]:
        """Build the exact dependency-free role intents used by the first wave.

        Orchestration preflight uses these requests before an HTTP stream is
        opened. Later role prompts depend on generated outputs and therefore
        cannot be rendered exactly in advance, but the first wave can and must
        include its static role envelope rather than validating only ``query``.
        """

        if isinstance(query, TemplatedPrompt):
            raise ValueError(
                "Conductor cannot derive role prompts from a tokenizer-owned "
                "pre-rendered chat prompt"
            )
        session = self.preflight_session(request_id_suffix)
        requests: list[tuple[RoleSpec, GenerationRequest]] = []
        head_enabled = self._head_enabled()
        for spec in self._units:
            if self._unit_deps[spec.name]:
                continue
            if spec.role_type == "executor":
                continue
            if spec.role_type == "head" and not head_enabled:
                continue
            (
                sampling_params,
                tools,
                tool_choice,
                tools_in_prompt,
                parallel_tool_calls,
                tool_call_protocol,
            ) = (
                self._request_intent(spec)
            )
            requests.append(
                (
                    spec,
                    GenerationRequest(
                        request_id=f"preflight-role-{spec.name}-{request_id_suffix}",
                        prompt=(
                            role_prompt := self._worker_prompt(
                                spec,
                                self._rendered_role_prompt(spec, query, {}),
                            )
                        ),
                        sampling_params=sampling_params,
                        cache_hint=self._cache_hint(
                            session,
                            multimodal=isinstance(role_prompt, MultimodalPrompt),
                        ),
                        chat_template_kwargs=self._worker_chat_template_kwargs(
                            spec,
                            role_prompt,
                        ),
                        tools=tools,
                        tool_choice=tool_choice,
                        tools_in_prompt=tools_in_prompt,
                        parallel_tool_calls=parallel_tool_calls,
                        tool_call_protocol=tool_call_protocol,
                    ),
                )
            )
        return tuple(requests)

    def _generation_request(
        self,
        run: _RunState,
        session: str,
        spec: RoleSpec,
        prompt: str,
        attempt: int,
    ) -> GenerationRequest:
        """Consume an exact first-wave preflight request when one is bound."""

        (
            sampling_params,
            tools,
            tool_choice,
            tools_in_prompt,
            parallel_tool_calls,
            tool_call_protocol,
        ) = (
            self._request_intent(spec, run)
        )
        request_prompt = self._worker_prompt(spec, prompt)
        candidate = GenerationRequest(
            request_id=f"{session}-{spec.name}-{attempt}",
            prompt=request_prompt,
            sampling_params=sampling_params,
            cache_hint=self._cache_hint(
                session,
                multimodal=isinstance(request_prompt, MultimodalPrompt),
            ),
            chat_template_kwargs=self._worker_chat_template_kwargs(
                spec,
                request_prompt,
            ),
            tools=tools,
            tool_choice=tool_choice,
            tools_in_prompt=tools_in_prompt,
            parallel_tool_calls=parallel_tool_calls,
            tool_call_protocol=tool_call_protocol,
        )
        if attempt != 0:
            return candidate
        prepared = run.prepared_initial_requests.pop(spec.name, None)
        if prepared is None:
            return candidate
        if replace(prepared, request_id=candidate.request_id) != candidate:
            raise RuntimeError(
                f"prepared initial role request {spec.name!r} no longer matches "
                "the execution contract"
            )
        return prepared

    @staticmethod
    def _prepared_initial_prompt(
        run: _RunState,
        spec: RoleSpec,
    ) -> str | None:
        prepared = run.prepared_initial_requests.get(spec.name)
        if prepared is None:
            return None
        prompt = prepared.prompt
        text = (
            prompt_text(prompt.base)
            if isinstance(prompt, MultimodalPrompt)
            else prompt_text(prompt)
        )
        if text is None:
            raise RuntimeError(
                f"prepared initial role request {spec.name!r} is not text"
            )
        return text

    @staticmethod
    def _refinement_prompt(
        base_prompt: str,
        text: str,
        verdict: str,
    ) -> str:
        return (
            f"{base_prompt}\n\nPrevious attempt:\n{text}\n\n"
            f"Verifier feedback:\n{verdict}\n\n"
            "Revise the answer addressing the feedback."
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
        event = TraceEvent(
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
        # Every conductor trace event funnels through here, so this is the
        # one seam that can feed aggregate per-stage latency attribution
        # (issue #495) without touching each dispatch site.
        if self._stage_observer is not None:
            self._stage_observer(event)
        return event

    @staticmethod
    def _unit_public_output(
        spec: RoleSpec,
        text: str,
        completions: tuple[CompletionOutput, ...],
    ) -> tuple[str, tuple[CompletionOutput, ...]]:
        """Reclaim reasoning-classified output on a closed-reasoning role.

        A ``reasoning_closed`` role's scaffold already terminated the model's
        private-reasoning span in the prompt, so an upstream reasoning parser
        that classified the whole generation as reasoning has misrouted the
        public answer (issue #496). Empty public text + non-empty reasoning
        is therefore the answer itself; a genuinely public text passes
        through untouched.
        """

        if not spec.reasoning_closed or text.strip():
            return text, completions
        reclaimed = "\n\n".join(
            completion.reasoning_content
            for completion in completions
            if completion.reasoning_content
        )
        if not reclaimed:
            return text, completions
        return reclaimed, tuple(
            replace(completion, text=completion.reasoning_content, reasoning_content=None)
            if completion.reasoning_content and not completion.text.strip()
            else completion
            for completion in completions
        )

    @staticmethod
    def _has_public_output(
        text: str,
        completions: tuple[CompletionOutput, ...],
    ) -> bool:
        """Whether any returned choice contains caller-visible text."""

        return bool(
            text.strip()
            or any(completion.text.strip() for completion in completions)
        )

    @staticmethod
    def _model_reasoning(completions: tuple[CompletionOutput, ...]) -> str | None:
        parts = tuple(
            completion.reasoning_content
            for completion in completions
            if completion.reasoning_content
        )
        return "\n\n".join(parts) or None

    def _record_intermediate(
        self,
        run: _RunState,
        spec: RoleSpec,
        attempt: int,
        observed: _GenerationObservation,
    ) -> IntermediateOutput | None:
        if (
            not self._expose_intermediate_outputs
            or spec.name == self._selected_final_unit().name
            # The head is committed public content; duplicating it into the
            # reasoning sections would repeat the answer opening (EO-D7).
            or spec.role_type == "head"
        ):
            return None
        identity = self._worker_trace[spec.worker]
        output = IntermediateOutput(
            node=spec.name,
            role=spec.role_type,
            attempt=attempt,
            worker=spec.worker,
            engine=identity.engine,
            model=identity.model,
            text=observed.text,
            model_reasoning=self._model_reasoning(observed.completions),
        )
        run.intermediate_outputs.append(output)
        return output

    @staticmethod
    async def _emit_intermediate(
        event_sink: Callable[[ConductorEvent], Awaitable[None]] | None,
        output: IntermediateOutput | None,
    ) -> None:
        """Stream one completed pre-final stage as it finishes (EO-D8)."""

        if event_sink is None or output is None:
            return
        await event_sink(
            ConductorEvent(
                kind="reasoning",
                text=f"{output.as_markdown()}\n\n---\n\n",
            )
        )

    def _attribution_summary(self, spec: RoleSpec) -> str:
        identity = self._worker_trace[spec.worker]
        model = identity.model or "unspecified"
        return "\n".join(
            (
                "### Final answer attribution",
                "",
                f"- L2 role: `{spec.role_type}`",
                f"- L1 worker: `{spec.worker}`",
                f"- Engine: `{identity.engine}`",
                f"- Model: `{model}`",
            )
        )

    def _reasoning_content(
        self,
        run: _RunState,
        *,
        include_final_attribution: bool = True,
    ) -> str | None:
        if not self._expose_intermediate_outputs:
            return None
        sections = [output.as_markdown() for output in run.intermediate_outputs]
        if not include_final_attribution:
            return "\n\n---\n\n".join(sections) or None
        final = self._final_output_unit(run)
        if final is None:
            return None
        sections.insert(0, self._attribution_summary(final))
        return "\n\n---\n\n".join(sections)

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
        request = self._generation_request(
            run,
            session,
            event_spec,
            prompt,
            attempt,
        )
        queued_at = utc_now_iso()
        budget_before = run.budget
        unknown_cost = run.budget.budget.max_cost_usd is not None
        reserved = run.budget.try_reserve(unknown_cost=unknown_cost)
        if reserved is None:
            raise _BudgetRefused
        run.budget = reserved
        started_at = utc_now_iso()
        from kairyu.telemetry import traced_span

        try:
            with traced_span(
                "kairyu.conductor.stage",
                {
                    "kairyu.request_id": request.request_id,
                    "kairyu.stage.name": event_spec.name,
                    "kairyu.stage.role": event_spec.role_type,
                    "kairyu.stage.worker": worker,
                    "kairyu.stage.operation": operation,
                    "kairyu.stage.attempt": attempt,
                    "kairyu.stream": False,
                },
            ) as span:
                result = await backend.generate(request)
                actual_cost = self._cost_model(request, result)
                reconciled = run.budget.reconcile_success(
                    cost=actual_cost,
                    unknown_cost=unknown_cost,
                )
                if span is not None:
                    span.set_attribute("kairyu.stage.status", "success")
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

    async def _run_executor_unit(
        self,
        run: _RunState,
        session: str,
        spec: RoleSpec,
        *,
        depth: int = 0,
        event_sink: Callable[[ConductorEvent], Awaitable[None]] | None = None,
    ) -> None:
        """Run one sandbox execution stage (ECO-D3).

        A stage with nothing runnable skips locally without reserving a step
        (the everyday non-coding path); a sandbox failure degrades to an
        ``unavailable`` report instead of failing the request. Steps are
        reserved strictly before the await (M1 D4); token usage is untouched.
        """

        assert spec.executor is not None
        config = spec.executor
        queued_at = utc_now_iso()
        codes: dict[str, str] = {}
        for name in config.code_from:
            code = await run_prompt_work(extract_python_block, run.outputs.get(name, ""))
            if code:
                codes[name] = code
        tests: str | None = None
        for name in config.tests_from:
            raw = run.outputs.get(name, "")
            if raw.strip() == NOT_APPLICABLE_SENTINEL:
                continue
            block = await run_prompt_work(extract_python_block, raw)
            if block:
                tests = block
                break

        def finish(
            report_text: str,
            *,
            status: str,
            attempt: int,
            timing: TraceTiming | None,
            budget: TraceBudget,
            metadata: dict,
            record: bool = True,
        ) -> IntermediateOutput | None:
            run.outputs[spec.name] = report_text
            run.trace.append(
                self._trace_event(
                    spec,
                    "execution",
                    operation="execution",
                    status=status,
                    attempt=attempt,
                    detail=f"attempt={attempt} execution={metadata.get('execution_status')}",
                    timing=timing,
                    budget=budget,
                    metadata=metadata,
                )
            )
            output = None
            if record:
                output = self._record_intermediate(
                    run,
                    spec,
                    attempt,
                    _GenerationObservation(
                        text=report_text,
                        completions=(),
                        timing=timing or TraceTiming(),
                        usage=None,
                        budget=budget,
                    ),
                )
            if depth == 0 and spec.name not in run.completion_order:
                run.completion_order.append(spec.name)
            return output

        if not codes or tests is None:
            output = finish(
                render_single_report(skipped_report("no_runnable_code")),
                status="skipped",
                attempt=depth,
                # started_at mirrors queued_at: trace consumers require a
                # complete timing envelope on retained execution events.
                timing=TraceTiming(
                    queued_at=queued_at,
                    started_at=queued_at,
                    completed_at=utc_now_iso(),
                ),
                budget=TraceBudget.between(run.budget, run.budget),
                metadata={"execution_status": "skipped", "mode": config.mode},
                record=False,
            )
            await self._emit_intermediate(event_sink, output)
            return
        budget_before = run.budget
        reserved = run.budget.try_reserve()
        if reserved is None:
            output = finish(
                render_single_report(skipped_report("budget")),
                status="skipped",
                attempt=depth,
                timing=TraceTiming(
                    queued_at=queued_at,
                    started_at=queued_at,
                    completed_at=utc_now_iso(),
                ),
                budget=TraceBudget.between(run.budget, run.budget),
                metadata={"execution_status": "skipped", "reason": "budget"},
                record=False,
            )
            await self._emit_intermediate(event_sink, output)
            return
        run.budget = reserved
        started_at = utc_now_iso()
        backend = self._execution_workers[spec.worker]

        def execution_request(name: str, code: str) -> ExecutionRequest:
            return ExecutionRequest(
                request_id=f"{session}-{spec.name}-{depth}-{name}",
                language="python",
                files={"solution.py": code, "test_solution.py": tests},
                entry="pytest",
                limits=config.limits,
            )

        try:
            names = tuple(codes)
            results = await asyncio.gather(
                *(backend.execute(execution_request(name, codes[name])) for name in names),
                return_exceptions=True,
            )
            reports: dict[str, ExecutionReport] = {}
            for name, outcome in zip(names, results, strict=True):
                if isinstance(outcome, BaseException):
                    if not isinstance(outcome, Exception):
                        raise outcome
                    reports[name] = unavailable_report(type(outcome).__name__)
                else:
                    reports[name] = outcome
        except BaseException:
            run.budget = run.budget.release()
            raise
        run.budget = run.budget.reconcile_success(cost=0.0)
        if config.mode == "matrix":
            report_text = render_matrix_report(reports)
        else:
            report_text = render_single_report(next(iter(reports.values())))
        statuses = {report.status for report in reports.values()}
        overall = "ok" if statuses == {"ok"} else ",".join(sorted(statuses))
        metadata = {
            "execution_status": overall,
            "mode": config.mode,
            "candidates": len(reports),
            "passed": sum(report.passed for report in reports.values()),
            "failed": sum(report.failed for report in reports.values()),
            "errors": sum(report.errors for report in reports.values()),
            "duration_ms": sum(report.duration_ms for report in reports.values()),
        }
        output = finish(
            report_text,
            status="success",
            attempt=depth,
            timing=TraceTiming(
                queued_at=queued_at,
                started_at=started_at,
                completed_at=utc_now_iso(),
            ),
            budget=TraceBudget.between(
                budget_before,
                run.budget,
                steps_consumed=1,
                cost_consumed_usd=0.0,
            ),
            metadata=metadata,
        )
        await self._emit_intermediate(event_sink, output)

    async def _stream_head_unit(
        self,
        run: _RunState,
        session: str,
        query: str,
        spec: RoleSpec,
        event_sink: Callable[[ConductorEvent], Awaitable[None]],
    ) -> None:
        """Stream the head role's PUBLIC deltas from t=0 (EO-D7).

        The head's committed bytes can never be retracted, so failure degrades
        instead of raising: with zero bytes emitted the continuation becomes
        the sole public stream; after bytes were emitted the partial prefix is
        committed and downstream roles continue from it.
        """

        if run.budget.is_exhausted:
            run.trace.append(
                self._trace_event(
                    spec,
                    "skipped:budget",
                    operation="generation",
                    status="skipped",
                    attempt=0,
                    budget=TraceBudget.between(run.budget, run.budget),
                    metadata={"reason": "budget", "head": True},
                )
            )
            return
        prompt = self._prepared_initial_prompt(run, spec)
        if prompt is None:
            prompt = await run_prompt_work(
                self._rendered_role_prompt,
                spec,
                query,
                dict(run.outputs),
            )
        backend = self._workers[spec.worker]
        request = self._generation_request(run, session, spec, prompt, 0)
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
                    metadata={"reason": "budget", "head": True},
                )
            )
            return
        run.budget = reserved
        started_at = utc_now_iso()
        emitted = 0
        last_result: GenerationResult | None = None
        latest_usage = None
        first_token_at = None
        text_parts: list[str] = []
        legacy_text = ""
        from kairyu.telemetry import traced_span

        def head_trace_usage() -> TraceUsage | None:
            if latest_usage is None:
                return None
            run.usage[0] += latest_usage.prompt_tokens
            run.usage[1] += latest_usage.completion_tokens
            run.cached_tokens += latest_usage.cached_tokens
            self._observe_usage(run)
            return TraceUsage(
                prompt_tokens=latest_usage.prompt_tokens,
                completion_tokens=latest_usage.completion_tokens,
                cached_tokens=latest_usage.cached_tokens,
            )

        try:
            with traced_span(
                "kairyu.conductor.stage",
                {
                    "kairyu.request_id": request.request_id,
                    "kairyu.stage.name": spec.name,
                    "kairyu.stage.role": spec.role_type,
                    "kairyu.stage.worker": spec.worker,
                    "kairyu.stage.operation": "generation",
                    "kairyu.stage.attempt": 0,
                    "kairyu.stream": True,
                },
            ) as span:
                async for partial in backend.stream(request):
                    last_result = partial
                    if partial.usage is not None:
                        latest_usage = partial.usage
                        self._observe_usage(run, latest_usage)
                    completion = partial.completions[0] if partial.completions else None
                    if (
                        completion is not None
                        and completion.text_delta is None
                        and completion.text_offset is None
                    ):
                        text = completion.text
                        if not text.startswith(legacy_text):
                            raise RuntimeError(
                                "head worker stream must emit cumulative, "
                                "prefix-stable text"
                            )
                        delta = text[len(legacy_text) :]
                        legacy_text = text
                        emitted = len(text)
                    else:
                        delta, emitted = (
                            completion.delta_after(emitted)
                            if completion is not None
                            else ("", emitted)
                        )
                    text_parts.append(delta)
                    if first_token_at is None and partial.completions:
                        first_token_at = utc_now_iso()
                    if delta:
                        await event_sink(ConductorEvent(kind="delta", text=delta))
                if last_result is None:
                    raise RuntimeError("head worker stream produced no result")
                committed = last_result.text or "".join(text_parts)
                run.outputs[spec.name] = committed
                run.head_completion_tokens = _completion_tokens_for_public_budget(
                    latest_usage,
                    last_result.completions,
                    committed,
                    unreported_upper_bound=request.sampling_params.max_tokens,
                )
                run.head_usage_reported = latest_usage is not None
                actual_cost = self._cost_model(request, last_result)
                run.budget = run.budget.reconcile_success(
                    cost=actual_cost,
                    unknown_cost=unknown_cost,
                )
                if span is not None:
                    span.set_attribute("kairyu.stage.status", "success")
        except Exception as error:
            partial_text = "".join(text_parts)
            if partial_text:
                run.outputs[spec.name] = partial_text
                run.head_completion_tokens = _completion_tokens_for_public_budget(
                    latest_usage,
                    last_result.completions if last_result is not None else (),
                    partial_text,
                    unreported_upper_bound=request.sampling_params.max_tokens,
                )
                run.head_usage_reported = latest_usage is not None
            run.budget = run.budget.release(unknown_cost=unknown_cost)
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
                    usage=head_trace_usage(),
                    budget=TraceBudget.between(budget_before, run.budget),
                    metadata={
                        "streamed": True,
                        "head": True,
                        "partial": bool(partial_text),
                    },
                    error=TraceError(type=type(error).__name__),
                )
            )
            return
        except BaseException:
            run.budget = run.budget.release(unknown_cost=unknown_cost)
            raise
        run.trace.append(
            self._trace_event(
                spec,
                "generated",
                operation="generation",
                status="success",
                attempt=0,
                detail="attempt=0 streamed=true head=true",
                timing=TraceTiming(
                    queued_at=queued_at,
                    started_at=started_at,
                    first_token_at=first_token_at,
                    completed_at=utc_now_iso(),
                ),
                usage=head_trace_usage(),
                budget=TraceBudget.between(
                    budget_before,
                    run.budget,
                    steps_consumed=1,
                    cost_consumed_usd=actual_cost,
                ),
                metadata={"streamed": True, "head": True},
            )
        )
        run.completion_order.append(spec.name)

    async def _run_unit(
        self,
        run: _RunState,
        session: str,
        query: str,
        spec: RoleSpec,
        *,
        event_sink: Callable[[ConductorEvent], Awaitable[None]] | None = None,
    ) -> None:
        if spec.role_type == "executor":
            await self._run_executor_unit(run, session, spec, event_sink=event_sink)
            return
        if spec.role_type == "head" and event_sink is not None:
            await self._stream_head_unit(run, session, query, spec, event_sink)
            return
        if spec.name == self._selected_final_unit().name:
            remaining = self._public_budget_remaining(run)
            if remaining is not None and remaining < 1:
                # The head consumed the caller's whole public limit; the
                # committed prefix stands alone (EO-D7) and the response
                # reports the truncation as "length" (issue #496).
                run.public_budget_exhausted = True
                run.trace.append(
                    self._trace_event(
                        spec,
                        "skipped:public_budget",
                        operation="generation",
                        status="skipped",
                        attempt=0,
                        budget=TraceBudget.between(run.budget, run.budget),
                        metadata={"reason": "public_budget"},
                    )
                )
                return
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
        rendered = self._prepared_initial_prompt(run, spec)
        if rendered is None:
            rendered = await run_prompt_work(
                self._rendered_role_prompt,
                spec,
                query,
                dict(run.outputs),
            )
        base_prompt = self._prompt_body(spec, rendered)
        verifier = self._verifier_for.get(spec.name)
        prompt = rendered
        depth = 0
        is_final_unit = spec.name == self._selected_final_unit().name
        empty_retry_used = False
        while True:
            try:
                observed = await self._generate(
                    run,
                    session,
                    spec.name,
                    spec.worker,
                    prompt,
                    depth + (1 if empty_retry_used else 0),
                    spec=spec,
                    operation="generation",
                )
            except _BudgetRefused:
                refused_attempt = depth + (1 if empty_retry_used else 0)
                run.trace.append(
                    self._trace_event(
                        spec,
                        "skipped:budget",
                        operation="generation",
                        status="skipped",
                        attempt=refused_attempt,
                        budget=TraceBudget.between(run.budget, run.budget),
                        metadata={"reason": "budget"},
                    )
                )
                if spec.name not in run.outputs:
                    if is_final_unit and empty_retry_used:
                        # The first final attempt was empty and its one bounded
                        # retry could not be admitted. Treat that as the same
                        # unusable final result as a second empty generation;
                        # otherwise _final_output_unit() may publish an
                        # internal stage as a successful answer.
                        run.final_unit_unusable = True
                        run.trace.append(
                            self._trace_event(
                                spec,
                                "failed",
                                operation="generation",
                                status="failed",
                                attempt=refused_attempt,
                                detail="empty_output",
                                error=TraceError(type="EmptyFinalOutput"),
                            )
                        )
                    return
                break
            text, completions = self._unit_public_output(
                spec,
                observed.text,
                observed.completions,
            )
            if (
                is_final_unit
                and verifier is None
                and not empty_retry_used
                and not self._has_public_output(text, completions)
                and not self._head_committed_text(run)
            ):
                # One bounded re-dispatch before the run is declared unusable
                # (issue #496): an empty public answer must never surface as
                # a silent successful "stop".
                empty_retry_used = True
                run.trace.append(
                    self._trace_event(
                        spec,
                        "retry:empty_output",
                        operation="generation",
                        status="retried",
                        attempt=depth,
                        detail="empty public output",
                        budget=TraceBudget.between(run.budget, run.budget),
                        metadata={"reason": "empty_output"},
                    )
                )
                continue
            run.outputs[spec.name] = text
            intermediate = self._record_intermediate(run, spec, depth, observed)
            if spec.role_type == "head":
                run.head_completion_tokens = _completion_tokens_for_public_budget(
                    observed.usage,
                    observed.completions,
                    text,
                    unreported_upper_bound=self._head_sampling_params().max_tokens,
                )
                run.head_usage_reported = observed.usage is not None
            if is_final_unit:
                run.final_completions = completions
                run.final_unit_completion_tokens = (
                    observed.usage.completion_tokens
                    if observed.usage is not None
                    else None
                )
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
            await self._emit_intermediate(event_sink, intermediate)
            if verifier is None:
                break
            # Inline-bound executors re-run against the fresh attempt so the
            # verifier never judges stale execution evidence (ECO-D3).
            for exec_spec in self._inline_executors.get(spec.name, ()):
                await self._run_executor_unit(
                    run,
                    session,
                    exec_spec,
                    depth=depth,
                    event_sink=event_sink,
                )
            verifier_prompt = await run_prompt_work(
                self._rendered_role_prompt,
                verifier,
                query,
                dict(run.outputs),
            )
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
            verdict_trace_spec = verifier
            verdict_trace_observed: _GenerationObservation | None = (
                verifier_observed
            )
            if _verdict_is_inconclusive(verdict):
                # A truncated or empty verdict is not evidence of a draft
                # defect, and counting it as FAIL burns a full refinement
                # round (issue #495). One bounded re-verification demands an
                # immediate verdict; its outcome then stands either way.
                retry_prompt = (
                    self._prompt_body(verifier, verifier_prompt)
                    + "\nYour previous verification ran out of output budget "
                    "before emitting a verdict. Answer immediately: PASS or "
                    "FAIL on the first line, then at most three short bullet "
                    "points."
                    + verifier.prompt_suffix
                )
                run.trace.append(
                    self._trace_event(
                        verifier,
                        "verified:inconclusive",
                        operation="verification",
                        status="success",
                        attempt=depth,
                        detail=f"attempt={depth} inconclusive=true",
                        timing=verifier_observed.timing,
                        usage=verifier_observed.usage,
                        budget=verifier_observed.budget,
                        metadata={
                            "pass": False,
                            "inconclusive": True,
                            "reverification": False,
                        },
                    )
                )
                retry_spec = replace(
                    verifier, name=f"{verifier.name}:reverify"
                )
                # The first observation has already been emitted. If the
                # retry cannot dispatch or fails, the final round outcome
                # remains attributable to the original verifier without
                # counting that stage timing twice.
                verdict_trace_observed = None
                try:
                    retry_observed = await self._generate(
                        run,
                        session,
                        verifier.name,
                        verifier.worker,
                        retry_prompt,
                        depth,
                        spec=retry_spec,
                        operation="verification",
                    )
                    verifier_observed = retry_observed
                    verdict = verifier_observed.text
                    verdict_trace_spec = retry_spec
                    verdict_trace_observed = retry_observed
                except _BudgetRefused:
                    run.trace.append(
                        self._trace_event(
                            retry_spec,
                            "skipped:budget",
                            operation="verification",
                            status="skipped",
                            attempt=depth,
                            budget=TraceBudget.between(run.budget, run.budget),
                            metadata={"reason": "budget"},
                        )
                    )
                except _ObservedGenerationError:
                    pass
            run.outputs[verifier.name] = verdict
            passed = _is_pass(verdict)
            verdict_inconclusive = _verdict_is_inconclusive(verdict)
            # One immediate re-verification is the bounded recovery for a
            # missing verdict. If that attempt is also inconclusive, retain
            # the best draft instead of converting missing control output
            # into a synthetic FAIL and paying a full refinement round.
            can_refine = (
                run.budget.can_refine(depth) and not verdict_inconclusive
            )
            verifier_intermediate = self._record_intermediate(
                run,
                verifier,
                depth,
                verifier_observed,
            )
            run.trace.append(
                self._trace_event(
                    verdict_trace_spec,
                    "verified",
                    operation="verification",
                    status="success",
                    attempt=depth,
                    detail=f"attempt={depth} pass={passed}",
                    timing=(
                        verdict_trace_observed.timing
                        if verdict_trace_observed is not None
                        else None
                    ),
                    usage=(
                        verdict_trace_observed.usage
                        if verdict_trace_observed is not None
                        else None
                    ),
                    budget=(
                        verdict_trace_observed.budget
                        if verdict_trace_observed is not None
                        else TraceBudget.between(run.budget, run.budget)
                    ),
                    metadata={
                        "pass": passed,
                        "inconclusive": verdict_inconclusive,
                        "refinement_exhausted": (
                            not passed
                            and not can_refine
                            and not verdict_inconclusive
                        ),
                    },
                )
            )
            await self._emit_intermediate(event_sink, verifier_intermediate)
            if passed or not can_refine:
                break
            depth += 1
            refined = await run_prompt_work(
                self._refinement_prompt,
                base_prompt,
                text,
                verdict,
            )
            prompt = f"{refined}{spec.prompt_suffix}"
        if (
            is_final_unit
            and not self._has_public_output(
                run.outputs.get(spec.name, ""),
                run.final_completions,
            )
            and not self._head_committed_text(run)
        ):
            # An empty public answer after the bounded retry is a failure the
            # caller must see, never a silent stop with empty content or a
            # substituted internal stage (issue #496). A committed head is
            # exempt: EO-D7 already defines the head-only best-so-far answer.
            run.final_unit_unusable = True
            run.trace.append(
                self._trace_event(
                    spec,
                    "failed",
                    operation="generation",
                    status="failed",
                    attempt=depth,
                    detail="empty_output",
                    error=TraceError(type="EmptyFinalOutput"),
                )
            )
        run.completion_order.append(spec.name)

    async def _run_unit_safe(
        self,
        run: _RunState,
        session: str,
        query: str,
        spec: RoleSpec,
        *,
        event_sink: Callable[[ConductorEvent], Awaitable[None]] | None = None,
    ) -> None:
        # A transient backend failure on one unit must not destroy the whole
        # multi-agent run: record it and let the Conductor return the best
        # result produced so far (O4). Sibling units keep their completed work.
        try:
            await self._run_unit(run, session, query, spec, event_sink=event_sink)
        except _ObservedGenerationError:
            self._mark_failed_final_unit(run, spec)
        except Exception as error:
            self._mark_failed_final_unit(run, spec)
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

    def _mark_failed_final_unit(self, run: _RunState, spec: RoleSpec) -> None:
        """A failed selected final unit must not fall back to internal stages.

        Without this the last completed internal stage would be published as
        the answer with finish_reason "stop" (issue #496). A committed head
        keeps its best-so-far contract (EO-D7).
        """

        if (
            spec.name == self._selected_final_unit().name
            and spec.name not in run.outputs
            and not self._head_committed_text(run)
        ):
            run.final_unit_unusable = True

    def _final_text(self, run: _RunState) -> str:
        unit = self._final_output_unit(run)
        return run.outputs[unit.name] if unit is not None else ""

    def _public_text(self, run: _RunState) -> str:
        """The complete public answer: committed head prefix + continuation.

        With an enabled head the committed prefix cannot be retracted, so a
        missing continuation returns the head text alone (best-so-far); only
        an exact verbatim head repetition is deduplicated (EO-D7).
        """

        if self._head is None or not self._head_enabled():
            return self._final_text(run)
        head_text = run.outputs.get(self._head.name, "")
        continuation = run.outputs.get(self._selected_final_unit().name)
        if continuation is None:
            return head_text
        return head_text + dedup_public_continuation(head_text, continuation)

    def _public_completions(
        self,
        run: _RunState,
        public_text: str,
    ) -> tuple[CompletionOutput, ...]:
        """Merged public completion for the split head+continuation stream."""

        if self._head is None or not self._head_enabled():
            return run.final_completions
        if run.public_budget_exhausted:
            # The caller's public limit was consumed before the continuation
            # could run: the committed head is the whole answer and the
            # truncation is reported honestly (issue #496).
            finish_reason = "length"
        else:
            finish_reason = (
                run.final_completions[0].finish_reason
                if run.final_completions
                else None
            )
        return (
            CompletionOutput(
                index=0,
                text=public_text,
                token_ids=(),
                finish_reason=finish_reason or "stop",
            ),
        )

    def _head_exclusions(self, run: _RunState) -> frozenset[str]:
        """Skip a head whose caller intent is incompatible (EO-D7)."""

        if self._head is None or self._head_enabled():
            return frozenset()
        run.trace.append(
            self._trace_event(
                self._head,
                "skipped:intent",
                operation="generation",
                status="skipped",
                attempt=0,
                metadata={"reason": "intent", "head": True},
            )
        )
        return frozenset({self._head.name})

    def _head_committed_text(self, run: _RunState) -> str:
        """Committed head bytes for this run; empty when absent or disabled."""

        if self._head is None or not self._head_enabled():
            return ""
        return run.outputs.get(self._head.name, "")

    def _public_completion_tokens(self, run: _RunState) -> int | None:
        """Backend-reported tokens of the public answer, ``None`` if unknown.

        The public answer is head + selected final unit; either stage missing
        its backend usage makes the total unknowable, and the wire layer then
        falls back to the m9 approximation over the public completion.
        """

        final_tokens = run.final_unit_completion_tokens
        if self._head is None or not self._head_enabled():
            return final_tokens
        if self._head.name not in run.outputs:
            # Zero-byte head failure: the continuation is the whole answer.
            return final_tokens
        if not run.head_usage_reported:
            return None
        head_tokens = run.head_completion_tokens or 0
        if self._selected_final_unit().name not in run.outputs:
            # Continuation skipped or failed: the committed head stands alone.
            return head_tokens
        if final_tokens is None:
            return None
        return head_tokens + final_tokens

    def _final_output_unit(self, run: _RunState) -> RoleSpec | None:
        terminal = self._terminal_units()
        synthesizers = [unit for unit in terminal if unit.role_type == "synthesizer"]
        for unit in synthesizers + terminal:
            if unit.name in run.outputs:
                return unit
        if run.final_unit_unusable:
            # A failed/empty final unit surfaces as an explicit error at the
            # Orchestrator boundary; internal stage text never substitutes
            # for the public answer (issue #496).
            return None
        # Budget exhaustion best-so-far (EO-D7/O4): the last completed
        # generation stage stands in — never an executor's machine output or
        # the head, whose committed text _public_text already carries.
        for name in reversed(run.completion_order):
            candidate = self._by_name[name]
            if candidate.role_type not in {"executor", "head"}:
                return candidate
        return None

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
        event_sink: Callable[[ConductorEvent], Awaitable[None]] | None = None,
    ) -> None:
        pending = {name: set(deps) for name, deps in self._unit_deps.items() if name not in exclude}
        for deps in pending.values():
            deps.difference_update(exclude)
        while pending:
            ready = [name for name, deps in pending.items() if not deps]
            await asyncio.gather(
                *(
                    self._run_unit_safe(
                        run,
                        session,
                        query,
                        self._by_name[name],
                        event_sink=event_sink,
                    )
                    for name in ready
                )
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
        *,
        dedupe_against: str = "",
        bare_deltas: bool = False,
        attempt: int = 0,
    ) -> AsyncIterator[ConductorEvent]:
        remaining = self._public_budget_remaining(run)
        if remaining is not None and remaining < 1:
            run.public_budget_exhausted = True
            run.trace.append(
                self._trace_event(
                    spec,
                    "skipped:public_budget",
                    operation="generation",
                    status="skipped",
                    attempt=0,
                    budget=TraceBudget.between(run.budget, run.budget),
                    metadata={"reason": "public_budget"},
                )
            )
            return
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

        prompt = self._prepared_initial_prompt(run, spec)
        if prompt is None:
            prompt = await run_prompt_work(
                self._rendered_role_prompt,
                spec,
                query,
                dict(run.outputs),
            )
        backend = self._workers[spec.worker]
        request = self._generation_request(run, session, spec, prompt, attempt)
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
                    attempt=attempt,
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
        text_parts: list[str] = []
        legacy_text = ""
        deduper = _ContinuationDeduper(dedupe_against if bare_deltas else "")
        from kairyu.telemetry import traced_span

        try:
            with traced_span(
                "kairyu.conductor.stage",
                {
                    "kairyu.request_id": request.request_id,
                    "kairyu.stage.name": spec.name,
                    "kairyu.stage.role": spec.role_type,
                    "kairyu.stage.worker": spec.worker,
                    "kairyu.stage.operation": "generation",
                    "kairyu.stage.attempt": 0,
                    "kairyu.stream": True,
                },
            ) as span:
                async for partial in backend.stream(request):
                    last_result = partial
                    if partial.usage is not None:
                        latest_usage = partial.usage
                        self._observe_usage(run, latest_usage)
                    completion = partial.completions[0] if partial.completions else None
                    if (
                        completion is not None
                        and completion.text_delta is None
                        and completion.text_offset is None
                    ):
                        text = completion.text
                        if not text.startswith(legacy_text):
                            raise RuntimeError(
                                "final worker stream must emit cumulative, "
                                "prefix-stable text"
                            )
                        delta = text[len(legacy_text) :]
                        legacy_text = text
                        emitted = len(text)
                    else:
                        delta, emitted = (
                            completion.delta_after(emitted)
                            if completion is not None
                            else ("", emitted)
                        )
                    text_parts.append(delta)
                    if first_token_at is None and partial.completions:
                        first_token_at = utc_now_iso()
                    if bare_deltas:
                        # Bare deltas let the server own public offsets across
                        # the head/continuation seam; only n=1 reaches here.
                        deduped = deduper.feed(delta)
                        if deduped:
                            yield ConductorEvent(kind="delta", text=deduped)
                    else:
                        yield ConductorEvent(
                            kind="delta",
                            text=delta,
                            completions=partial.completions,
                        )
                if bare_deltas:
                    tail = deduper.flush()
                    if tail:
                        yield ConductorEvent(kind="delta", text=tail)
                if last_result is None:
                    raise RuntimeError("final worker stream produced no result")
                unit_text, unit_completions = self._unit_public_output(
                    spec,
                    last_result.text or "".join(text_parts),
                    last_result.completions,
                )
                run.outputs[spec.name] = unit_text
                actual_cost = self._cost_model(request, last_result)
                reconciled = run.budget.reconcile_success(
                    cost=actual_cost,
                    unknown_cost=unknown_cost,
                )
                if span is not None:
                    span.set_attribute("kairyu.stage.status", "success")
        except Exception as error:
            if text_parts:
                run.outputs[spec.name] = "".join(text_parts)
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
        run.final_completions = unit_completions
        run.final_unit_completion_tokens = (
            latest_usage.completion_tokens if latest_usage is not None else None
        )

    async def run(
        self,
        query: str,
        budget: Budget | None = None,
        *,
        session: str | None = None,
        prepared_initial_requests: Mapping[str, GenerationRequest] | None = None,
    ) -> ConductorResult:
        if isinstance(query, TemplatedPrompt):
            raise ValueError(
                "Conductor cannot derive role prompts from a tokenizer-owned "
                "pre-rendered chat prompt"
            )
        if prepared_initial_requests and session is None:
            raise ValueError("prepared initial requests require their bound session")
        run = _RunState(
            budget=BudgetState(budget=budget or Budget()),
            prepared_initial_requests=dict(prepared_initial_requests or {}),
        )
        session = session or uuid.uuid4().hex[:12]
        await self._run_pending(run, session, query, exclude=self._head_exclusions(run))
        public_text = self._public_text(run)
        return ConductorResult(
            final_text=public_text,
            completions=self._public_completions(run, public_text),
            outputs=dict(run.outputs),
            budget_state=run.budget,
            trace=tuple(run.trace),
            usage=tuple(run.usage),
            cached_tokens=run.cached_tokens,
            reasoning_content=self._reasoning_content(run),
            public_completion_tokens=self._public_completion_tokens(run),
            final_unit_ok=not run.final_unit_unusable,
        )

    async def stream(
        self,
        query: str,
        budget: Budget | None = None,
        *,
        session: str | None = None,
        prepared_initial_requests: Mapping[str, GenerationRequest] | None = None,
    ) -> AsyncIterator[ConductorEvent]:
        """Run pre-final DAG waves, then pull final backend deltas directly.

        A verifier attached to the final role would make early deltas
        provisional and therefore impossible to retract in OpenAI SSE.  Such a
        DAG is rejected explicitly; the supported shape verifies drafts before
        an unverified final worker/synthesizer.
        """

        if isinstance(query, TemplatedPrompt):
            raise ValueError(
                "Conductor cannot derive role prompts from a tokenizer-owned "
                "pre-rendered chat prompt"
            )
        if prepared_initial_requests and session is None:
            raise ValueError("prepared initial requests require their bound session")
        run = _RunState(
            budget=BudgetState(budget=budget or Budget()),
            prepared_initial_requests=dict(prepared_initial_requests or {}),
        )
        session = session or uuid.uuid4().hex[:12]
        final = self._stream_final_unit()
        head_active = self._head is not None and self._head_enabled()
        exclude = frozenset({final.name}) | self._head_exclusions(run)

        def partial_result(final_text: str) -> ConductorResult:
            return ConductorResult(
                final_text=final_text,
                completions=self._public_completions(run, final_text)
                if head_active
                else run.final_completions,
                outputs=dict(run.outputs),
                budget_state=run.budget,
                trace=tuple(run.trace),
                usage=tuple(run.usage),
                cached_tokens=run.cached_tokens,
                reasoning_content=self._reasoning_content(
                    run,
                    include_final_attribution=False,
                ),
                public_completion_tokens=self._public_completion_tokens(run),
                final_unit_ok=not run.final_unit_unusable,
            )

        # Phase A (EO-D7/EO-D8): one producer task runs the pre-final DAG.  The
        # head unit streams committed public deltas and every completed stage
        # streams its reasoning section as it finishes, merged through one
        # bounded queue.  Per-event overhead is microseconds against a
        # multi-millisecond token cadence; the single-producer continuation
        # tail below stays pull-through (m11 D1).
        emitted_text = ""
        queue: asyncio.Queue[ConductorEvent | None] = asyncio.Queue(maxsize=256)

        async def sink(event: ConductorEvent) -> None:
            await queue.put(event)

        async def pump() -> None:
            try:
                await self._run_pending(
                    run,
                    session,
                    query,
                    exclude=exclude,
                    event_sink=sink,
                )
            except asyncio.CancelledError:
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass
                raise
            except BaseException:
                await queue.put(None)
                raise
            else:
                await queue.put(None)

        producer = asyncio.create_task(pump())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if item.kind == "delta":
                    emitted_text += item.text
                yield item
            await producer
        except BaseException as error:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            if isinstance(error, Exception):
                raise ConductorStreamError(
                    error,
                    partial_result(
                        emitted_text if head_active else self._final_text(run)
                    ),
                ) from error
            raise

        # Phase B: pull the continuation directly from its backend iterator.
        head_text = (
            run.outputs.get(self._head.name, "")
            if head_active and self._head is not None
            else ""
        )
        try:
            for final_attempt in range(2):
                async for event in self._stream_unit(
                    run,
                    session,
                    query,
                    final,
                    dedupe_against=head_text,
                    bare_deltas=head_active,
                    attempt=final_attempt,
                ):
                    if event.kind == "delta":
                        emitted_text += event.text
                    yield event
                if (
                    self._has_public_output(
                        run.outputs.get(final.name, ""),
                        run.final_completions,
                    )
                    or final.name not in run.outputs
                    or self._head_committed_text(run)
                ):
                    # Non-empty answer, budget-skip (no dispatch), or a
                    # committed head-only answer: no empty-output retry.
                    break
                if final_attempt == 0:
                    run.trace.append(
                        self._trace_event(
                            final,
                            "retry:empty_output",
                            operation="generation",
                            status="retried",
                            attempt=0,
                            detail="empty public output",
                            budget=TraceBudget.between(run.budget, run.budget),
                            metadata={"reason": "empty_output"},
                        )
                    )
        except Exception as error:
            raise ConductorStreamError(
                error,
                partial_result(emitted_text if head_active else self._final_text(run)),
            ) from error
        if (
            not self._has_public_output(
                run.outputs.get(final.name, ""),
                run.final_completions,
            )
            and final.name in run.outputs
            and not self._head_committed_text(run)
        ):
            run.final_unit_unusable = True
            run.trace.append(
                self._trace_event(
                    final,
                    "failed",
                    operation="generation",
                    status="failed",
                    attempt=1,
                    detail="empty_output",
                    error=TraceError(type="EmptyFinalOutput"),
                )
            )
        final_text = self._public_text(run)
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
                completions=self._public_completions(run, final_text),
                outputs=dict(run.outputs),
                budget_state=run.budget,
                trace=tuple(run.trace),
                usage=tuple(run.usage),
                cached_tokens=run.cached_tokens,
                reasoning_content=self._reasoning_content(
                    run,
                    include_final_attribution=False,
                ),
                public_completion_tokens=self._public_completion_tokens(run),
                final_unit_ok=not run.final_unit_unusable,
            ),
        )
