"""EngineBackend protocol and its request/result types.

Every layer above (Router, Conductor, MoA, LLM entrypoint, server) depends only
on this module, never on a concrete engine. The M2 custom engine plugs in here.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams


class Shutdownable(Protocol):
    async def shutdown(self) -> None: ...


async def shutdown_all(resources: Iterable[Shutdownable], label: str) -> None:
    """Shutdown each unique resource and report every failed cleanup."""
    unique = list({id(resource): resource for resource in resources}.values())
    results = await asyncio.gather(
        *(resource.shutdown() for resource in unique), return_exceptions=True
    )
    errors: list[Exception] = []
    for resource, result in zip(unique, results, strict=True):
        if isinstance(result, Exception):
            errors.append(result)
        elif isinstance(result, BaseException):
            # ``gather(return_exceptions=True)`` returns a child coroutine's
            # self-cancellation as a value. Do not silently treat an unfinished
            # resource cleanup as success, but keep the public aggregate an
            # ExceptionGroup so ordinary lifecycle handlers can report it.
            error = RuntimeError(
                f"{type(resource).__name__}.shutdown raised "
                f"{type(result).__name__}"
            )
            error.__cause__ = result
            errors.append(error)
    if errors:
        raise ExceptionGroup(f"{label} shutdown failed", errors)


async def shutdown_all_cancellation_safe(
    resources: Iterable[Shutdownable],
    label: str,
) -> None:
    """Finish shutdown of detached resources before propagating cancellation.

    Once an owner removes a resource from its live registry, no later lifecycle
    pass can find it. Shield that final shutdown and preserve cancellation as
    the outward result, chaining any cleanup failure for diagnostics.
    """

    cleanup = asyncio.create_task(shutdown_all(resources, label))
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError as cancelled:
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        try:
            cleanup.result()
        except BaseException as cleanup_error:
            raise cancelled from cleanup_error
        raise


class UpstreamClientError(Exception):
    """A backend rejected the request itself (HTTP 4xx): the client's request
    was bad, NOT a sign the replica is unhealthy. The ReplicaPool must not count
    it as a replica failure, or one malformed client could eject the fleet (O1).
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class CacheHint:
    """KV-affinity hint plumbed through now, consumed by the M2 Radix KV manager.

    ``session_id`` groups requests of one orchestration. ``prefix_fingerprint``
    is an optional trusted ``xxh3-64-v1`` key for the first complete
    256-character shared-prompt chunk. An empty value declares session-only
    affinity; prefix-enabled pools do not speculate about cross-session reuse.
    """

    session_id: str
    prefix_fingerprint: str = ""


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    # vLLM-compatible scheduling priority: smaller integers run first.
    priority: int = 0
    # Bounded semantic class for metrics and trusted Kairyu-to-Kairyu transport.
    # Scheduling itself uses ``priority``; arbitrary priority integers never
    # become metric labels.
    scheduling_class: str = "interactive"
    cache_hint: CacheHint | None = None
    tools: tuple[Mapping[str, object], ...] = ()
    tool_choice: str | Mapping[str, object] | None = None
    tools_in_prompt: bool = False
    # Process-local gateway timestamp, never serialized to an upstream. When
    # present, ReplicaPool records request-ingress-to-selection latency rather
    # than only the pure hash/least-load function cost (G5 F1a).
    placement_started_ns: int | None = None


@dataclass(frozen=True)
class AdmissionUpperBound:
    """Worst-case shared-engine work reserved before dispatch."""

    tokens: int
    # Standard usage counts prompt tokens once even when n/best_of duplicates
    # prefill work.  Only single-candidate requests may refund to wire usage.
    refundable_on_exact_usage: bool


def prompt_with_tool_intent(request: GenerationRequest) -> str:
    """Render native-engine tool intent exactly once when no HF template did."""

    if not request.tools or request.tools_in_prompt or request.tool_choice == "none":
        return request.prompt
    choice = request.tool_choice
    if isinstance(choice, Mapping):
        named = (choice.get("function") or {}).get("name")
        policy = f"You must call the function {named!r}."
    elif choice == "required":
        policy = "You must call one of the available functions."
    else:
        policy = "Call a function when it is useful."
    schemas = json.dumps(
        list(request.tools),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"{request.prompt}\n\nAvailable functions:\n{schemas}\n"
        f"{policy} Emit each call exactly as "
        '<tool_call>{"name":"function_name","arguments":{}}</tool_call>.'
    )


def admission_upper_bound(request: GenerationRequest) -> AdmissionUpperBound:
    """Transport-neutral, no-I/O upper bound for one generation.

    The gateway intentionally does not own model tokenizers.  We count the
    complete native prompt/tool intent, response-format metadata, and a fixed
    chat-template envelope in UTF-8 work units, then multiply both prefill and
    decode by the actual candidate fan-out.  This stays O(request bytes) and
    avoids placing tenant bookkeeping in the scheduler token hot path.
    """

    params = request.sampling_params
    if params.max_tokens is None:
        raise ValueError("tenant-isolated generation requires a finite max_tokens")
    candidates = max(params.n, params.best_of or params.n)
    prompt = prompt_with_tool_intent(request)
    metadata = json.dumps(
        params.extra_args,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    # Covers the fixed role/control tokens inserted by common HF/OpenAI chat
    # templates.  Supplied prompt, tool schemas, and response schema are counted
    # explicitly above.
    fixed_template_envelope = 256
    prompt_upper = max(
        1,
        len(prompt.encode("utf-8"))
        + len(metadata.encode("utf-8"))
        + fixed_template_envelope,
    )
    return AdmissionUpperBound(
        tokens=candidates * (prompt_upper + params.max_tokens),
        refundable_on_exact_usage=candidates == 1,
    )


@dataclass(frozen=True)
class GenerationUsage:
    """Backend-reported token accounting (m9 D1): the source of usage truth.

    ``prompt_tokens`` is counted once per request (not per completion);
    ``completion_tokens`` sums across completions; ``cached_tokens`` is the
    prompt prefix served from the radix cache.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    prompt: str
    completions: tuple[CompletionOutput, ...]
    finished: bool = True
    usage: GenerationUsage | None = None

    @property
    def text(self) -> str:
        """Convenience accessor for the first completion's text."""
        return self.completions[0].text if self.completions else ""


@runtime_checkable
class EngineBackend(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]: ...

    async def shutdown(self) -> None: ...


@dataclass(frozen=True)
class EngineReadiness:
    """An engine's own answer to "could I serve a request right now?".

    Optional: `/readyz` only consults engines that implement ``readiness()``, so
    remote and mock backends are unaffected. It must stay CHEAP — the endpoint is
    polled by load balancers — which means reporting known-fatal state, never
    running a probe generation.

    ``fatal`` separates "stop sending work" from "replace this process". Marking a
    node unready for something it could recover from is a trap: the load balancer
    stops sending work, so the traffic that would prove recovery never arrives.
    An engine should therefore only report unready for a condition nothing
    in-process can undo — and then say so, so `/health` can ask for a restart.

    ``detail`` reaches an UNAUTHENTICATED endpoint. Exception classes and fixed
    strings only; a message can carry an upstream URL, a path, or a credential.
    """

    ready: bool
    detail: str = ""
    fatal: bool = False
