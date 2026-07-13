"""Shared tenant-aware usage accounting primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from kairyu.engine.backend import GenerationUsage
from kairyu.entrypoints.server.protocol import Usage
from kairyu.outputs import CompletionOutput


class UsageLedgerSink(Protocol):
    def record(
        self,
        tenant: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None: ...


class TokenLimiterSink(Protocol):
    def charge_tokens(self, tenant: str, tokens: int) -> None: ...


def _approx_tokens(text: str) -> int:
    return len(text.split())


def resolve_usage_counts(
    usage: GenerationUsage | Usage | None,
    *,
    prompt: str,
    completions: Sequence[CompletionOutput],
) -> tuple[int, int]:
    """Return backend/wire counts, or derive the existing wire approximation."""
    if usage is not None:
        return usage.prompt_tokens, usage.completion_tokens
    prompt_tokens = _approx_tokens(prompt)
    completion_tokens = sum(
        len(completion.token_ids)
        if completion.token_ids
        else _approx_tokens(completion.text)
        for completion in completions
    )
    return prompt_tokens, completion_tokens


def record_tenant_usage(
    *,
    tenant: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    ledger: UsageLedgerSink | None = None,
    limiter: TokenLimiterSink | None = None,
) -> None:
    """Record one explicit tenant event to whichever sinks are configured."""
    if ledger is not None:
        ledger.record(tenant, model, prompt_tokens, completion_tokens)
    if limiter is not None:
        limiter.charge_tokens(tenant, prompt_tokens + completion_tokens)


class StreamUsageOwner:
    """Accumulate one dispatched stream and finalize its usage at most once."""

    def __init__(
        self,
        *,
        tenant: str,
        model: str,
        prompt: str,
        ledger: UsageLedgerSink | None = None,
        limiter: TokenLimiterSink | None = None,
    ) -> None:
        self._tenant = tenant
        self._model = model
        self._prompt = prompt
        self._ledger = ledger
        self._limiter = limiter
        self._dispatched = False
        self._finalized = False
        self._usage: GenerationUsage | Usage | None = None
        self._completions: dict[int, CompletionOutput] = {}

    def mark_dispatched(self) -> None:
        self._dispatched = True

    @property
    def latest_usage(self) -> GenerationUsage | Usage | None:
        return self._usage

    def observe(
        self,
        usage: GenerationUsage | Usage | None,
        completions: Sequence[CompletionOutput],
    ) -> None:
        if usage is not None:
            self._usage = usage
        for completion in completions:
            self._completions[completion.index] = completion

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if not self._dispatched:
            return
        prompt_tokens, completion_tokens = resolve_usage_counts(
            self._usage,
            prompt=self._prompt,
            completions=tuple(
                self._completions[index] for index in sorted(self._completions)
            ),
        )
        record_tenant_usage(
            tenant=self._tenant,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ledger=self._ledger,
            limiter=self._limiter,
        )


def stream_usage_owner_from_state(
    state: object,
    *,
    tenant: str,
    model: str,
    prompt: str,
) -> StreamUsageOwner:
    """Build one stream owner from optional HTTP app-state sinks."""
    return StreamUsageOwner(
        tenant=tenant,
        model=model,
        prompt=prompt,
        ledger=getattr(state, "usage_ledger", None),
        limiter=getattr(state, "tenant_limiter", None),
    )


def record_state_usage(
    state: object,
    *,
    tenant: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Resolve optional HTTP app-state sinks and record one tenant event."""
    record_tenant_usage(
        tenant=tenant,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ledger=getattr(state, "usage_ledger", None),
        limiter=getattr(state, "tenant_limiter", None),
    )
