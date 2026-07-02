"""EngineBackend protocol and its request/result types.

Every layer above (Router, Conductor, MoA, LLM entrypoint, server) depends only
on this module, never on a concrete engine. The M2 custom engine plugs in here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams


@dataclass(frozen=True)
class CacheHint:
    """KV-affinity hint plumbed through now, consumed by the M2 Radix KV manager.

    ``session_id`` groups requests of one orchestration; ``prefix_fingerprint``
    identifies the shared prompt prefix so multi-step calls can hit cache.
    """

    session_id: str
    prefix_fingerprint: str = ""


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    cache_hint: CacheHint | None = None


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    prompt: str
    completions: tuple[CompletionOutput, ...]
    finished: bool = True

    @property
    def text(self) -> str:
        """Convenience accessor for the first completion's text."""
        return self.completions[0].text if self.completions else ""


@runtime_checkable
class EngineBackend(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]: ...

    async def shutdown(self) -> None: ...
