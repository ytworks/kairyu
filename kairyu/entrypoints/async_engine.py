"""vLLM AsyncLLMEngine-signature-compatible async entrypoint (design doc D2).

Wraps any EngineBackend; ``generate`` is an async generator of incremental
``RequestOutput`` snapshots, finishing with ``finished=True``, matching vLLM's
streaming contract.
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from kairyu.engine.backend import EngineBackend, GenerationRequest
from kairyu.outputs import RequestOutput
from kairyu.sampling_params import SamplingParams


@dataclass(frozen=True)
class AsyncEngineArgs:
    """Subset of vLLM's AsyncEngineArgs; unknown extras go in ``extra_args``."""

    model: str
    tokenizer: str | None = None
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    seed: int | None = 0
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool | None = None
    trust_remote_code: bool = False
    extra_args: dict = field(default_factory=dict, compare=False)


def _default_backend(args: AsyncEngineArgs) -> EngineBackend:
    if importlib.util.find_spec("vllm") is not None:
        from kairyu.engine.vllm_backend import VLLMBackend

        return VLLMBackend(model=args.model, enable_prefix_caching=args.enable_prefix_caching)
    from kairyu.engine.mock import MockBackend

    return MockBackend()


class AsyncLLMEngine:
    def __init__(self, backend: EngineBackend, model: str = "") -> None:
        self._backend = backend
        self.model = model
        self._aborted: set[str] = set()

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs) -> AsyncLLMEngine:
        return cls(backend=_default_backend(engine_args), model=engine_args.model)

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
    ) -> AsyncIterator[RequestOutput]:
        request = GenerationRequest(
            request_id=request_id, prompt=prompt, sampling_params=sampling_params
        )
        async for partial in self._backend.stream(request):
            if request_id in self._aborted:
                self._aborted.discard(request_id)
                return
            yield RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=(),
                outputs=partial.completions,
                finished=partial.finished,
            )

    async def abort(self, request_id: str) -> None:
        """Stop streaming a request; unknown ids are ignored (vLLM behavior)."""
        self._aborted.add(request_id)

    async def shutdown(self) -> None:
        await self._backend.shutdown()
