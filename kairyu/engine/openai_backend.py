"""External OpenAI-compatible API worker (OpenAI/Anthropic/Gemini compat endpoints).

Used by the Conductor for frontier-tier roles (design doc D1). Errors surface
explicitly: missing API key and non-2xx statuses raise RuntimeError with context.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx

from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.engine.registry import register_backend
from kairyu.outputs import CompletionOutput

_DEFAULT_TIMEOUT_S = 60.0


class OpenAICompatBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key_env = api_key_env
        self._timeout_s = timeout_s
        self._transport = transport

    def _api_key(self) -> str:
        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(
                f"API key environment variable {self._api_key_env!r} is not set "
                f"(required for backend at {self._base_url})"
            )
        return key

    def _payload(self, request: GenerationRequest) -> dict:
        params = request.sampling_params
        payload: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": params.temperature,
            "top_p": params.top_p,
            "n": params.n,
        }
        if params.max_tokens is not None:
            payload["max_tokens"] = params.max_tokens
        if params.stop:
            payload["stop"] = list(params.stop)
        if params.seed is not None:
            payload["seed"] = params.seed
        return payload

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        headers = {"Authorization": f"Bearer {self._api_key()}"}
        async with httpx.AsyncClient(
            timeout=self._timeout_s, transport=self._transport
        ) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                json=self._payload(request),
                headers=headers,
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"backend {self._base_url} returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        completions = tuple(
            CompletionOutput(
                index=choice.get("index", i),
                text=choice["message"]["content"] or "",
                token_ids=(),
                finish_reason=choice.get("finish_reason"),
            )
            for i, choice in enumerate(data.get("choices", []))
        )
        if not completions:
            raise RuntimeError(f"backend {self._base_url} returned no choices: {data}")
        return GenerationResult(
            request_id=request.request_id, prompt=request.prompt, completions=completions
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


register_backend("openai", OpenAICompatBackend)
