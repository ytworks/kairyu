"""External OpenAI-compatible API worker (OpenAI/Anthropic/Gemini compat endpoints).

Used by the Conductor for frontier-tier roles (design m1 D1) and as the remote
DP-replica member behind ReplicaPool (design m6 D2). Errors surface explicitly:
missing API key and non-2xx statuses raise RuntimeError with context.

m6 D2 fixes: one persistent pooled AsyncClient per backend (no per-request
handshake), real SSE streaming (cumulative partials, MockBackend semantics),
optional auth (``api_key_env=None`` for keyless node-to-node replicas), and
token-count passthrough (synthetic ids carrying ``usage.completion_tokens`` /
the streamed-delta count, mirroring MockBackend's count-only ids).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx

from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.engine.registry import register_backend
from kairyu.outputs import CompletionOutput

_DEFAULT_TIMEOUT_S = 60.0
_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"


class OpenAICompatBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = "OPENAI_API_KEY",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key_env = api_key_env
        self._timeout_s = timeout_s
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _api_key(self) -> str:
        assert self._api_key_env is not None
        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(
                f"API key environment variable {self._api_key_env!r} is not set "
                f"(required for backend at {self._base_url})"
            )
        return key

    def _headers(self) -> dict[str, str]:
        if self._api_key_env is None:
            return {}
        return {"Authorization": f"Bearer {self._api_key()}"}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout_s, transport=self._transport
            )
        return self._client

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
        response = await self._get_client().post(
            f"{self._base_url}/chat/completions",
            json=self._payload(request),
            headers=self._headers(),
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"backend {self._base_url} returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        choices = data.get("choices", [])
        completion_tokens = (data.get("usage") or {}).get("completion_tokens")
        token_ids: tuple[int, ...] = (
            tuple(range(completion_tokens))
            if completion_tokens is not None and len(choices) == 1
            else ()
        )
        completions = tuple(
            CompletionOutput(
                index=choice.get("index", i),
                text=choice["message"]["content"] or "",
                token_ids=token_ids,
                finish_reason=choice.get("finish_reason"),
            )
            for i, choice in enumerate(choices)
        )
        if not completions:
            raise RuntimeError(f"backend {self._base_url} returned no choices: {data}")
        return GenerationResult(
            request_id=request.request_id, prompt=request.prompt, completions=completions
        )

    def _partial(
        self,
        request: GenerationRequest,
        texts: dict[int, str],
        finish: dict[int, str],
        deltas_seen: dict[int, int],
        finished: bool,
    ) -> GenerationResult:
        completions = tuple(
            CompletionOutput(
                index=index,
                text=texts[index],
                token_ids=tuple(range(deltas_seen[index])) if finished else (),
                finish_reason=finish.get(index, "stop") if finished else None,
            )
            for index in sorted(texts)
        )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=completions,
            finished=finished,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        """Real SSE streaming: yields cumulative partials, then the final result."""
        payload = self._payload(request) | {"stream": True}
        async with self._get_client().stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        ) as response:
            if response.status_code != 200:
                body = (await response.aread())[:500]
                raise RuntimeError(
                    f"backend {self._base_url} returned HTTP {response.status_code}: {body!r}"
                )
            texts: dict[int, str] = {}
            finish: dict[int, str] = {}
            deltas_seen: dict[int, int] = {}
            async for line in response.aiter_lines():
                if not line.startswith(_SSE_DATA_PREFIX):
                    continue
                data_str = line[len(_SSE_DATA_PREFIX) :].strip()
                if data_str == _SSE_DONE:
                    break
                chunk = json.loads(data_str)
                changed = False
                for choice in chunk.get("choices", []):
                    index = choice.get("index", 0)
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        texts[index] = texts.get(index, "") + content
                        deltas_seen[index] = deltas_seen.get(index, 0) + 1
                        changed = True
                    if choice.get("finish_reason"):
                        finish[index] = choice["finish_reason"]
                if changed:
                    yield self._partial(request, texts, finish, deltas_seen, finished=False)
        if not texts:
            raise RuntimeError(f"backend {self._base_url} streamed no content")
        yield self._partial(request, texts, finish, deltas_seen, finished=True)

    async def shutdown(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


register_backend("openai", OpenAICompatBackend)
