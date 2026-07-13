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
from collections.abc import AsyncIterator, Mapping

import httpx

from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    UpstreamClientError,
)
from kairyu.engine.registry import register_backend
from kairyu.outputs import CompletionOutput, TokenLogprob
from kairyu.sampling_params import SamplingParams


def _raise_for_status(base_url: str, status_code: int, body: str) -> None:
    """4xx is a client-request error (not a replica health signal, O1); 5xx and
    everything else is a transport/server failure the pool should count."""
    message = f"backend {base_url} returned HTTP {status_code}: {body[:500]}"
    if 400 <= status_code < 500:
        raise UpstreamClientError(message, status_code)
    raise RuntimeError(message)

_DEFAULT_TIMEOUT_S = 60.0
_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"
# OpenAI exposes token text and bytes in logprobs, but not tokenizer token IDs.
_UNKNOWN_TOKEN_ID = -1


def _usage_from(data: dict | None) -> GenerationUsage | None:
    """Parse an upstream usage object incl. prompt_tokens_details (m9 D1)."""
    if not data:
        return None
    details = data.get("prompt_tokens_details") or {}
    return GenerationUsage(
        prompt_tokens=data.get("prompt_tokens", 0),
        completion_tokens=data.get("completion_tokens", 0),
        cached_tokens=details.get("cached_tokens", 0),
    )


def _token_logprob(raw: dict) -> TokenLogprob:
    raw_bytes = raw.get("bytes")
    return TokenLogprob(
        token=raw["token"],
        token_id=_UNKNOWN_TOKEN_ID,
        logprob=float(raw["logprob"]),
        bytes_=tuple(raw_bytes) if raw_bytes is not None else None,
        top=tuple(_token_logprob(item) for item in raw.get("top_logprobs") or ()),
    )


def _validated_extra_args(params: SamplingParams) -> Mapping[str, object]:
    unsupported: list[str] = []
    if params.best_of is not None:
        unsupported.append("best_of")
    if params.repetition_penalty != 1.0:
        unsupported.append("repetition_penalty")
    if params.stop_token_ids:
        unsupported.append("stop_token_ids")
    if params.min_tokens != 0:
        unsupported.append("min_tokens")
    if params.prompt_logprobs is not None:
        unsupported.append("prompt_logprobs")
    if params.ignore_eos:
        unsupported.append("ignore_eos")
    if not params.skip_special_tokens:
        unsupported.append("skip_special_tokens")
    if params.logprobs is not None and params.logprobs < 0:
        unsupported.append("logprobs")

    extra_args = params.extra_args
    if not isinstance(extra_args, Mapping):
        unsupported.append("extra_args")
        extra_args = {}
    else:
        unsupported.extend(
            f"extra_args.{key}" for key in extra_args if key != "response_format"
        )
    if unsupported:
        fields = ", ".join(unsupported)
        raise UpstreamClientError(
            f"OpenAI-compatible backend does not support fields: {fields}",
            status_code=400,
        )
    return extra_args


class OpenAICompatBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = "OPENAI_API_KEY",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
        request_stream_usage: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key_env = api_key_env
        self._timeout_s = timeout_s
        self._transport = transport
        # m9 D1: upstreams only emit the usage chunk when asked; disable for
        # upstreams that reject unknown stream_options
        self._request_stream_usage = request_stream_usage
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
        extra_args = _validated_extra_args(params)
        payload: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": params.temperature,
            "top_p": params.top_p,
            "n": params.n,
            "presence_penalty": params.presence_penalty,
            "frequency_penalty": params.frequency_penalty,
        }
        if params.max_tokens is not None:
            payload["max_tokens"] = params.max_tokens
        if params.stop:
            payload["stop"] = list(params.stop)
        if params.seed is not None:
            payload["seed"] = params.seed
        if params.top_k != -1:
            payload["top_k"] = params.top_k
        if params.min_p != 0.0:
            payload["min_p"] = params.min_p
        if params.logprobs is not None:
            payload["logprobs"] = True
            payload["top_logprobs"] = params.logprobs
        if "response_format" in extra_args:
            payload["response_format"] = extra_args["response_format"]
        return payload

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = self._payload(request)
        response = await self._get_client().post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        )
        if response.status_code != 200:
            _raise_for_status(self._base_url, response.status_code, response.text)
        data = response.json()
        choices = data.get("choices", [])
        completion_tokens = (data.get("usage") or {}).get("completion_tokens")
        token_ids: tuple[int, ...] = (
            tuple(range(completion_tokens))
            if completion_tokens is not None and len(choices) == 1
            else ()
        )
        completions_list = []
        for i, choice in enumerate(choices):
            raw_content = (choice.get("logprobs") or {}).get("content")
            logprob_content = (
                None
                if raw_content is None
                else tuple(_token_logprob(item) for item in raw_content)
            )
            completions_list.append(
                CompletionOutput(
                    index=choice.get("index", i),
                    text=choice["message"]["content"] or "",
                    token_ids=token_ids,
                    cumulative_logprob=(
                        None
                        if logprob_content is None
                        else sum((item.logprob for item in logprob_content), 0.0)
                    ),
                    finish_reason=choice.get("finish_reason"),
                    logprob_content=logprob_content,
                )
            )
        completions = tuple(completions_list)
        if not completions:
            raise RuntimeError(f"backend {self._base_url} returned no choices: {data}")
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=completions,
            usage=_usage_from(data.get("usage")),
        )

    def _partial(
        self,
        request: GenerationRequest,
        texts: dict[int, str],
        finish: dict[int, str],
        deltas_seen: dict[int, int],
        finished: bool,
        usage: GenerationUsage | None = None,
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
            usage=usage,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        """Real SSE streaming: yields cumulative partials, then the final result."""
        payload = self._payload(request) | {"stream": True}
        if self._request_stream_usage:
            payload["stream_options"] = {"include_usage": True}
        async with self._get_client().stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                _raise_for_status(self._base_url, response.status_code, body)
            texts: dict[int, str] = {}
            finish: dict[int, str] = {}
            deltas_seen: dict[int, int] = {}
            usage: GenerationUsage | None = None
            async for line in response.aiter_lines():
                if not line.startswith(_SSE_DATA_PREFIX):
                    continue
                data_str = line[len(_SSE_DATA_PREFIX) :].strip()
                if data_str == _SSE_DONE:
                    break
                chunk = json.loads(data_str)
                if chunk.get("usage"):  # final usage chunk has empty choices
                    usage = _usage_from(chunk["usage"])
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
        yield self._partial(request, texts, finish, deltas_seen, finished=True, usage=usage)

    async def shutdown(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


register_backend("openai", OpenAICompatBackend)
