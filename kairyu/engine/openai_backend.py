"""External OpenAI-compatible API worker (OpenAI/Anthropic/Gemini compat endpoints).

Used by the Conductor for frontier-tier roles (design m1 D1) and as the remote
DP-replica member behind ReplicaPool (design m6 D2). Errors surface explicitly:
missing API key and non-2xx statuses raise RuntimeError with context.

m6 D2 fixes: one persistent pooled AsyncClient per backend (no per-request
handshake), with optional externally-owned client injection or an owned client
factory for origin-local dynamic replicas; real SSE streaming (cumulative
partials, MockBackend semantics); optional auth (``api_key_env=None`` for
keyless node-to-node replicas); and token-count passthrough (synthetic ids
carrying ``usage.completion_tokens`` / the streamed-delta count, mirroring
MockBackend's count-only ids).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Mapping

import httpx

from kairyu.engine.backend import (
    AdmissionUpperBound,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    UpstreamClientError,
)
from kairyu.engine.backend import (
    admission_upper_bound as default_admission_upper_bound,
)
from kairyu.engine.openai_capabilities import (
    OpenAIRequestCapabilities,
    resolve_openai_capabilities,
)
from kairyu.engine.prompt import (
    MultimodalPrompt,
    TextPrompt,
    prompt_kind,
    prompt_text,
)
from kairyu.engine.registry import register_backend
from kairyu.engine.vision import ImageInputPolicy, InvalidImageInput
from kairyu.outputs import CompletionOutput, TokenLogprob
from kairyu.sampling_params import SamplingParams
from kairyu.sse import iter_sse_data


def _raise_for_status(base_url: str, status_code: int, body: str) -> None:
    """4xx is a client-request error (not a replica health signal, O1); 5xx and
    everything else is a transport/server failure the pool should count."""
    message = f"backend {base_url} returned HTTP {status_code}: {body[:500]}"
    if 400 <= status_code < 500:
        raise UpstreamClientError(message, status_code)
    raise RuntimeError(message)


_DEFAULT_TIMEOUT_S = 60.0
_SSE_DONE = "[DONE]"
# OpenAI exposes token text and bytes in logprobs, but not tokenizer token IDs.
_UNKNOWN_TOKEN_ID = -1


async def _await_task_cancellation_safe(task: asyncio.Task[None]) -> None:
    """Finish an owned-client close before propagating caller cancellation."""

    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        try:
            task.result()
        except BaseException as cleanup_error:
            raise cancelled from cleanup_error
        raise


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


def _message_text(message: Mapping[str, object]) -> str:
    """Preserve native OpenAI tool calls in Kairyu's backend-neutral text form."""

    text = message.get("content")
    parts = [text] if isinstance(text, str) and text else []
    for call in message.get("tool_calls") or ():
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        parts.append(
            "<tool_call>"
            + json.dumps(
                {"name": name, "arguments": arguments},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "</tool_call>"
        )
    return "".join(parts)


class OpenAIRequestValidationError(ValueError):
    """A request cannot be represented by the configured upstream contract."""


def _client_error(upstream: str, message: str) -> OpenAIRequestValidationError:
    return OpenAIRequestValidationError(f"OpenAI-compatible upstream {upstream!r} {message}")


def _active_sampling_fields(params: SamplingParams) -> set[str]:
    """Return non-neutral request intent that an upstream must execute."""

    active: set[str] = set()
    values_and_neutral = {
        "best_of": (params.best_of, None),
        "frequency_penalty": (params.frequency_penalty, 0.0),
        "ignore_eos": (params.ignore_eos, False),
        "logprobs": (params.logprobs, None),
        "max_tokens": (params.max_tokens, None),
        "min_p": (params.min_p, 0.0),
        "min_tokens": (params.min_tokens, 0),
        "n": (params.n, 1),
        "presence_penalty": (params.presence_penalty, 0.0),
        "prompt_logprobs": (params.prompt_logprobs, None),
        "repetition_penalty": (params.repetition_penalty, 1.0),
        "response_format": (params.extra_args.get("response_format"), None)
        if isinstance(params.extra_args, Mapping)
        else (None, None),
        "seed": (params.seed, None),
        "skip_special_tokens": (params.skip_special_tokens, True),
        "stop": (params.stop, ()),
        "stop_token_ids": (params.stop_token_ids, ()),
        "temperature": (params.temperature, 1.0),
        "top_k": (params.top_k, -1),
        "top_p": (params.top_p, 1.0),
    }
    for field, (value, neutral) in values_and_neutral.items():
        if value != neutral:
            active.add(field)
    return active


def _validate_sampling_values(
    params: SamplingParams,
    capabilities: OpenAIRequestCapabilities,
) -> None:
    invalid: list[str] = []
    if params.logprobs is not None and params.logprobs < 0:
        invalid.append("logprobs must be non-negative")
    if params.prompt_logprobs is not None and params.prompt_logprobs < 0:
        invalid.append("prompt_logprobs must be non-negative")
    if params.best_of is not None and params.best_of < params.n:
        invalid.append("best_of must be greater than or equal to n")
    if any(token_id < 0 for token_id in params.stop_token_ids):
        invalid.append("stop_token_ids must be non-negative")
    if capabilities.max_n is not None and params.n > capabilities.max_n:
        invalid.append(f"n must be <= {capabilities.max_n}")
    if (
        capabilities.max_temperature is not None
        and params.temperature > capabilities.max_temperature
    ):
        invalid.append(f"temperature must be <= {capabilities.max_temperature}")
    if invalid:
        raise _client_error(capabilities.upstream, "rejects request: " + "; ".join(invalid))


def _sampling_payload(
    params: SamplingParams,
    capabilities: OpenAIRequestCapabilities,
) -> dict[str, object]:
    extra_args = params.extra_args
    if not isinstance(extra_args, Mapping):
        raise _client_error(capabilities.upstream, "requires extra_args to be a mapping")
    if any(not isinstance(key, str) for key in extra_args):
        raise _client_error(capabilities.upstream, "requires string extra_args keys")

    _validate_sampling_values(params, capabilities)
    active = _active_sampling_fields(params)
    unsupported = active - capabilities.sampling_fields
    if unsupported:
        raise _client_error(
            capabilities.upstream,
            "does not support request fields: " + ", ".join(sorted(unsupported)),
        )

    payload: dict[str, object] = {}
    always = {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "n": params.n,
        "presence_penalty": params.presence_penalty,
        "frequency_penalty": params.frequency_penalty,
    }
    payload.update(
        (field, value)
        for field, value in always.items()
        if field in capabilities.sampling_fields
        and (field in active or field in capabilities.forward_neutral_fields)
    )

    optional = {
        "best_of": params.best_of,
        "ignore_eos": params.ignore_eos if params.ignore_eos else None,
        "min_p": params.min_p if params.min_p != 0.0 else None,
        "min_tokens": params.min_tokens if params.min_tokens != 0 else None,
        "prompt_logprobs": params.prompt_logprobs,
        "repetition_penalty": (
            params.repetition_penalty if params.repetition_penalty != 1.0 else None
        ),
        "seed": params.seed,
        "skip_special_tokens": False if not params.skip_special_tokens else None,
        "stop": list(params.stop) if params.stop else None,
        "stop_token_ids": list(params.stop_token_ids) if params.stop_token_ids else None,
        "top_k": params.top_k if params.top_k != -1 else None,
    }
    payload.update(
        (field, value)
        for field, value in optional.items()
        if value is not None and field in capabilities.sampling_fields
    )
    if params.max_tokens is not None and "max_tokens" in capabilities.sampling_fields:
        payload[capabilities.max_tokens_wire_name] = params.max_tokens
    if params.logprobs is not None and "logprobs" in capabilities.sampling_fields:
        payload["logprobs"] = True
        payload["top_logprobs"] = params.logprobs
    response_format = extra_args.get("response_format")
    if response_format is not None and "response_format" in capabilities.sampling_fields:
        payload["response_format"] = response_format

    vendor_args = {key: value for key, value in extra_args.items() if key != "response_format"}
    unapproved = set(vendor_args) - capabilities.extra_args
    if unapproved:
        fields = ", ".join(f"extra_args.{key}" for key in sorted(unapproved))
        raise _client_error(
            capabilities.upstream,
            f"does not allow vendor extension fields: {fields}",
        )
    payload.update(vendor_args)
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as error:
        raise _client_error(
            capabilities.upstream,
            f"requires JSON-serializable request fields: {error}",
        ) from error
    return payload


def _validate_tools(
    request: GenerationRequest,
    capabilities: OpenAIRequestCapabilities,
) -> None:
    if capabilities.strict_tools:
        return
    for index, tool in enumerate(request.tools):
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("strict") is True:
            raise _client_error(
                capabilities.upstream,
                f"does not support request field tools[{index}].function.strict",
            )


def _validated_request_payload(
    request: GenerationRequest,
    capabilities: OpenAIRequestCapabilities,
) -> dict[str, object]:
    kind = prompt_kind(request.prompt)
    if kind not in capabilities.prompt_kinds:
        raise _client_error(
            capabilities.upstream,
            f"does not support prompt kind: {kind}",
        )
    _validate_tools(request, capabilities)
    if request.priority != 0 and not capabilities.priority:
        raise _client_error(
            capabilities.upstream,
            "does not support request field: priority",
        )
    payload = _sampling_payload(request.sampling_params, capabilities)
    if capabilities.priority:
        payload["priority"] = request.priority
    return payload


class OpenAICompatBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = "OPENAI_API_KEY",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
        request_stream_usage: bool = True,
        upstream: str = "generic",
        capabilities: Mapping[str, object] | OpenAIRequestCapabilities | None = None,
        image_input_policy: Mapping[str, object] | ImageInputPolicy | None = None,
        client: httpx.AsyncClient | None = None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        if client is not None and client_factory is not None:
            raise ValueError("client and client_factory are mutually exclusive")
        if client is not None and transport is not None:
            raise ValueError("client and transport are mutually exclusive")
        if client_factory is not None and transport is not None:
            raise ValueError("client_factory and transport are mutually exclusive")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key_env = api_key_env
        self._timeout_s = timeout_s
        self._transport = transport
        # m9 D1: upstreams only emit the usage chunk when asked; disable for
        # upstreams that reject unknown stream_options
        if type(request_stream_usage) is not bool:
            raise ValueError("request_stream_usage must be a boolean")
        self._request_stream_usage = request_stream_usage
        self._capabilities = resolve_openai_capabilities(upstream, capabilities)
        unsupported_prompt_kinds = self._capabilities.prompt_kinds - {
            "text",
            "multimodal",
        }
        if unsupported_prompt_kinds:
            raise ValueError(
                "OpenAICompatBackend currently implements text and multimodal "
                "chat prompts; "
                "unsupported configured prompt kinds: "
                f"{sorted(unsupported_prompt_kinds)}"
            )
        if isinstance(image_input_policy, ImageInputPolicy):
            self._image_input_policy = image_input_policy
        elif isinstance(image_input_policy, Mapping):
            self._image_input_policy = ImageInputPolicy(**dict(image_input_policy))
        elif image_input_policy is None:
            self._image_input_policy = None
        else:
            raise ValueError(
                "OpenAI-compatible image_input_policy must be a mapping or "
                "ImageInputPolicy"
            )
        supports_multimodal = "multimodal" in self._capabilities.prompt_kinds
        if supports_multimodal and self._image_input_policy is None:
            raise ValueError(
                "OpenAI-compatible multimodal capability requires image_input_policy"
            )
        if not supports_multimodal and self._image_input_policy is not None:
            raise ValueError(
                "OpenAI-compatible image_input_policy requires the multimodal "
                "prompt capability"
            )
        if supports_multimodal and not self._request_stream_usage:
            raise ValueError(
                "OpenAI-compatible multimodal streaming requires "
                "request_stream_usage=true for exact processed prompt usage"
            )
        if self._image_input_policy is not None:
            self._image_input_policy.require_runtime()
        self._client = client
        self._client_factory = client_factory
        # An injected instance has an external owner. A factory creates this
        # backend's origin-local client, so normal replica shutdown owns it.
        self._owns_client = client is None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def base_url(self) -> str:
        """The upstream OpenAI-compatible base (``.../v1``), trailing slash stripped."""
        return self._base_url

    @property
    def capabilities(self) -> OpenAIRequestCapabilities:
        """Resolved immutable request contract for this upstream."""

        return self._capabilities

    @property
    def request_validation_key(self) -> tuple[
        OpenAIRequestCapabilities,
        ImageInputPolicy | None,
    ] | None:
        """Identity for replicas with exactly equivalent request validation.

        ``ReplicaPool`` may validate one representative for backends that
        explicitly publish the same immutable key.  The OpenAI compatibility
        validator depends only on this resolved capability contract, never on
        the replica address, model instance, or HTTP client. A subclass must
        explicitly opt back in because an overridden validator may depend on
        additional per-instance state.
        """

        if type(self) is not OpenAICompatBackend:
            return None
        return self._capabilities, self._image_input_policy

    def validate_request(self, request: GenerationRequest) -> None:
        """Fail unsupported intent before dispatch, metering, or client creation."""

        _validated_request_payload(request, self._capabilities)
        if isinstance(request.prompt, MultimodalPrompt):
            if type(request.prompt.base) is not str and not isinstance(
                request.prompt.base,
                TextPrompt,
            ):
                raise _client_error(
                    self._capabilities.upstream,
                    "does not support token-based multimodal prompt bases",
                )
            assert self._image_input_policy is not None
            self._image_input_policy.validate_prompt_headers(request.prompt)

    def _dispatch_preflight(
        self,
        request: GenerationRequest,
    ) -> dict[str, object]:
        try:
            self.validate_request(request)
            return _validated_request_payload(request, self._capabilities)
        except InvalidImageInput as error:
            raise UpstreamClientError(
                str(error),
                status_code=400,
                code=error.code,
                public_message=str(error),
            ) from error
        except OpenAIRequestValidationError as error:
            raise UpstreamClientError(
                str(error),
                status_code=400,
                public_message=str(error),
            ) from error

    def admission_upper_bound(self, request: GenerationRequest) -> AdmissionUpperBound:
        """Reserve a configured processor ceiling, never media-byte heuristics."""

        if not isinstance(request.prompt, MultimodalPrompt):
            return default_admission_upper_bound(request)
        policy = self._image_input_policy
        if policy is None:  # Defensive: validation owns the normal error.
            raise ValueError("multimodal admission requires image_input_policy")
        max_tokens = request.sampling_params.max_tokens
        if max_tokens is None:
            raise ValueError("multimodal admission requires a finite max_tokens")
        candidates = max(
            request.sampling_params.n,
            request.sampling_params.best_of or request.sampling_params.n,
        )
        return AdmissionUpperBound(
            tokens=candidates * (policy.max_processed_prompt_tokens + max_tokens),
            refundable_on_exact_usage=candidates == 1,
        )

    async def fetch_backends(self, *, timeout_s: float = 2.0) -> dict | None:
        """Best-effort GET of a kairyu replica's ``/backends`` (m13 introspection).

        When this backend points at a kairyu node (the DP-replica case), the node
        is a kairyu server whose OpenAI base is ``<root>/v1``, so its introspection
        endpoint lives at ``<root>/backends``. Reuses this backend's pooled client
        (and any test transport) and keyless/auth headers. Returns ``None`` for a
        non-kairyu upstream (no ``/backends`` → non-200) or any transport error —
        introspection must never raise into the caller's ``/backends`` handler."""
        root = self._base_url[: -len("/v1")] if self._base_url.endswith("/v1") else self._base_url
        try:
            response = await self._get_client().get(
                f"{root}/backends", headers=self._headers(), timeout=timeout_s
            )
        except (httpx.HTTPError, RuntimeError):  # RuntimeError: missing api key
            return None
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        engines = payload.get("engines")
        if isinstance(engines, list):
            # One replica process may serve several models. This backend routes
            # only to ``self._model``; forwarding topology from another row
            # would make the gateway report an unrelated TP/EP configuration.
            payload = dict(payload)
            payload["engines"] = [
                item
                for item in engines
                if isinstance(item, dict) and item.get("model") == self._model
            ]
        return payload

    def _api_key(self) -> str:
        assert self._api_key_env is not None
        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(
                f"API key environment variable {self._api_key_env!r} is not set "
                f"(required for backend at {self._base_url})"
            )
        return key

    def _headers(self, request: GenerationRequest | None = None) -> dict[str, str]:
        headers = (
            {}
            if self._api_key_env is None
            else {"Authorization": f"Bearer {self._api_key()}"}
        )
        if request is not None and self._capabilities.upstream == "kairyu":
            if request.scheduling_class not in {"interactive", "batch"}:
                raise ValueError(
                    f"invalid scheduling class {request.scheduling_class!r}"
                )
            headers["X-Kairyu-Scheduling-Class"] = request.scheduling_class
        # Propagate only W3C traceparent/tracestate. The helper is a deferred
        # no-op when tracing or OpenTelemetry is unavailable and intentionally
        # does not forward baggage, prompts, outputs, or credentials.
        from kairyu.telemetry import inject_trace_context

        inject_trace_context(headers)
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError(f"backend {self._base_url} is shut down")
        if self._client is not None and self._client.is_closed:
            if not self._owns_client:
                raise RuntimeError("externally owned HTTP client is closed")
            self._client = None
        if self._client is None:
            self._client = (
                self._client_factory()
                if self._client_factory is not None
                else httpx.AsyncClient(
                    timeout=self._timeout_s,
                    transport=self._transport,
                )
            )
            if self._client.is_closed:
                self._client = None
                raise RuntimeError("client_factory returned a closed HTTP client")
        return self._client

    async def _validated_image_urls(
        self,
        request: GenerationRequest,
    ) -> tuple[str, ...] | None:
        prompt = request.prompt
        if not isinstance(prompt, MultimodalPrompt):
            return None
        assert self._image_input_policy is not None
        try:
            # Pillow must decompress the complete raster to reject truncated
            # or malicious compressed inputs. Keep that bounded CPU work off
            # the server event loop and do it once, after tenant admission.
            return await asyncio.to_thread(
                self._image_input_policy.validate_prompt,
                prompt,
            )
        except InvalidImageInput as error:
            raise UpstreamClientError(
                str(error),
                status_code=400,
                code=error.code,
                public_message=str(error),
            ) from error

    async def prepare_request(self, request: GenerationRequest) -> None:
        """Fully verify media once before quota debit or streaming headers."""

        self._dispatch_preflight(request)
        await self._validated_image_urls(request)

    def _payload(
        self,
        request: GenerationRequest,
        *,
        validated: dict[str, object],
        image_urls: tuple[str, ...] | None,
    ) -> dict:
        prompt = request.prompt
        if isinstance(prompt, MultimodalPrompt):
            if image_urls is None:
                raise RuntimeError(
                    "multimodal payload requires fully validated image data"
                )
            messages: list[dict[str, object]] = []
            for message in prompt.messages:
                content: list[dict[str, object]] = []
                for part in message.content:
                    if part.type == "text":
                        content.append({"type": "text", "text": part.text})
                        continue
                    assert part.item_index is not None
                    image_url: dict[str, object] = {
                        "url": image_urls[part.item_index],
                    }
                    if part.detail is not None:
                        image_url["detail"] = part.detail
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        }
                    )
                messages.append({"role": message.role, "content": content})
        else:
            text = prompt_text(prompt)
            assert text is not None
            messages = [{"role": "user", "content": text}]
        payload: dict = {
            "model": self._model,
            "messages": messages,
            **validated,
        }
        if request.tools:
            payload["tools"] = list(request.tools)
            if request.tool_choice is not None:
                payload["tool_choice"] = request.tool_choice
        return payload

    @staticmethod
    def _require_exact_multimodal_usage(
        request: GenerationRequest,
        usage: GenerationUsage | None,
    ) -> None:
        if isinstance(request.prompt, MultimodalPrompt) and (
            usage is None or usage.prompt_tokens < 1
        ):
            raise RuntimeError(
                "multimodal OpenAI-compatible upstream omitted exact processed "
                "prompt usage"
            )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        validated = self._dispatch_preflight(request)
        image_urls = await self._validated_image_urls(request)
        payload = self._payload(
            request,
            validated=validated,
            image_urls=image_urls,
        )
        response = await self._get_client().post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._headers(request),
            timeout=self._timeout_s,
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
                None if raw_content is None else tuple(_token_logprob(item) for item in raw_content)
            )
            completions_list.append(
                CompletionOutput(
                    index=choice.get("index", i),
                    text=_message_text(choice["message"]),
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
        usage = _usage_from(data.get("usage"))
        self._require_exact_multimodal_usage(request, usage)
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=completions,
            usage=usage,
        )

    def _partial(
        self,
        request: GenerationRequest,
        texts: dict[int, str],
        finish: dict[int, str],
        deltas_seen: dict[int, int],
        logprobs: dict[int, list[TokenLogprob]],
        finished: bool,
        usage: GenerationUsage | None = None,
    ) -> GenerationResult:
        completions = tuple(
            CompletionOutput(
                index=index,
                text=texts.get(index, ""),
                token_ids=(tuple(range(deltas_seen.get(index, 0))) if finished else ()),
                finish_reason=finish.get(index, "stop") if finished else None,
                cumulative_logprob=(
                    sum(item.logprob for item in logprobs[index]) if index in logprobs else None
                ),
                logprob_content=(tuple(logprobs[index]) if index in logprobs else None),
            )
            for index in sorted(texts.keys() | finish.keys() | logprobs.keys())
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
        validated = self._dispatch_preflight(request)
        image_urls = await self._validated_image_urls(request)
        payload = self._payload(
            request,
            validated=validated,
            image_urls=image_urls,
        ) | {"stream": True}
        if self._request_stream_usage:
            payload["stream_options"] = {"include_usage": True}
        async with self._get_client().stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._headers(request),
            timeout=self._timeout_s,
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                _raise_for_status(self._base_url, response.status_code, body)
            texts: dict[int, str] = {}
            finish: dict[int, str] = {}
            deltas_seen: dict[int, int] = {}
            logprobs: dict[int, list[TokenLogprob]] = {}
            usage: GenerationUsage | None = None
            async for data_str in iter_sse_data(response):
                data_str = data_str.strip()
                if data_str == _SSE_DONE:
                    break
                chunk = json.loads(data_str)
                if chunk.get("usage"):  # final usage chunk has empty choices
                    usage = _usage_from(chunk["usage"])
                changed = False
                for choice in chunk.get("choices", []):
                    index = choice.get("index", 0)
                    texts.setdefault(index, "")
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        texts[index] = texts.get(index, "") + content
                        deltas_seen[index] = deltas_seen.get(index, 0) + 1
                        changed = True
                    if choice.get("finish_reason"):
                        finish[index] = choice["finish_reason"]
                    raw_logprobs = (choice.get("logprobs") or {}).get("content")
                    if raw_logprobs:
                        logprobs.setdefault(index, []).extend(
                            _token_logprob(item) for item in raw_logprobs
                        )
                        changed = True
                if changed:
                    yield self._partial(
                        request,
                        texts,
                        finish,
                        deltas_seen,
                        logprobs,
                        finished=False,
                    )
        if not texts and not finish:
            raise RuntimeError(f"backend {self._base_url} streamed no choices")
        self._require_exact_multimodal_usage(request, usage)
        yield self._partial(
            request,
            texts,
            finish,
            deltas_seen,
            logprobs,
            finished=True,
            usage=usage,
        )

    async def _shutdown_impl(self) -> None:
        if not self._owns_client:
            return
        client = self._client
        if client is not None and not client.is_closed:
            await client.aclose()
        self._client = None

    async def shutdown(self) -> None:
        task = self._close_task
        if task is None:
            # Terminal before the first await: no concurrent request can
            # recreate a just-detached replica's origin-local client.
            self._closed = True
            task = asyncio.create_task(self._shutdown_impl())
            self._close_task = task
        await _await_task_cancellation_safe(task)


register_backend("openai", OpenAICompatBackend)
