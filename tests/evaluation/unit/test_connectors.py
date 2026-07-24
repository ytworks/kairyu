import base64
import json

import httpx
import pytest
from pydantic import ValidationError

import kairyu.evaluation.connectors as connector_module
from kairyu.evaluation.connectors import (
    ConnectorError,
    ConnectorErrorCode,
    ConnectorImagePart,
    ConnectorImageURL,
    ConnectorJSONSchema,
    ConnectorMessage,
    ConnectorRequest,
    ConnectorResponse,
    ConnectorResponseFormat,
    ConnectorResult,
    ConnectorTextPart,
    ConnectorUsage,
    FakeOpenAIConnector,
    OpenAICompatibleConnector,
    canonical_connector_request_json,
    canonical_connector_request_sha256,
    normalize_openai_base_url,
)
from kairyu.evaluation.safety import SecretSafetyError, SecretValueRegistry


def _request(request_id="request-1"):
    return ConnectorRequest(
        request_id=request_id,
        model="model-a",
        messages=(
            ConnectorMessage(role="system", content=" preserve system whitespace "),
            ConnectorMessage(role="user", content="Question\nA) one\nB) two"),
        ),
        temperature=0.0,
        top_p=0.9,
        max_tokens=128,
        seed=7,
        timeout_seconds=5,
    )


def _success_result(request_id="request-1", content="ANSWER: B"):
    return ConnectorResult(
        response=ConnectorResponse(
            request_id=request_id,
            content=content,
            finish_reason="stop",
            provider_request_id="provider-1",
            provider_model="served-model",
            usage=ConnectorUsage(
                prompt_tokens=10,
                completion_tokens=3,
                total_tokens=13,
            ),
            latency_seconds=0.25,
        )
    )


def _clock(*values):
    iterator = iter(values)
    return lambda: next(iterator)


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_data_url(extra=b""):
    encoded = base64.b64encode(_PNG_SIGNATURE + extra).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _multimodal_request(request_id="vision-request-1", *, reasoning_effort=None):
    return ConnectorRequest(
        request_id=request_id,
        model="vision-model-a",
        messages=(
            ConnectorMessage(role="system", content="Return a grounded verdict."),
            ConnectorMessage(
                role="user",
                content=(
                    ConnectorTextPart(text="Inspect this image."),
                    ConnectorImagePart(
                        image_url=ConnectorImageURL(
                            url=_png_data_url(b"fixture"),
                            detail="low",
                        )
                    ),
                ),
            ),
        ),
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        seed=11,
        timeout_seconds=7,
        reasoning_effort=reasoning_effort,
        response_format=ConnectorResponseFormat(
            json_schema=ConnectorJSONSchema(
                name="hle_judge_result",
                strict=True,
                schema={
                    "required": ["correct", "explanation"],
                    "additionalProperties": False,
                    "properties": {
                        "explanation": {"type": "string"},
                        "correct": {"type": "boolean"},
                    },
                    "type": "object",
                },
            )
        ),
    )


def test_request_response_schemas_are_frozen_validated_and_preserve_messages():
    request = _request()

    assert len(request.messages) == 2
    assert request.messages[0].content == " preserve system whitespace "
    with pytest.raises(ValidationError, match="frozen"):
        request.model = "changed"
    with pytest.raises(ValidationError):
        ConnectorRequest(
            request_id="request-1",
            model="model-a",
            messages=(),
        )
    with pytest.raises(ValidationError):
        ConnectorUsage(prompt_tokens=2, completion_tokens=2, total_tokens=1)
    with pytest.raises(ValidationError):
        ConnectorResult()
    with pytest.raises(ValidationError):
        ConnectorResult(
            response=_success_result().response,
            error=ConnectorError(
                request_id="request-1",
                code="malformed",
                detail="bad",
            ),
        )
    with pytest.raises(SecretSafetyError):
        ConnectorMessage(
            role="user",
            content="Authorization: Bearer abcdefghijklmnop",
        )


def test_fake_connector_is_offline_fixed_by_request_id_and_cancellable():
    connector = FakeOpenAIConnector(
        {
            "request-1": _success_result(),
            "request-2": _success_result("request-2", "ANSWER: A"),
        }
    )

    first = connector.complete(_request("request-1"))
    second = connector.complete(_request("request-2"))
    cancelled = connector.complete(_request("request-1"), cancel_requested=lambda: True)
    missing = connector.complete(_request("unknown"))

    assert first.response.content == "ANSWER: B"
    assert second.response.content == "ANSWER: A"
    assert [request.request_id for request in connector.requests] == [
        "request-1",
        "request-2",
        "unknown",
    ]
    assert cancelled.error.code is ConnectorErrorCode.CANCELLED
    assert missing.error.code is ConnectorErrorCode.MALFORMED


def test_fake_connector_enforces_empty_opt_in_and_per_call_attempt_budget():
    empty = _success_result(content="")
    retried = _success_result().model_copy(
        update={"response": _success_result().response.model_copy(update={"attempts": 3})}
    )
    connector = FakeOpenAIConnector(
        {
            "request-1": empty,
            "request-2": retried.model_copy(
                update={"response": retried.response.model_copy(update={"request_id": "request-2"})}
            ),
        }
    )

    rejected_empty = connector.complete(_request())
    accepted_empty = connector.complete(_request().model_copy(update={"allow_empty_content": True}))
    accepted_budget = connector.complete(_request("request-2"), max_attempts=3)
    rejected_budget = connector.complete(_request("request-2"), max_attempts=2)

    assert rejected_empty.error.code is ConnectorErrorCode.EMPTY_RESPONSE
    assert accepted_empty.response.content == ""
    assert accepted_budget.response.attempts == 3
    assert rejected_budget.error.code is ConnectorErrorCode.MALFORMED
    assert rejected_budget.error.attempts == 2
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            connector.complete(_request(), max_attempts=invalid)


def test_fake_connector_rejects_mismatched_or_secret_bearing_fixed_results():
    with pytest.raises(ValueError, match="must match"):
        FakeOpenAIConnector({"different": _success_result()})

    secret = "fixed-response-secret-value"
    registry = SecretValueRegistry([secret])
    with pytest.raises(SecretSafetyError):
        FakeOpenAIConnector(
            {"request-1": _success_result(content=f"ANSWER: B {secret}")},
            secret_registry=registry,
        )


@pytest.mark.parametrize(
    "endpoint",
    (
        "https://user:password@example.test/v1",
        "https://example.test/v1?api_key=secret-value",
        "https://example.test/v1?safe=value",
        "https://example.test/v1#fragment",
        "ftp://example.test/v1",
        "https:///missing-host",
        "https://example.test/v1/chat/completions",
    ),
)
def test_endpoint_rejects_credentials_queries_fragments_and_non_http(endpoint):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        with pytest.raises(ValueError) as raised:
            OpenAICompatibleConnector(endpoint, client=client)

    assert "secret-value" not in str(raised.value)
    assert "password" not in str(raised.value)


def test_success_maps_multi_message_wire_usage_ids_finish_reason_and_latency():
    captured = {}
    secret = "ordinary-provider-secret-value"
    registry = SecretValueRegistry()

    def handler(request):
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-request-id": "provider-header-id"},
            json={
                "id": "provider-body-id",
                "model": "served-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ANSWER: B"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        connector = OpenAICompatibleConnector(
            "https://example.test",
            client=client,
            secret_env_name="MODEL_CREDENTIAL",
            secret_resolver=lambda name: secret if name == "MODEL_CREDENTIAL" else None,
            secret_registry=registry,
            max_retries=0,
            clock=_clock(10.0, 10.25),
        )
        result = connector.complete(_request())

    assert captured == {
        "url": "https://example.test/v1/chat/completions",
        "authorization": f"Bearer {secret}",
        "body": {
            "model": "model-a",
            "messages": [
                {"role": "system", "content": " preserve system whitespace "},
                {"role": "user", "content": "Question\nA) one\nB) two"},
            ],
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 128,
            "seed": 7,
            "stream": False,
        },
    }
    assert registry.matches(secret)
    assert result.error is None
    assert result.response.content == "ANSWER: B"
    assert result.response.finish_reason == "length"
    assert result.response.provider_request_id == "provider-header-id"
    assert result.response.provider_model == "served-model"
    assert result.response.usage.total_tokens == 16
    assert result.response.latency_seconds == 0.25
    assert result.response.attempts == 1
    assert secret not in result.model_dump_json()
    assert secret not in repr(result)
    assert secret not in repr(connector)


def test_missing_runtime_credential_is_structured_and_makes_no_http_call():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        connector = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            secret_env_name="MISSING_CREDENTIAL",
            secret_resolver=lambda _: None,
            clock=_clock(1.0, 1.1),
        )
        result = connector.complete(_request())

    assert calls == []
    assert result.error.code is ConnectorErrorCode.CREDENTIAL_MISSING
    assert result.error.attempts == 0


def test_rate_limit_retries_are_bounded_with_injected_backoff():
    calls = []
    backoffs = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429, text="do not persist this response body")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "ANSWER: A"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        connector = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_retries=2,
            backoff=backoffs.append,
            clock=_clock(0.0, 0.3),
        )
        result = connector.complete(_request())

    assert len(calls) == 3
    assert backoffs == [1, 2]
    assert result.response.attempts == 3
    assert result.response.content == "ANSWER: A"


def test_per_call_attempt_budget_caps_configured_retries_and_stays_off_wire():
    calls = []
    backoffs = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, text="untrusted retry response")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_retries=4,
            backoff=backoffs.append,
            clock=_clock(0.0, 0.2),
        ).complete(_request(), max_attempts=2)

    assert len(calls) == 2
    assert backoffs == [1]
    assert result.error.code is ConnectorErrorCode.RATE_LIMIT
    assert result.error.attempts == 2
    assert "max_attempts" not in json.loads(calls[0].content)


def test_exhausted_rate_limit_is_structured_without_response_body():
    secret_body = "rate-limit-body-must-not-leak"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                429,
                headers={"x-request-id": "provider-429"},
                text=secret_body,
            )
        )
    ) as client:
        connector = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_retries=1,
            backoff=lambda _: None,
            clock=_clock(0.0, 0.2),
        )
        result = connector.complete(_request())

    assert result.error.code is ConnectorErrorCode.RATE_LIMIT
    assert result.error.attempts == 2
    assert result.error.status_code == 429
    assert result.error.provider_request_id == "provider-429"
    assert secret_body not in result.model_dump_json()
    assert secret_body not in repr(result)


def test_timeout_retries_are_bounded_and_structured():
    attempts = []
    backoffs = []

    def handler(request):
        attempts.append(request)
        raise httpx.ReadTimeout("timeout body that must not be persisted", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        connector = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_retries=1,
            backoff=backoffs.append,
            clock=_clock(0.0, 0.4),
        )
        result = connector.complete(_request())

    assert len(attempts) == 2
    assert backoffs == [1]
    assert result.error.code is ConnectorErrorCode.TIMEOUT
    assert result.error.attempts == 2
    assert "timeout body" not in result.model_dump_json()


@pytest.mark.parametrize(
    ("response", "code"),
    (
        (httpx.Response(200, content=b"not-json"), ConnectorErrorCode.MALFORMED),
        (httpx.Response(200, json={"choices": []}), ConnectorErrorCode.EMPTY_RESPONSE),
        (
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
            ),
            ConnectorErrorCode.EMPTY_RESPONSE,
        ),
        (
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": 123}, "finish_reason": "stop"}]},
            ),
            ConnectorErrorCode.MALFORMED,
        ),
        (httpx.Response(503, text="unsafe upstream body"), ConnectorErrorCode.HTTP_ERROR),
    ),
)
def test_malformed_empty_and_http_errors_are_structured(response, code):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: response),
    ) as client:
        connector = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_retries=0,
            clock=_clock(0.0, 0.1),
        )
        result = connector.complete(_request())

    assert result.error.code is code
    assert result.error.attempts == 1
    assert "unsafe upstream body" not in result.model_dump_json()


def test_empty_content_opt_in_is_local_and_changes_canonical_evidence():
    wire_bodies = []

    def handler(request):
        wire_bodies.append(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
        )

    default_request = _request()
    allowed_request = default_request.model_copy(update={"allow_empty_content": True})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        connector = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_retries=0,
            clock=_clock(0.0, 0.1, 1.0, 1.1),
        )
        rejected = connector.complete(default_request)
        accepted = connector.complete(allowed_request)

    assert rejected.error.code is ConnectorErrorCode.EMPTY_RESPONSE
    assert accepted.response.content == ""
    assert wire_bodies[0] == wire_bodies[1]
    assert "allow_empty_content" not in json.loads(wire_bodies[0])
    assert canonical_connector_request_sha256(default_request) != (
        canonical_connector_request_sha256(allowed_request)
    )
    assert (
        json.loads(canonical_connector_request_json(default_request))["allow_empty_content"]
        is False
    )
    assert (
        json.loads(canonical_connector_request_json(allowed_request))["allow_empty_content"] is True
    )


def test_response_byte_limit_is_enforced_before_json_parse():
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": "x" * 1_000},
                    "finish_reason": "stop",
                }
            ]
        }
    ).encode()
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body))
    ) as client:
        connector = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_response_bytes=100,
            max_retries=0,
            clock=_clock(0.0, 0.1),
        )
        result = connector.complete(_request())

    assert result.error.code is ConnectorErrorCode.RESPONSE_TOO_LARGE
    assert result.error.attempts == 1
    assert "x" * 100 not in result.model_dump_json()


def test_cancel_is_checked_before_call_after_response_and_before_retry():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        before = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            clock=_clock(0.0, 0.1),
        ).complete(_request(), cancel_requested=lambda: True)
    assert before.error.code is ConnectorErrorCode.CANCELLED
    assert before.error.attempts == 0
    assert calls == []

    cancelled = [False]

    def cancel_during_response(request):
        cancelled[0] = True
        return httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})

    with httpx.Client(transport=httpx.MockTransport(cancel_during_response)) as client:
        after = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            clock=_clock(0.0, 0.1),
        ).complete(_request(), cancel_requested=lambda: cancelled[0])
    assert after.error.code is ConnectorErrorCode.CANCELLED
    assert after.error.attempts == 1

    retry_calls = []
    stop_retry = [False]

    def rate_limited(request):
        retry_calls.append(request)
        return httpx.Response(429)

    def backoff(_):
        stop_retry[0] = True

    with httpx.Client(transport=httpx.MockTransport(rate_limited)) as client:
        retry_cancelled = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_retries=3,
            backoff=backoff,
            clock=_clock(0.0, 0.1),
        ).complete(_request(), cancel_requested=lambda: stop_retry[0])
    assert retry_cancelled.error.code is ConnectorErrorCode.CANCELLED
    assert retry_cancelled.error.attempts == 1
    assert len(retry_calls) == 1


def test_runtime_secret_in_endpoint_or_request_id_is_rejected_before_http():
    secret = "ordinary-runtime-secret-value"
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        unsafe_endpoint = OpenAICompatibleConnector(
            f"https://example.test/{secret}",
            client=client,
            secret_env_name="MODEL_CREDENTIAL",
            secret_resolver=lambda _: secret,
            clock=_clock(0.0, 0.1),
        ).complete(_request())
        unsafe_request_id = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            secret_env_name="MODEL_CREDENTIAL",
            secret_resolver=lambda _: secret,
            clock=_clock(0.0, 0.1),
        ).complete(_request(secret))

    assert calls == []
    assert unsafe_endpoint.error.code is ConnectorErrorCode.MALFORMED
    assert unsafe_request_id.error.code is ConnectorErrorCode.MALFORMED
    assert unsafe_request_id.request_id == "unsafe-request"
    for result in (unsafe_endpoint, unsafe_request_id):
        assert secret not in result.model_dump_json()
        assert secret not in repr(result)


def test_registered_secret_in_request_or_response_never_leaks():
    secret = "provider-runtime-canary-value"
    registry = SecretValueRegistry()
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "id": secret,
                "choices": [
                    {
                        "message": {"content": f"ANSWER: A {secret}"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        connector = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            secret_env_name="MODEL_CREDENTIAL",
            secret_resolver=lambda _: secret,
            secret_registry=registry,
            max_retries=0,
            clock=_clock(0.0, 0.1, 0.2, 0.3),
        )
        unsafe_response = connector.complete(_request())
        unsafe_request = connector.complete(
            _request("request-2").model_copy(
                update={"messages": (ConnectorMessage(role="user", content=f"question {secret}"),)}
            )
        )

    assert len(calls) == 1
    assert unsafe_response.error.code is ConnectorErrorCode.MALFORMED
    assert unsafe_request.error.code is ConnectorErrorCode.MALFORMED
    for result in (unsafe_response, unsafe_request):
        assert secret not in result.model_dump_json()
        assert secret not in repr(result)


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    (
        ("https://example.test", "https://example.test/v1"),
        ("https://example.test/", "https://example.test/v1"),
        ("https://example.test/provider/", "https://example.test/provider/v1"),
        ("https://example.test/provider/v1/", "https://example.test/provider/v1"),
        ("http://localhost:8000", "http://localhost:8000/v1"),
        ("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000/v1"),
        ("http://[::1]:8000/provider/", "http://[::1]:8000/provider/v1"),
    ),
)
def test_openai_base_url_is_canonical_and_idempotent(endpoint, expected):
    canonical = normalize_openai_base_url(endpoint)

    assert canonical == expected
    assert normalize_openai_base_url(canonical) == canonical


@pytest.mark.parametrize(
    "endpoint",
    (
        "http://example.test",
        "http://localhost.example.test",
        "http://192.0.2.10:8000/v1",
        "http://0.0.0.0:8000/v1",
    ),
)
def test_plain_http_is_restricted_to_loopback_hosts(endpoint):
    with pytest.raises(ValueError, match="loopback"):
        normalize_openai_base_url(endpoint)


def test_multimodal_structured_wire_body_is_canonical_and_exact():
    captured = {}
    structured_content = '{"correct":true,"explanation":"visible evidence"}'

    def handler(request):
        captured["content"] = request.content
        captured["content_type"] = request.headers["content-type"]
        return httpx.Response(
            200,
            json={
                "model": "served-vision-model",
                "choices": [
                    {
                        "message": {"content": structured_content},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    request = _multimodal_request()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        connector = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_retries=0,
            clock=_clock(2.0, 2.2),
        )
        result = connector.complete(request)

    expected_body = {
        "model": "vision-model-a",
        "messages": [
            {"role": "system", "content": "Return a grounded verdict."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _png_data_url(b"fixture"),
                            "detail": "low",
                        },
                    },
                ],
            },
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "seed": 11,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "hle_judge_result",
                "strict": True,
                "schema": {
                    "required": ["correct", "explanation"],
                    "additionalProperties": False,
                    "properties": {
                        "explanation": {"type": "string"},
                        "correct": {"type": "boolean"},
                    },
                    "type": "object",
                },
            },
        },
    }
    expected_wire = json.dumps(
        expected_body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    assert captured == {
        "content": expected_wire,
        "content_type": "application/json",
    }
    assert result.error is None
    assert result.response.content == structured_content
    assert result.response.provider_model == "served-vision-model"
    assert result.response.attempts == 1


@pytest.mark.parametrize(
    "reasoning_effort",
    ("none", "minimal", "low", "medium", "high", "xhigh"),
)
def test_reasoning_effort_accepts_only_reviewed_openai_values(reasoning_effort):
    assert (
        _multimodal_request(reasoning_effort=reasoning_effort).reasoning_effort == reasoning_effort
    )

    payload = _multimodal_request().model_dump(mode="json")
    payload["reasoning_effort"] = f"unsupported-{reasoning_effort}"
    with pytest.raises(ValidationError, match="reasoning_effort"):
        ConnectorRequest.model_validate(payload)


def test_reasoning_effort_changes_canonical_evidence_and_openai_wire():
    base = _multimodal_request()
    reasoned = _multimodal_request(reasoning_effort="high")

    assert json.loads(canonical_connector_request_json(base))["reasoning_effort"] is None
    assert json.loads(canonical_connector_request_json(reasoned))["reasoning_effort"] == "high"
    assert canonical_connector_request_sha256(base) != (
        canonical_connector_request_sha256(reasoned)
    )

    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "reasoned response"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_retries=0,
            clock=_clock(3.0, 3.1),
        ).complete(reasoned)

    assert result.error is None
    assert captured["body"]["reasoning_effort"] == "high"


def test_canonical_multimodal_request_evidence_is_stable_and_deeply_frozen():
    first = _multimodal_request()
    payload = first.model_dump(mode="json")
    schema = payload["response_format"]["json_schema"]["schema"]
    payload["response_format"]["json_schema"]["schema"] = dict(reversed(tuple(schema.items())))
    second = ConnectorRequest.model_validate(payload)

    first_canonical = canonical_connector_request_json(first)
    second_canonical = canonical_connector_request_json(second)
    assert first_canonical == second_canonical
    assert canonical_connector_request_sha256(first) == (canonical_connector_request_sha256(second))
    assert len(canonical_connector_request_sha256(first)) == 64
    assert "schema_definition" not in first_canonical
    assert json.loads(first_canonical)["response_format"]["json_schema"]["schema"] == schema

    frozen_schema = first.response_format.json_schema.schema_definition
    with pytest.raises(TypeError, match="immutable"):
        frozen_schema["type"] = "array"
    with pytest.raises(TypeError, match="immutable"):
        frozen_schema["required"].append("new_field")


@pytest.mark.parametrize(
    ("mime_type", "image_bytes"),
    (
        ("image/png", b"\x89PNG\r\n\x1a\n"),
        ("image/jpeg", b"\xff\xd8\xff\xe0"),
        ("image/gif", b"GIF89a"),
        ("image/webp", b"RIFF\x00\x00\x00\x00WEBP"),
    ),
)
def test_inline_image_mime_allowlist_accepts_canonical_data(mime_type, image_bytes):
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    message = ConnectorMessage.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect."},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        }
    )

    assert isinstance(message.content, tuple)
    assert isinstance(message.content[0], ConnectorTextPart)
    assert isinstance(message.content[1], ConnectorImagePart)
    assert message.model_dump(mode="json") == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Inspect."},
            {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": None},
            },
        ],
    }


@pytest.mark.parametrize(
    "url",
    (
        "https://example.test/image.png",
        "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
        "data:image/png;base64,%%%",
        "data:image/png;base64,aGVsbG8=",
        _png_data_url().rstrip("="),
    ),
)
def test_image_url_rejects_remote_disallowed_malformed_and_mismatched_data(url):
    with pytest.raises(ValidationError) as raised:
        ConnectorImageURL(url=url)

    assert url not in str(raised.value)


def test_inline_images_enforce_decoded_per_image_and_aggregate_bounds(monkeypatch):
    monkeypatch.setattr(connector_module, "_MAX_INLINE_IMAGE_BYTES", len(_PNG_SIGNATURE))

    with pytest.raises(ValidationError, match="decoded inline image"):
        ConnectorImageURL(url=_png_data_url(b"x"))

    image = ConnectorImagePart(image_url=ConnectorImageURL(url=_png_data_url()))
    with pytest.raises(ValidationError, match="per-request byte limit"):
        ConnectorRequest(
            request_id="aggregate-images",
            model="vision-model",
            messages=(
                ConnectorMessage(role="user", content=(image,)),
                ConnectorMessage(role="user", content=(image,)),
            ),
        )


def test_image_parts_are_restricted_to_user_messages():
    image = ConnectorImagePart(image_url=ConnectorImageURL(url=_png_data_url()))

    with pytest.raises(ValidationError, match="restricted to user messages"):
        ConnectorMessage(role="system", content=(image,))


def test_structured_format_rejects_non_strict_invalid_or_oversized_schema(monkeypatch):
    with pytest.raises(ValidationError, match="portable characters"):
        ConnectorJSONSchema(name="invalid name", schema={"type": "object"})
    with pytest.raises(ValidationError):
        ConnectorJSONSchema(
            name="not_strict",
            strict=False,
            schema={"type": "object"},
        )
    with pytest.raises(ValidationError, match="finite and serializable"):
        ConnectorJSONSchema(name="non_finite", schema={"const": float("nan")})

    monkeypatch.setattr(connector_module, "_MAX_JSON_SCHEMA_BYTES", 32)
    with pytest.raises(ValidationError, match="byte limit"):
        ConnectorJSONSchema(
            name="too_large",
            schema={"description": "x" * 100},
        )


def test_multimodal_structured_retries_reuse_identical_canonical_body():
    wire_bodies = []
    backoffs = []

    def handler(request):
        wire_bodies.append(request.content)
        if len(wire_bodies) == 1:
            return httpx.Response(429, text="untrusted retry response")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"correct":false,"explanation":"no"}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_retries=1,
            backoff=backoffs.append,
            clock=_clock(0.0, 0.3),
        ).complete(_multimodal_request())

    assert len(wire_bodies) == 2
    assert wire_bodies[0] == wire_bodies[1]
    assert backoffs == [1]
    assert result.response.attempts == 2


def test_multimodal_structured_timeout_is_bounded_without_payload_leakage():
    wire_bodies = []
    backoffs = []

    def handler(request):
        wire_bodies.append(request.content)
        raise httpx.ReadTimeout(
            "timeout body and request payload must not be persisted",
            request=request,
        )

    request = _multimodal_request()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_retries=1,
            backoff=backoffs.append,
            clock=_clock(0.0, 0.4),
        ).complete(request)

    assert len(wire_bodies) == 2
    assert wire_bodies[0] == wire_bodies[1]
    assert backoffs == [1]
    assert result.error.code is ConnectorErrorCode.TIMEOUT
    assert result.error.attempts == 2
    assert _png_data_url(b"fixture") not in result.model_dump_json()
    assert "timeout body" not in result.model_dump_json()


def test_multimodal_response_byte_limit_still_fails_before_json_parse():
    response_body = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": "x" * 1_000},
                    "finish_reason": "stop",
                }
            ]
        }
    ).encode()
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=response_body))
    ) as client:
        result = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            max_response_bytes=100,
            max_retries=0,
            clock=_clock(0.0, 0.1),
        ).complete(_multimodal_request())

    assert result.error.code is ConnectorErrorCode.RESPONSE_TOO_LARGE
    assert result.error.attempts == 1
    assert "x" * 100 not in result.model_dump_json()


def test_registered_secret_hidden_inside_inline_image_is_rejected_before_http():
    secret = "image-runtime-canary-value"
    data_url = _png_data_url(secret.encode())
    request = ConnectorRequest(
        request_id="secret-image-request",
        model="vision-model",
        messages=(
            ConnectorMessage(
                role="user",
                content=(
                    ConnectorTextPart(text="Inspect."),
                    ConnectorImagePart(image_url=ConnectorImageURL(url=data_url)),
                ),
            ),
        ),
    )
    calls = []

    def handler(http_request):
        calls.append(http_request)
        return httpx.Response(200)

    registry = SecretValueRegistry()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = OpenAICompatibleConnector(
            "https://example.test/v1",
            client=client,
            secret_env_name="MODEL_CREDENTIAL",
            secret_resolver=lambda _: secret,
            secret_registry=registry,
            clock=_clock(0.0, 0.1),
        ).complete(request)

    assert calls == []
    assert result.error.code is ConnectorErrorCode.MALFORMED
    assert result.error.attempts == 0
    assert secret not in result.model_dump_json()
    assert data_url not in result.model_dump_json()
    assert secret not in repr(result)
