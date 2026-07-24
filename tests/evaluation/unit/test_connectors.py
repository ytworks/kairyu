import json

import httpx
import pytest
from pydantic import ValidationError

from kairyu.evaluation.connectors import (
    ConnectorError,
    ConnectorErrorCode,
    ConnectorMessage,
    ConnectorRequest,
    ConnectorResponse,
    ConnectorResult,
    ConnectorUsage,
    FakeOpenAIConnector,
    OpenAICompatibleConnector,
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
