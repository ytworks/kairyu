import json

import httpx
import pytest

from kairyu.batch.store import BatchStore
from kairyu.batch.worker import BatchWorker
from kairyu.engine.backend import GenerationResult
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.outputs import CompletionOutput


class RecordingBackend(MockBackend):
    def __init__(
        self,
        *,
        supports_n: bool = True,
        validation_error: str | None = None,
        runtime_error: str | None = None,
        output_text: str | None = None,
    ) -> None:
        super().__init__()
        self.supports_n = supports_n
        self.validation_error = validation_error
        self.runtime_error = runtime_error
        self.output_text = output_text
        self.calls = 0

    def validate_request(self, request) -> None:
        if self.validation_error is not None:
            raise ValueError(self.validation_error)

    async def generate(self, request):
        self.calls += 1
        if self.runtime_error is not None:
            raise RuntimeError(self.runtime_error)
        result = await super().generate(request)
        if self.output_text is None:
            return result
        return GenerationResult(
            request_id=result.request_id,
            prompt=result.prompt,
            completions=(
                CompletionOutput(
                    index=0,
                    text=self.output_text,
                    token_ids=(),
                    finish_reason="stop",
                ),
            ),
            usage=result.usage,
        )


def _body(scenario: str) -> dict:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
    }
    tool = {"type": "function", "function": {"name": "weather", "parameters": {}}}
    if scenario == "invalid_tool_choice":
        body["tool_choice"] = "sometimes"
    elif scenario == "undeclared_named_tool":
        body.update(
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "missing"}},
        )
    elif scenario == "required_tool_unsatisfied":
        body.update(tools=[tool], tool_choice="required")
    elif scenario == "invalid_response_format":
        body["response_format"] = {"type": "yaml"}
    elif scenario == "image_input":
        body["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://x/image"}},
                ],
            }
        ]
    elif scenario == "unsupported_n":
        body["n"] = 2
    elif scenario == "unknown_model":
        body["model"] = "missing"
    elif scenario == "invalid_sampling":
        body["temperature"] = -1
    return body


def _backend(scenario: str) -> RecordingBackend:
    if scenario == "unsupported_n":
        return RecordingBackend(supports_n=False)
    if scenario == "backend_validation":
        return RecordingBackend(validation_error="prompt rejected by backend")
    if scenario == "backend_failure":
        return RecordingBackend(
            runtime_error="http://replica-internal:9000 secret=abc"
        )
    return RecordingBackend(output_text="plain answer")


async def _interactive_error(body: dict, backend: RecordingBackend):
    app = create_app({"m": backend})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post("/v1/chat/completions", json=body)
    return response.status_code, response.json().get("error"), backend.calls


async def _batch_error(tmp_path, body: dict, backend: RecordingBackend):
    store = BatchStore(tmp_path)
    worker = BatchWorker(store, {"m": backend}, max_concurrency=1)
    line = {
        "custom_id": "parity",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    file = store.save_file(json.dumps(line).encode(), "input.jsonl", "batch")
    job = store.create_batch(file.id, "/v1/chat/completions")
    await worker.process(job.id)
    completed = store.get_batch(job.id)
    if completed.error_file_id is None:
        return None, backend.calls
    record = json.loads(store.read_file_content(completed.error_file_id))
    return record["error"], backend.calls


@pytest.mark.parametrize(
    ("scenario", "status_code", "code", "expected_dispatches"),
    [
        ("invalid_tool_choice", 400, "invalid_request", 0),
        ("undeclared_named_tool", 400, "invalid_request", 0),
        ("required_tool_unsatisfied", 502, "tool_choice_not_satisfied", 1),
        ("invalid_response_format", 400, "invalid_request", 0),
        ("image_input", 400, "invalid_request", 0),
        ("unsupported_n", 400, "invalid_request", 0),
        ("unknown_model", 404, "model_not_found", 0),
        ("invalid_sampling", 400, "invalid_request", 0),
        ("backend_validation", 400, "invalid_request", 0),
        ("backend_failure", 502, "backend_error", 1),
    ],
)
async def test_interactive_and_batch_chat_errors_have_parity(
    tmp_path, scenario, status_code, code, expected_dispatches
):
    body = _body(scenario)
    http_backend = _backend(scenario)
    batch_backend = _backend(scenario)

    actual_status, http_error, http_calls = await _interactive_error(body, http_backend)
    batch_error, batch_calls = await _batch_error(tmp_path, body, batch_backend)

    assert actual_status == status_code
    assert http_error is not None
    assert batch_error is not None
    assert batch_error["code"] == http_error["code"] == code
    assert batch_error["type"] == http_error["type"]
    assert batch_error["message"] == http_error["message"]
    assert http_calls == batch_calls == expected_dispatches
    if scenario == "backend_failure":
        serialized = json.dumps((http_error, batch_error))
        assert "replica-internal" not in serialized
        assert "secret=abc" not in serialized
