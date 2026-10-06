"""Responses inputs the engine maps or refuses: images and remote compaction."""

from __future__ import annotations

import json

import openai
import pytest
from fastapi.testclient import TestClient

from kairyu.engine.backend import AdmissionUpperBound, GenerationResult, GenerationUsage
from kairyu.engine.mock import MockBackend
from kairyu.engine.prompt import MultimodalPrompt
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.outputs import CompletionOutput
from tests.server._legacy_chat import create_legacy_app

_IMAGE = "data:image/png;base64,iVBORw0KGgo="


class VisionBackend(MockBackend):
    """A backend that owns multimodal templating, like a VLM replica."""

    def __init__(self):
        super().__init__()
        self.prompts: list = []

    def validate_request(self, request) -> None:
        return None

    def admission_upper_bound(self, request) -> AdmissionUpperBound:
        return AdmissionUpperBound(tokens=64, refundable_on_exact_usage=True)

    async def generate(self, request):
        self.prompts.append(request.prompt)
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(
                    index=0, text="a red square", token_ids=(1,), finish_reason="stop"
                ),
            ),
            usage=GenerationUsage(prompt_tokens=5, completion_tokens=3),
        )


def _app(tmp_path, backend, **kwargs):
    return create_legacy_app(
        {"m": backend},
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
        **kwargs,
    )


def _sdk(http: TestClient) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(http.base_url) + "/v1", api_key="sk", http_client=http)


def _image_message(*extra: dict) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "what is this?"},
            {"type": "input_image", "image_url": _IMAGE, "detail": "high"},
            *extra,
        ],
    }


def test_input_images_reach_a_vision_backend_in_order(tmp_path):
    backend = VisionBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        response = _sdk(http).responses.create(model="m", input=[_image_message()])

    assert response.output_text == "a red square"
    prompt = backend.prompts[0]
    assert isinstance(prompt, MultimodalPrompt)
    assert [item.data for item in prompt.items] == [_IMAGE]
    parts = prompt.messages[-1].content
    assert [(part.type, part.text, part.detail) for part in parts] == [
        ("text", "what is this?", None),
        ("item", None, "high"),
    ]


class _FailingVisionStream(VisionBackend):
    def __init__(self, error: Exception):
        super().__init__()
        self.error = error

    async def stream(self, request):
        raise self.error
        yield  # pragma: no cover


@pytest.mark.parametrize(
    "error, code",
    [
        (RuntimeError("replica went away"), "server_error"),
        (ValueError("prompt tokens (9) already fill max_model_len (8)"), "context_length_exceeded"),
    ],
    ids=["upstream-failure", "overflow"],
)
def test_an_image_stream_failing_before_usage_still_fails_in_band(tmp_path, error, code):
    with TestClient(_app(tmp_path, _FailingVisionStream(error))) as http:
        stream = http.post(
            "/v1/responses", json={"model": "m", "input": [_image_message()], "stream": True}
        )
    lines = stream.text.splitlines()
    events = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
    assert [event["type"] for event in events[-2:]] == ["error", "response.failed"]
    assert events[-1]["response"]["error"]["code"] == code


_CALL = {"type": "function_call", "call_id": "call_1", "name": "look", "arguments": "{}"}
_OUTPUT = {"type": "function_call_output", "call_id": "call_1", "output": "done"}


@pytest.mark.parametrize(
    "backend, items, param",
    [
        (VisionBackend(), [_image_message(), _CALL, _OUTPUT], "input[0].content[1]"),
        (
            VisionBackend(),
            [_image_message({"type": "input_image", "file_id": "file-1", "detail": "auto"})],
            "input[0].content[2].file_id",
        ),
        (
            VisionBackend(),
            [_image_message({"type": "input_file", "file_data": "JVBERi0="})],
            "input[0].content[2].type",
        ),
        (MockBackend(), [_image_message()], "input[0].content[1]"),
    ],
    ids=["images-beside-tool-history", "file-id", "input-file", "text-only-model"],
)
def test_unrepresentable_inputs_are_typed_400s(tmp_path, backend, items, param):
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post("/v1/responses", json={"model": "m", "input": items})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == param


@pytest.mark.parametrize("auto", [False, True], ids=["engine", "auto"])
def test_compact_returns_user_messages_and_a_resumable_compaction_item(tmp_path, auto):
    backend = MockBackend({"Summarize the conversation": "goal: ship it", "resume": "resumed"})
    app = create_legacy_app(
        {} if auto else {"m": backend},
        orchestrators=(
            {"m": Orchestrator({"tier1": backend, "tier2": backend})} if auto else None
        ),
        settings=ServerSettings(usage_ledger_path=str(tmp_path / "usage.jsonl")),
    )
    history = [
        {"role": "user", "content": "ship it"},
        {"role": "assistant", "content": "working"},
    ]
    with TestClient(app) as http:
        compacted = _sdk(http).responses.compact(model="m", input=history)
        resumed = http.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": [
                    # The SDK types compact output as assistant items, but the
                    # spec returns the kept user messages (input_text parts).
                    *[
                        item.model_dump(exclude_none=True, warnings=False)
                        for item in compacted.output
                    ],
                    {"role": "user", "content": "resume"},
                ],
            },
        )

    assert compacted.object == "response.compaction"
    assert [item.type for item in compacted.output] == ["message", "compaction"]
    assert compacted.output[0].content[0].text == "ship it"
    assert resumed.status_code == 200
    assert any("goal: ship it" in str(prompt) for prompt in backend.prompts_seen)
