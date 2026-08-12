import json
import logging

import httpx
import pytest

from kairyu.batch.store import BatchStore
from kairyu.batch.worker import BatchWorker
from kairyu.engine.mock import MockBackend
from kairyu.engine.prompt import MultimodalPrompt, TemplatedPrompt
from kairyu.entrypoints.chat_template import ChatTemplate, render_chat
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    render_prompt,
    validate_chat_input,
    validate_orchestration_chat_input,
)
from kairyu.entrypoints.server.protocol import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPart,
)
from kairyu.orchestration.orchestrator import Orchestrator


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


def _chat_body(model: str = "m") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_low_level_app_rejects_unverified_special_tokens_before_route_startup():
    template = ChatTemplate("{{ bos_token }}{{ messages[0].content }}")

    with pytest.raises(
        ValueError,
        match="without verified values.*bos_token",
    ):
        create_app(
            {"m": MockBackend()},
            orchestrators={"auto": Orchestrator({"tier": MockBackend()})},
            chat_templates={"m": template},
        )


def test_batch_worker_rejects_unverified_special_tokens_at_construction(tmp_path):
    template = ChatTemplate("{{ bos_token }}{{ messages[0].content }}")

    with pytest.raises(
        ValueError,
        match="without verified values.*bos_token",
    ):
        BatchWorker(
            BatchStore(tmp_path),
            {"m": MockBackend()},
            chat_templates={"m": template},
        )


def test_public_render_prompt_rejects_unverified_special_tokens():
    template = ChatTemplate("{{ bos_token }}{{ messages[0].content }}")
    request = ChatCompletionRequest.model_validate(_chat_body())

    with pytest.raises(ChatRequestError, match="without verified values.*bos_token"):
        render_prompt(request, {"m": template})


def test_low_level_app_logs_missing_policy_at_construction(caplog):
    with caplog.at_level(logging.WARNING, logger="kairyu.entrypoints.server.app"):
        create_app({"m": MockBackend()})

    assert "no configured chat rendering policy" in caplog.text
    assert "['m']" in caplog.text


def test_batch_worker_logs_missing_policy_at_construction(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="kairyu.batch"):
        BatchWorker(BatchStore(tmp_path), {"m": MockBackend()})

    assert "no configured chat rendering policy" in caplog.text
    assert "['m']" in caplog.text


def test_low_level_app_rejects_template_and_legacy_policy_overlap():
    with pytest.raises(ValueError, match="both chat_templates and legacy_chat_models"):
        create_app(
            {"m": MockBackend()},
            chat_templates={"m": ChatTemplate("ok")},
            legacy_chat_models={"m"},
        )


def test_orchestration_conversation_json_is_context_not_response_schema():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "PAC1 Antagonistの適応症は？"},
            ],
        }
    )

    validated = validate_orchestration_chat_input(request)

    assert "conversation data, not a response schema" in validated.prompt
    assert "do not add a role/content envelope" in validated.prompt
    assert "PAC1 Antagonistの適応症は？" in validated.prompt
    assert "Kairyu L2 conversation (JSON)" not in validated.prompt


def test_batch_worker_rejects_template_and_legacy_policy_overlap(tmp_path):
    with pytest.raises(ValueError, match="both chat_templates and legacy_chat_models"):
        BatchWorker(
            BatchStore(tmp_path),
            {"m": MockBackend()},
            chat_templates={"m": ChatTemplate("ok")},
            legacy_chat_models={"m"},
        )


def test_public_render_prompt_rejects_template_and_legacy_policy_overlap():
    request = ChatCompletionRequest.model_validate(_chat_body())

    with pytest.raises(
        ChatRequestError,
        match="both chat_templates and legacy_chat_models",
    ):
        render_prompt(
            request,
            {"m": ChatTemplate("ok")},
            legacy_chat_models={"m"},
        )


async def test_checkpoint_authoritative_absent_special_token_keeps_hf_semantics():
    backend = MockBackend()
    template = ChatTemplate.from_tokenizer_metadata(
        "{% if bos_token is defined %}BOS{% else %}NO_BOS{% endif %}:"
        "{{ messages[0].content }}",
        {},
    )
    app = create_app({"m": backend}, chat_templates={"m": template})

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat_body())

    assert response.status_code == 200
    assert backend.prompts_seen == ("NO_BOS:hello",)
    assert isinstance(backend.prompts_seen[0], TemplatedPrompt)


async def test_checkpoint_special_token_cannot_be_injected_by_request_kwargs():
    backend = MockBackend()
    template = ChatTemplate.from_tokenizer_metadata(
        "{{ image_token }}{{ messages[0].content }}",
        {},
    )
    app = create_app({"m": backend}, chat_templates={"m": template})
    body = {
        **_chat_body(),
        "chat_template_kwargs": {"image_token": "<untrusted>"},
    }

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert "reserved template variables" in response.json()["error"]["message"]
    assert "image_token" in response.json()["error"]["message"]
    assert backend.prompts_seen == ()


async def test_strict_http_policy_requires_template_before_dispatch():
    backend = MockBackend()
    app = create_app({"m": backend})

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat_body())

    assert response.status_code == 400
    assert "chat_templates" in response.json()["error"]["message"]
    assert "legacy_chat_models" in response.json()["error"]["message"]
    assert backend.prompts_seen == ()


async def test_explicit_legacy_model_keeps_role_concat_under_strict_policy():
    backend = MockBackend()
    app = create_app(
        {"m": backend},
        legacy_chat_models={"m"},
    )

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat_body())

    assert response.status_code == 200
    assert backend.prompts_seen == (render_chat(_chat_body()["messages"]),)
    assert type(backend.prompts_seen[0]) is str


async def test_hf_template_marks_prompt_as_already_templated():
    backend = MockBackend()
    app = create_app(
        {"m": backend},
        chat_templates={
            "m": ChatTemplate("<BOS>{{ messages[0].content }}<ASSISTANT>")
        },
    )

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat_body())

    assert response.status_code == 200
    assert backend.prompts_seen == ("<BOS>hello<ASSISTANT>",)
    assert isinstance(backend.prompts_seen[0], TemplatedPrompt)


async def test_strict_policy_reaches_route_preview_and_responses():
    backend = MockBackend()
    app = create_app(
        {"m": backend},
        orchestrators={"auto": Orchestrator({"tier": MockBackend()})},
    )

    async with _client(app) as client:
        preview = await client.post(
            "/v1/route",
            json={"model": "auto", "messages": _chat_body()["messages"]},
        )
        response = await client.post(
            "/v1/responses",
            json={"model": "m", "input": "hello"},
        )

    for actual in (preview, response):
        assert actual.status_code == 400
        assert "chat_templates" in actual.json()["error"]["message"]
        assert "legacy_chat_models" in actual.json()["error"]["message"]
    assert backend.prompts_seen == ()


async def test_strict_policy_reaches_batch_worker(tmp_path):
    backend = MockBackend()
    worker = BatchWorker(
        BatchStore(tmp_path),
        {"m": backend},
    )
    line = {
        "custom_id": "strict",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": _chat_body(),
    }

    output, error, usage = await worker._run_line(
        json.loads(json.dumps(line)),
        "/v1/chat/completions",
        set(),
    )

    assert output is None
    assert usage is None
    assert "chat_templates" in error["error"]["message"]
    assert "legacy_chat_models" in error["error"]["message"]
    assert backend.prompts_seen == ()


@pytest.mark.parametrize(
    ("template", "body"),
    [
        (
            ChatTemplate(
                {
                    "default": "ok",
                    "tool_use": "{{ tools[0].function.name }}",
                }
            ),
            {**_chat_body(), "tools": []},
        ),
        (
            ChatTemplate("{{ messages[0].content + 1 }}"),
            _chat_body(),
        ),
    ],
)
async def test_runtime_template_error_is_controlled_for_http_and_batch(
    tmp_path,
    template,
    body,
):
    backend = MockBackend()
    app = create_app({"m": backend}, chat_templates={"m": template})

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert backend.prompts_seen == ()

    worker = BatchWorker(
        BatchStore(tmp_path),
        {"m": backend},
        chat_templates={"m": template},
    )
    line = {
        "custom_id": "invalid-template-input",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    output, error, usage = await worker._run_line(
        json.loads(json.dumps(line)),
        "/v1/chat/completions",
        set(),
    )

    assert output is None
    assert usage is None
    assert error["error"]["type"] == "invalid_request_error"
    assert backend.prompts_seen == ()


async def test_multimodal_chat_rejects_unforwarded_template_kwargs_predispatch():
    backend = MockBackend()
    app = create_app({"vlm": backend}, legacy_chat_models={"vlm"})
    body = {
        "model": "vlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/image.png"},
                    },
                ],
            }
        ],
        "chat_template_kwargs": {"enable_thinking": True},
    }

    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert "upstream-owned multimodal" in response.json()["error"]["message"]
    assert backend.prompts_seen == ()


def test_multimodal_chat_bypasses_strict_local_text_template_policy():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "vlm",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.test/image.png"},
                        },
                    ],
                }
            ],
        }
    )

    validated = validate_chat_input(
        request,
        None,
        allow_multimodal=True,
    )

    assert isinstance(validated.prompt, MultimodalPrompt)


def test_multimodal_orchestration_retains_images_and_a_text_routing_view():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "read this chart "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,opaque",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        }
    )

    validated = validate_orchestration_chat_input(request)

    assert isinstance(validated.prompt, str)
    assert "read this chart <image:0>" in validated.prompt
    multimodal = validated.orchestration_multimodal_prompt
    assert isinstance(multimodal, MultimodalPrompt)
    assert multimodal.items[0].data == "data:image/png;base64,opaque"
    assert multimodal.messages[0].content[1].detail == "high"


def test_text_preparation_avoids_message_dump_and_visits_parts_once(monkeypatch):
    request = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hel"},
                        {"type": "text", "text": "lo"},
                    ],
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"q":"tokyo"}',
                            },
                        }
                    ],
                },
            ],
        }
    )
    raw_parts = request.messages[0].content
    assert isinstance(raw_parts, list)
    expected_part_ids = [id(part) for part in raw_parts]
    dumped_part_ids: list[int] = []
    original_part_dump = ContentPart.model_dump

    def fail_message_dump(*_args, **_kwargs):
        raise AssertionError("ChatMessage.model_dump must not be used")

    def record_part_dump(self, *args, **kwargs):
        dumped_part_ids.append(id(self))
        return original_part_dump(self, *args, **kwargs)

    monkeypatch.setattr(ChatMessage, "model_dump", fail_message_dump)
    monkeypatch.setattr(ContentPart, "model_dump", record_part_dump)
    template = ChatTemplate(
        "{{ messages[0].content }}|"
        "{% if messages[1].content is defined %}CONTENT{% else %}NO_CONTENT{% endif %}|"
        "{{ messages[1].tool_calls[0].function.arguments | tojson }}"
    )

    validated = validate_chat_input(request, {"m": template})

    assert validated.prompt == 'hello|NO_CONTENT|{"q": "tokyo"}'
    assert isinstance(validated.prompt, TemplatedPrompt)
    assert dumped_part_ids == expected_part_ids


def test_multimodal_preparation_visits_parts_once_and_keeps_global_item_indices(
    monkeypatch,
):
    request = ChatCompletionRequest.model_validate(
        {
            "model": "vlm",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,first",
                                "detail": "low",
                            },
                        },
                        {"type": "text", "text": " then "},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "next "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,second",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
        }
    )
    raw_parts = [
        part
        for message in request.messages
        for part in (message.content if isinstance(message.content, list) else [])
    ]
    expected_part_ids = [id(part) for part in raw_parts]
    dumped_part_ids: list[int] = []
    original_part_dump = ContentPart.model_dump

    def fail_message_dump(*_args, **_kwargs):
        raise AssertionError("ChatMessage.model_dump must not be used")

    def record_part_dump(self, *args, **kwargs):
        dumped_part_ids.append(id(self))
        return original_part_dump(self, *args, **kwargs)

    monkeypatch.setattr(ChatMessage, "model_dump", fail_message_dump)
    monkeypatch.setattr(ContentPart, "model_dump", record_part_dump)

    validated = validate_chat_input(request, None, allow_multimodal=True)

    prompt = validated.prompt
    assert isinstance(prompt, MultimodalPrompt)
    assert [item.data for item in prompt.items] == [
        "data:image/png;base64,first",
        "data:image/png;base64,second",
    ]
    assert [
        (part.type, part.text, part.item_index, part.detail)
        for message in prompt.messages
        for part in message.content
    ] == [
        ("item", None, 0, "low"),
        ("text", " then ", None, None),
        ("text", "next ", None, None),
        ("item", None, 1, "high"),
    ]
    assert prompt.base.prompt == "user: <image:0> then \nuser: next <image:1>"
    assert dumped_part_ids == expected_part_ids


@pytest.mark.parametrize(
    ("system_content", "expected"),
    [
        (
            "Keep the original instruction.",
            "Keep the original instruction.\n\n"
            "Call at most one function in this response.",
        ),
        (
            [{"type": "text", "text": "Keep the original instruction."}],
            "Keep the original instruction."
            "Call at most one function in this response.",
        ),
        (None, "Call at most one function in this response."),
    ],
)
def test_single_tool_constraint_preserves_original_content_shape(
    system_content,
    expected,
):
    request = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "system", "content": system_content}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                }
            ],
            "parallel_tool_calls": False,
        }
    )

    prompt = render_prompt(
        request,
        {"m": ChatTemplate("{{ messages[0].content }}")},
    )

    assert prompt == expected


@pytest.mark.parametrize(
    ("template_kwargs", "templates", "expected"),
    [
        (
            {"enable_thinking": True},
            {"vlm": ChatTemplate("ok")},
            "upstream-owned multimodal",
        ),
        (None, {"vlm": ChatTemplate("ok")}, "cannot combine"),
        (None, None, "tool transcript fields"),
    ],
)
def test_multimodal_errors_keep_template_then_message_order(
    template_kwargs,
    templates,
    expected,
):
    request = ChatCompletionRequest.model_validate(
        {
            "model": "vlm",
            "messages": [
                {"role": "assistant", "content": [], "tool_calls": []},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,image"},
                        }
                    ],
                },
            ],
            "chat_template_kwargs": template_kwargs,
        }
    )

    with pytest.raises(ChatRequestError, match=expected):
        validate_chat_input(
            request,
            templates,
            allow_multimodal=True,
        )
