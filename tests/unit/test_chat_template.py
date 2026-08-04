"""HF Jinja chat templates: transformers parity + wire-form handling (m9 D2)."""

import json
from pathlib import Path

import pytest

from kairyu.entrypoints.chat_template import ChatTemplate, flatten_content, render_chat

TEMPLATES = Path(__file__).parent.parent / "fixtures" / "templates"

MESSAGES = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "こんにちは <tag> & 'quotes'"},
    {"role": "assistant", "content": "hi"},
    {"role": "user", "content": "call the tool"},
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "look things up <fast>",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]
TOOL_TRANSCRIPT = [
    {"role": "user", "content": "weather in tokyo?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                # OpenAI wire form: arguments is a JSON STRING
                "function": {"name": "lookup", "arguments": '{"q": "tokyo"}'},
            }
        ],
    },
    {"role": "tool", "content": '{"temp": 21}', "tool_call_id": "call_1"},
]


def _transformers_render(template_source, messages, tools=None, **special):
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models

    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.BPE(unk_token="[UNK]"))
    )
    tok.chat_template = template_source
    for name, value in special.items():
        setattr(tok, name, value)
    return tok.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
    )


def _save_tiny_tokenizer(
    directory: Path,
    template_source: str,
    *,
    family: str,
    template_location: str = "config",
) -> None:
    """Write a fully offline HF tokenizer with realistic chat special tokens."""
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    if family == "llama":
        specials = [
            "[UNK]",
            "<|begin_of_text|>",
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
        ]
        bos_token = "<|begin_of_text|>"
        eos_token = "<|eot_id|>"
        extra_special_tokens = ["<|start_header_id|>", "<|end_header_id|>"]
    elif family == "qwen":
        specials = ["[UNK]", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        bos_token = None
        eos_token = "<|im_end|>"
        extra_special_tokens = ["<|im_start|>"]
    else:  # pragma: no cover - helper contract
        raise AssertionError(f"unknown tokenizer family: {family}")

    vocab = {token: token_id for token_id, token in enumerate(specials)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    if bos_token is not None:
        # If apply_chat_template accidentally enabled tokenizer-added specials,
        # this would prepend a second BOS to the template-owned BOS.
        backend.post_processor = processors.TemplateProcessing(
            single=f"{bos_token} $A",
            special_tokens=[(bos_token, vocab[bos_token])],
        )

    kwargs = {
        "tokenizer_object": backend,
        "unk_token": "[UNK]",
        "eos_token": eos_token,
        "extra_special_tokens": extra_special_tokens,
        "chat_template": template_source,
    }
    if bos_token is not None:
        kwargs["bos_token"] = bos_token
    else:
        kwargs["pad_token"] = "<|endoftext|>"
    tokenizer = transformers.PreTrainedTokenizerFast(**kwargs)
    directory.mkdir()
    tokenizer.save_pretrained(directory)

    config_path = directory / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if bos_token is not None:
        # Real tokenizer configs use both string and AddedToken object forms.
        config["bos_token"] = {
            "__type": "AddedToken",
            "content": bos_token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
    if template_location == "config":
        (directory / "chat_template.jinja").unlink()
        config["chat_template"] = template_source
    elif template_location != "dedicated":  # pragma: no cover - helper contract
        raise AssertionError(f"unknown template location: {template_location}")
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _assert_auto_loaded_parity(
    directory: Path,
    messages,
    *,
    tools=None,
):
    transformers = pytest.importorskip("transformers")

    from kairyu.engine.tokenizer import HFTokenizer, load_tokenizer_chat_metadata

    reference = transformers.AutoTokenizer.from_pretrained(
        directory, local_files_only=True
    )
    metadata = load_tokenizer_chat_metadata(directory)
    ours = ChatTemplate(
        metadata.templates,
        special_tokens=metadata.special_tokens,
    ).render(messages, tools=tools)
    theirs = reference.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    assert ours.encode("utf-8") == theirs.encode("utf-8")

    ours_ids = HFTokenizer(directory).encode(ours)
    theirs_ids = reference.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    assert ours_ids == tuple(theirs_ids)
    return metadata, reference, ours, ours_ids


@pytest.mark.parametrize("template_name", ["qwen-style.jinja", "llama3-style.jinja"])
def test_render_matches_transformers_no_tools(template_name):
    source = (TEMPLATES / template_name).read_text()
    ours = ChatTemplate(source, special_tokens={"bos_token": "<BOS>"}).render(MESSAGES)
    theirs = _transformers_render(source, MESSAGES, bos_token="<BOS>")
    assert ours == theirs


@pytest.mark.parametrize(
    ("family", "template_name"),
    [("llama", "llama3-style.jinja"), ("qwen", "qwen-style.jinja")],
)
def test_tokenizer_config_template_bytes_and_ids_match_transformers(
    tmp_path,
    family,
    template_name,
):
    source = (TEMPLATES / template_name).read_text()
    tokenizer_dir = tmp_path / family
    _save_tiny_tokenizer(tokenizer_dir, source, family=family)

    _metadata, reference, _rendered, token_ids = _assert_auto_loaded_parity(
        tokenizer_dir,
        MESSAGES,
    )
    if family == "llama":
        assert reference.bos_token_id is not None
        assert token_ids.count(reference.bos_token_id) == 1


def test_dedicated_template_precedes_tokenizer_config_for_bytes_and_ids(tmp_path):
    dedicated = (TEMPLATES / "llama3-style.jinja").read_text()
    tokenizer_dir = tmp_path / "llama"
    _save_tiny_tokenizer(
        tokenizer_dir,
        dedicated,
        family="llama",
        template_location="dedicated",
    )
    config_path = tokenizer_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = "CONFIG_FALLBACK:{{ messages[0].content }}"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata, reference, rendered, _token_ids = _assert_auto_loaded_parity(
        tokenizer_dir,
        MESSAGES,
    )
    assert metadata.templates == {"default": dedicated}
    assert metadata.template_sources == (str(tokenizer_dir / "chat_template.jinja"),)
    assert reference.chat_template == dedicated
    assert not rendered.startswith("CONFIG_FALLBACK:")


def test_named_tokenizer_templates_match_transformers_selection(tmp_path):
    default = "DEFAULT:{{ bos_token }}{{ messages[0].content }}"
    tool_use = (
        "TOOL:{{ bos_token }}"
        "{% if tools %}{{ tools[0].function.name }}{% else %}EMPTY{% endif %}"
    )
    tokenizer_dir = tmp_path / "llama"
    _save_tiny_tokenizer(tokenizer_dir, default, family="llama")
    config_path = tokenizer_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = [
        {"name": "default", "template": default},
        {"name": "tool_use", "template": tool_use},
    ]
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    messages = [{"role": "user", "content": "hello"}]

    _assert_auto_loaded_parity(tokenizer_dir, messages)
    _assert_auto_loaded_parity(tokenizer_dir, messages, tools=[])
    _assert_auto_loaded_parity(tokenizer_dir, messages, tools=TOOLS)


def test_named_dedicated_templates_match_transformers_5_12_layout(tmp_path):
    transformers = pytest.importorskip("transformers")
    default = "DEFAULT:{{ messages[0].content }}"
    tool_use = "TOOL:{% if tools %}SET{% else %}EMPTY{% endif %}"
    config_dir = tmp_path / "config-layout"
    _save_tiny_tokenizer(config_dir, default, family="qwen")
    config_path = config_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["chat_template"] = [
        {"name": "default", "template": default},
        {"name": "tool_use", "template": tool_use},
    ]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    reference = transformers.AutoTokenizer.from_pretrained(
        config_dir,
        local_files_only=True,
    )
    dedicated_dir = tmp_path / "dedicated-layout"
    reference.save_pretrained(dedicated_dir)
    assert (dedicated_dir / "chat_template.jinja").is_file()
    assert (
        dedicated_dir / "additional_chat_templates" / "tool_use.jinja"
    ).is_file()

    messages = [{"role": "user", "content": "hello"}]
    _assert_auto_loaded_parity(dedicated_dir, messages)
    _assert_auto_loaded_parity(dedicated_dir, messages, tools=[])
    _assert_auto_loaded_parity(dedicated_dir, messages, tools=TOOLS)


def test_explicit_template_keeps_tokenizer_special_token_context(tmp_path):
    source = (TEMPLATES / "llama3-style.jinja").read_text()
    tokenizer_dir = tmp_path / "llama"
    _save_tiny_tokenizer(tokenizer_dir, source, family="llama")

    from kairyu.engine.tokenizer import HFTokenizer, load_tokenizer_chat_metadata

    metadata = load_tokenizer_chat_metadata(tokenizer_dir)
    rendered = ChatTemplate(
        "{{ bos_token }}OVERRIDE:{{ messages[0].content }}",
        special_tokens=metadata.special_tokens,
    ).render([{"role": "user", "content": "hello"}])
    assert rendered == "<|begin_of_text|>OVERRIDE:hello"
    reference = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True
    )
    assert HFTokenizer(tokenizer_dir).encode(rendered) == tuple(
        reference(rendered, add_special_tokens=False)["input_ids"]
    )


def test_render_matches_transformers_with_tools():
    source = (TEMPLATES / "qwen-style.jinja").read_text()
    ours = ChatTemplate(source).render(MESSAGES, tools=TOOLS)
    theirs = _transformers_render(source, MESSAGES, tools=TOOLS)
    assert ours == theirs
    assert "<fast>" in ours  # HF tojson: no html escaping of < > &


def test_tool_call_arguments_string_is_parsed_to_dict():
    source = (TEMPLATES / "qwen-style.jinja").read_text()
    ours = ChatTemplate(source).render(TOOL_TRANSCRIPT)
    # arguments rendered as a JSON object, not a double-encoded string
    assert '"arguments": {"q": "tokyo"}' in ours
    # parity with transformers given the already-dict form
    dict_form = json.loads(json.dumps(TOOL_TRANSCRIPT))
    dict_form[1]["tool_calls"][0]["function"]["arguments"] = {"q": "tokyo"}
    theirs = _transformers_render(source, dict_form)
    assert ours == theirs


def test_tools_none_not_empty_list():
    source = "{% if tools is not none %}HAS_TOOLS{% else %}NO_TOOLS{% endif %}"
    assert ChatTemplate(source).render([{"role": "user", "content": "x"}]) == "NO_TOOLS"


def test_named_templates_select_default_or_tool_use():
    template = ChatTemplate(
        {
            "default": "DEFAULT:{{ messages[0].content }}",
            "tool_use": (
                "TOOLS:{% if tools %}{{ tools[0].function.name }}"
                "{% else %}EMPTY{% endif %}"
            ),
        }
    )
    messages = [{"role": "user", "content": "hello"}]

    assert template.render(messages) == "DEFAULT:hello"
    assert template.render(messages, tools=[]) == "TOOLS:EMPTY"
    assert template.render(messages, tools=TOOLS) == "TOOLS:lookup"


def test_generation_tag_renders_with_transformers_parity():
    source = (
        "{{ messages[0].content }}|"
        "{% generation %}{{ messages[1].content }}{% endgeneration %}"
    )
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    assert ChatTemplate(source).render(messages) == _transformers_render(
        source,
        messages,
    )


def test_omitted_documents_is_none_with_transformers_parity():
    source = "{% if documents is none %}NONE{% else %}OTHER{% endif %}"
    messages = [{"role": "user", "content": "question"}]

    assert ChatTemplate(source).render(messages) == _transformers_render(
        source,
        messages,
    )


def test_absent_special_tokens_remain_undefined_with_transformers_parity():
    source = (
        "{% if bos_token is defined %}BOS{% else %}NO_BOS{% endif %}|"
        "{% if eos_token is defined %}EOS{% else %}NO_EOS{% endif %}"
    )
    messages = [{"role": "user", "content": "question"}]

    assert ChatTemplate(source).render(messages) == _transformers_render(
        source,
        messages,
    )


def test_template_reports_unresolved_tokenizer_owned_variables():
    template = ChatTemplate(
        "{{ bos_token }}{{ image_token }}{{ enable_thinking }}",
        special_tokens={"bos_token": "<BOS>"},
    )

    assert template.referenced_special_token_variables == frozenset(
        {"bos_token", "image_token"}
    )
    assert template.unresolved_special_token_variables == frozenset(
        {"image_token"}
    )


def test_http_message_dump_preserves_omitted_keys_with_transformers_parity():
    from kairyu.engine.prompt import TemplatedPrompt
    from kairyu.entrypoints.server.chat_service import render_prompt
    from kairyu.entrypoints.server.protocol import ChatCompletionRequest

    source = (
        "{% if messages[0].name is defined %}NAME{% else %}NO_NAME{% endif %}|"
        "{% if messages[1].content is defined %}CONTENT{% else %}NO_CONTENT{% endif %}"
    )
    wire_messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {}},
                }
            ],
        },
    ]
    request = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": wire_messages}
    )

    ours = render_prompt(request, {"m": ChatTemplate(source)})
    theirs = _transformers_render(source, wire_messages)

    assert ours == theirs == "NO_NAME|NO_CONTENT"
    assert isinstance(ours, TemplatedPrompt)


def test_named_templates_without_default_fail_for_non_tool_request():
    template = ChatTemplate({"tool_use": "TOOLS"})

    with pytest.raises(ValueError, match="no 'default'.*available templates: tool_use"):
        template.render([{"role": "user", "content": "hello"}])

    assert template.render(
        [{"role": "user", "content": "hello"}], tools=TOOLS
    ) == "TOOLS"


@pytest.mark.parametrize(
    "source",
    [
        "",
        "   ",
        {"default": ""},
        {"default": "\n\t"},
        {"": "ok"},
        {"   ": "ok"},
    ],
)
def test_empty_template_sources_are_rejected(source):
    with pytest.raises(ValueError, match="non-empty"):
        ChatTemplate(source)


def test_request_template_kwargs_reach_the_jinja_template():
    source = "{% if enable_thinking %}THINK{% else %}ANSWER{% endif %}"
    template = ChatTemplate(source)
    messages = [{"role": "user", "content": "x"}]

    assert template.render(
        messages, template_kwargs={"enable_thinking": True}
    ) == "THINK"
    assert template.render(
        messages, template_kwargs={"enable_thinking": False}
    ) == "ANSWER"


@pytest.mark.parametrize(
    "name",
    [
        "messages",
        "tools",
        "documents",
        "add_generation_prompt",
        "chat_template",
        "continue_final_message",
        "return_assistant_tokens_mask",
        "conversations",
    ],
)
def test_special_tokens_cannot_replace_hf_template_control_variables(name):
    with pytest.raises(ValueError, match="control variables"):
        ChatTemplate("ok", special_tokens={name: "malicious"})


@pytest.mark.parametrize(
    "special_tokens",
    [
        {1: "token"},
        {"image_token": 1},
    ],
)
def test_special_token_context_requires_string_names_and_values(special_tokens):
    with pytest.raises(TypeError, match="special-token"):
        ChatTemplate("ok", special_tokens=special_tokens)


@pytest.mark.parametrize(
    "reserved",
    [
        "messages",
        "tools",
        "documents",
        "add_generation_prompt",
        "chat_template",
        "continue_final_message",
        "return_assistant_tokens_mask",
        "conversations",
        "bos_token",
    ],
)
def test_request_template_kwargs_cannot_replace_trusted_variables(reserved):
    with pytest.raises(ValueError, match="reserved template variables"):
        ChatTemplate("ok").render(
            [{"role": "user", "content": "x"}],
            template_kwargs={reserved: "untrusted"},
        )


def test_request_template_kwargs_cannot_inject_checkpoint_special_token():
    template = ChatTemplate.from_tokenizer_metadata(
        "{{ image_token }}{{ messages[0].content }}",
        {},
    )

    with pytest.raises(ValueError, match="reserved template variables.*image_token"):
        template.render(
            [{"role": "user", "content": "x"}],
            template_kwargs={"image_token": "<untrusted>"},
        )


def test_unknown_role_raises_via_template():
    source = (TEMPLATES / "qwen-style.jinja").read_text()
    with pytest.raises(ValueError, match="unknown role"):
        ChatTemplate(source).render([{"role": "alien", "content": "x"}])


def test_load_from_path_and_inline(tmp_path):
    path = tmp_path / "t.jinja"
    path.write_text("X{{ messages[0].content }}")
    assert ChatTemplate.load(str(path)).render([{"role": "user", "content": "y"}]) == "Xy"
    inline = ChatTemplate.load("A{{ messages[0].content }}")
    assert inline.render([{"role": "user", "content": "b"}]) == "Ab"
    with pytest.raises(ValueError, match="not found"):
        ChatTemplate.load(str(tmp_path / "missing.jinja"))


def test_load_named_templates_from_paths_and_inline(tmp_path):
    path = tmp_path / "default.jinja"
    path.write_text("DEFAULT")
    template = ChatTemplate.load(
        {"default": str(path), "tool_use": "TOOL:{{ tools[0].function.name }}"}
    )

    assert template.render([{"role": "user", "content": "x"}]) == "DEFAULT"
    assert template.render(
        [{"role": "user", "content": "x"}], tools=TOOLS
    ) == "TOOL:lookup"


def test_legacy_render_chat_unchanged():
    out = render_chat([{"role": "user", "content": "hello"}])
    assert out == "user: hello\nassistant:"


@pytest.mark.parametrize(
    "renderer",
    [
        lambda messages: render_chat(messages),
        lambda messages: ChatTemplate("{{ messages[0].content }}").render(messages),
    ],
)
def test_text_renderers_reject_images_instead_of_dropping_them(renderer):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/image"},
                },
            ],
        }
    ]

    with pytest.raises(ValueError, match="cannot execute"):
        renderer(messages)


@pytest.mark.parametrize(
    "renderer",
    [
        lambda messages: render_chat(messages),
        lambda messages: ChatTemplate("{{ messages[0].content }}").render(messages),
    ],
)
def test_text_renderers_reject_alternate_prompt_carriers(renderer):
    messages = [
        {
            "role": "user",
            "content": "visible text",
            "prompt_token_ids": [1, 2, 3],
        }
    ]

    with pytest.raises(ValueError, match="alternate prompt carriers"):
        renderer(messages)


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_audio", "input_audio": {"data": "AA=="}},
        {"type": "text"},
        {"type": "text", "text": "x", "unknown": True},
        {"type": "image_url", "image_url": {}},
    ],
)
def test_flatten_content_never_drops_unknown_or_malformed_parts(part):
    with pytest.raises(ValueError, match="content part 0"):
        flatten_content([part])


def test_ascii_preserved_in_unicode_content():
    source = "{{ messages | tojson }}"
    out = ChatTemplate(source).render([{"role": "user", "content": "日本語"}])
    assert "日本語" in out  # ensure_ascii=False


async def test_per_model_template_reaches_the_engine(tmp_path):
    """The HTTP path renders with the per-model template from the spec (m9 D2);
    the batch worker shares the exact same render_prompt + template map."""
    import httpx

    from kairyu.engine.mock import MockBackend
    from kairyu.entrypoints.server.app import create_app

    template = tmp_path / "chatml.jinja"
    template.write_text(
        "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    templated, plain = MockBackend(), MockBackend()
    app = create_app(
        engines={"templated": templated, "plain": plain},
        chat_templates={"templated": ChatTemplate.load(str(template))},
        legacy_chat_models={"plain"},
    )
    body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        await client.post("/v1/chat/completions", json={"model": "templated", **body})
        await client.post("/v1/chat/completions", json={"model": "plain", **body})
    assert templated.prompts_seen == ("<|user|>hello<|end|><|assistant|>",)
    assert plain.prompts_seen == (render_chat(body["messages"]),)


def test_builder_threads_templates_from_spec(tmp_path):
    from kairyu.deploy.builder import build_app_from_spec
    from kairyu.deploy.spec import load_deployment_spec

    template = tmp_path / "t.jinja"
    template.write_text("T:{{ messages[0].content }}")
    spec = load_deployment_spec(
        f"""
engines:
  m: {{ backend: mock }}
chat_templates:
  m: {str(template)}
"""
    )
    app = build_app_from_spec(spec)
    assert app.state.deployment_spec.chat_templates == {"m": str(template)}


def test_spec_rejects_template_for_unknown_model():
    import pytest as _pytest

    from kairyu.deploy.spec import load_deployment_spec

    with _pytest.raises(ValueError, match="unknown models"):
        load_deployment_spec(
            """
engines:
  a: { backend: mock }
chat_templates:
  ghost: "{{ messages }}"
"""
        )
