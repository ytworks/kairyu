"""HF Jinja chat templates: transformers parity + wire-form handling (m9 D2)."""

import json
from pathlib import Path

import pytest

from kairyu.entrypoints.chat_template import ChatTemplate, render_chat

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


@pytest.mark.parametrize("template_name", ["qwen-style.jinja", "llama3-style.jinja"])
def test_render_matches_transformers_no_tools(template_name):
    source = (TEMPLATES / template_name).read_text()
    ours = ChatTemplate(source, special_tokens={"bos_token": "<BOS>"}).render(MESSAGES)
    theirs = _transformers_render(source, MESSAGES, bos_token="<BOS>")
    assert ours == theirs


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


def test_legacy_render_chat_unchanged():
    out = render_chat([{"role": "user", "content": "hello"}])
    assert out == "user: hello\nassistant:"


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
