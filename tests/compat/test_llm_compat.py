"""Contract tests pinning the vLLM-compatible public API surface (design doc D2).

The shape below mirrors vLLM's official ``examples/offline_inference/basic.py``:
importing LLM/SamplingParams, calling ``llm.generate(prompts, sampling_params)``
and reading ``output.prompt`` / ``output.outputs[0].text``.
"""

import json

import pytest

from kairyu import (
    LLM,
    RequestOutput,
    SamplingParams,
    TemplatedPrompt,
    TextPrompt,
    TokensPrompt,
)
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.chat_template import ChatTemplate, render_chat
from kairyu.sampling_params import GENERATION_CONFIG_SAMPLING_FIELDS


@pytest.fixture()
def llm() -> LLM:
    return LLM(model="mock-model", backend=MockBackend())


def test_vllm_basic_example_shape(llm):
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)
    for prompt, output in zip(prompts, outputs, strict=True):
        assert isinstance(output, RequestOutput)
        assert output.prompt == prompt
        assert isinstance(output.outputs[0].text, str)
        assert output.outputs[0].text


def test_generate_accepts_single_string(llm):
    outputs = llm.generate("just one prompt")
    assert len(outputs) == 1
    assert outputs[0].prompt == "just one prompt"


def test_omitted_offline_params_use_model_defaults_but_supplied_params_are_explicit():
    class RecordingBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.params = []

        async def generate(self, request):
            self.params.append(request.sampling_params)
            return await super().generate(request)

    backend = RecordingBackend()
    local_llm = LLM(model="mock-model", backend=backend)

    local_llm.generate("omitted")
    local_llm.generate("explicit", SamplingParams())

    assert backend.params[0].generation_config_omitted == (
        GENERATION_CONFIG_SAMPLING_FIELDS
    )
    assert backend.params[1].generation_config_omitted == frozenset()


def test_generate_accepts_one_token_prompt_without_treating_it_as_a_batch(llm):
    outputs = llm.generate(TokensPrompt((3, 5, 8)))

    assert len(outputs) == 1
    assert outputs[0].prompt is None
    assert outputs[0].prompt_token_ids == (3, 5, 8)


def test_generate_rejects_a_mapping_instead_of_dispatching_its_keys_as_prompts():
    backend = MockBackend()
    local_llm = LLM(model="mock-model", backend=backend)

    with pytest.raises(TypeError, match="PromptInput or a sequence"):
        local_llm.generate(
            {
                "prompt": "hello",
                "multi_modal_data": {"image": "opaque"},
            }
        )

    assert backend.prompts_seen == ()


def test_generate_rejects_an_invalid_batch_item_before_any_dispatch():
    backend = MockBackend()
    local_llm = LLM(model="mock-model", backend=backend)

    with pytest.raises(TypeError, match=r"prompts\[1\].*supported PromptInput"):
        local_llm.generate(["valid", {"prompt_token_ids": [1, 2, 3]}])

    assert backend.prompts_seen == ()


def test_generation_request_rechecks_prompt_authority_after_low_level_mutation():
    backend = MockBackend()
    local_llm = LLM(model="mock-model", backend=backend)
    params = SamplingParams()
    object.__setattr__(params, "extra_args", {"prompt_token_ids": [1, 2, 3]})

    with pytest.raises(ValueError, match="prompt-owned fields"):
        local_llm.generate("text authority", params)

    assert backend.prompts_seen == ()


def test_batch_prepares_every_request_in_input_order_before_any_dispatch():
    class PreparingBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.events = []

        def validate_request(self, request):
            self.events.append(("validate", request.prompt))
            super().validate_request(request)

        async def prepare_request(self, request):
            self.events.append(("prepare", request.prompt))

        async def generate(self, request):
            self.events.append(("generate", request.prompt))
            return await super().generate(request)

    backend = PreparingBackend()
    local_llm = LLM(model="mock-model", backend=backend)

    outputs = local_llm.generate(["first", "second", "third"])

    assert [output.prompt for output in outputs] == ["first", "second", "third"]
    assert backend.events[:6] == [
        ("validate", "first"),
        ("validate", "second"),
        ("validate", "third"),
        ("prepare", "first"),
        ("prepare", "second"),
        ("prepare", "third"),
    ]
    assert [event for event in backend.events if event[0] == "generate"] == [
        ("generate", "first"),
        ("generate", "second"),
        ("generate", "third"),
    ]


def test_batch_prepare_failure_stops_in_order_before_any_dispatch():
    failure = ValueError("second prepare failed")

    class RejectingPrepareBackend(MockBackend):
        def __init__(self):
            super().__init__()
            self.prepared = []

        async def prepare_request(self, request):
            self.prepared.append(request.prompt)
            if request.prompt == "second":
                raise failure

    backend = RejectingPrepareBackend()
    local_llm = LLM(model="mock-model", backend=backend)

    with pytest.raises(ValueError, match="second prepare failed") as raised:
        local_llm.generate(["first", "second", "third"])

    assert raised.value is failure
    assert backend.prepared == ["first", "second"]
    assert backend.prompts_seen == ()


@pytest.mark.parametrize(
    "prompt",
    [
        TextPrompt("typed"),
        TemplatedPrompt("templated"),
        TokensPrompt((1, 2, 3)),
    ],
)
def test_typed_prompt_is_not_dispatched_to_an_undeclared_legacy_backend(prompt):
    class LegacyBackend:
        calls = 0

        async def generate(self, _request):
            self.calls += 1
            raise AssertionError("typed prompt was dispatched")

    backend = LegacyBackend()
    legacy = LLM(model="legacy", backend=backend)

    with pytest.raises(ValueError, match="legacy-string text-only"):
        legacy.generate(prompt)

    assert backend.calls == 0


def test_generate_accepts_per_prompt_params(llm):
    params = [SamplingParams(n=1), SamplingParams(n=2)]
    outputs = llm.generate(["a", "b"], params)
    assert len(outputs[0].outputs) == 1
    assert len(outputs[1].outputs) == 2


def test_generate_params_length_mismatch_rejected(llm):
    with pytest.raises(ValueError, match="sampling_params"):
        llm.generate(["a", "b"], [SamplingParams()])


def test_vllm_style_constructor_kwargs_accepted():
    llm = LLM(
        model="mock-model",
        tokenizer=None,
        tensor_parallel_size=1,
        dtype="auto",
        seed=0,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        trust_remote_code=False,
        max_model_len=4096,  # unknown kwargs stored, never fatal
        backend=MockBackend(),
    )
    assert llm.model == "mock-model"
    assert llm.engine_kwargs["max_model_len"] == 4096


def test_chat_auto_loads_local_tokenizer_template_and_special_tokens(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": (
                    "{{ bos_token }}{% for message in messages %}"
                    "{{ message.role }}={{ message.content }}|{% endfor %}"
                    "{% if add_generation_prompt %}assistant={% endif %}"
                ),
                "bos_token": "<BOS>",
            }
        ),
        encoding="utf-8",
    )
    backend = MockBackend()
    local_llm = LLM(model=str(tmp_path), backend=backend)
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Say hi."},
    ]
    outputs = local_llm.chat(messages)

    assert len(outputs) == 1
    assert outputs[0].outputs[0].text
    assert backend.prompts_seen == (
        "<BOS>system=You are terse.|user=Say hi.|assistant=",
    )
    assert isinstance(backend.prompts_seen[0], TemplatedPrompt)


def test_chat_missing_template_fails_before_dispatch():
    backend = MockBackend()
    local_llm = LLM(model="hub/model", backend=backend)

    with pytest.raises(ValueError, match="local tokenizer snapshot.*chat_template=.*legacy_chat"):
        local_llm.chat([{"role": "user", "content": "hello"}])

    assert backend.prompts_seen == ()


def test_explicit_chat_template_cannot_silently_drop_unavailable_bos():
    backend = MockBackend()
    local_llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        backend=backend,
        chat_template="{{ bos_token }}USER={{ messages[0].content }}",
    )

    with pytest.raises(
        ValueError,
        match="special-token variables.*bos_token.*not materialized locally",
    ):
        local_llm.chat([{"role": "user", "content": "hello"}])

    assert backend.prompts_seen == ()


def test_self_contained_explicit_chat_template_needs_no_local_metadata():
    backend = MockBackend()
    local_llm = LLM(
        model="hub/model",
        backend=backend,
        chat_template="<BOS>USER={{ messages[0].content }}",
    )

    local_llm.chat([{"role": "user", "content": "hello"}])

    assert backend.prompts_seen == ("<BOS>USER=hello",)
    assert isinstance(backend.prompts_seen[0], TemplatedPrompt)


def test_explicit_template_rejects_missing_local_special_token_value(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    backend = MockBackend()
    local_llm = LLM(
        model=str(tmp_path),
        backend=backend,
        chat_template="{{ bos_token }}USER={{ messages[0].content }}",
    )

    with pytest.raises(
        ValueError,
        match="special-token variables.*bos_token.*does not define",
    ):
        local_llm.chat([{"role": "user", "content": "hello"}])

    assert backend.prompts_seen == ()


def test_constructor_chat_template_object_cannot_bypass_special_token_guard():
    backend = MockBackend()
    local_llm = LLM(
        model="hub/model",
        backend=backend,
        chat_template=ChatTemplate(
            "{{ bos_token }}USER={{ messages[0].content }}"
        ),
    )

    with pytest.raises(
        ValueError,
        match="ChatTemplate references.*bos_token.*without supplying",
    ):
        local_llm.chat([{"role": "user", "content": "hello"}])

    assert backend.prompts_seen == ()


def test_chat_explicit_legacy_is_the_only_role_concat_path():
    backend = MockBackend()
    with pytest.warns(RuntimeWarning, match="role-concatenation"):
        local_llm = LLM(model="hub/model", backend=backend, legacy_chat=True)
    messages = [{"role": "user", "content": "hello"}]

    local_llm.chat(messages)

    assert backend.prompts_seen == (render_chat(messages),)
    assert type(backend.prompts_seen[0]) is str


def test_per_call_template_overrides_constructor_template(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<BOS>"}),
        encoding="utf-8",
    )
    backend = MockBackend()
    local_llm = LLM(
        model=str(tmp_path),
        backend=backend,
        chat_template="constructor:{{ bos_token }}{{ messages[0].content }}",
    )

    local_llm.chat([{"role": "user", "content": "one"}])
    local_llm.chat(
        [{"role": "user", "content": "two"}],
        chat_template="call:{{ bos_token }}{{ messages[0].content }}",
    )

    assert backend.prompts_seen == (
        "constructor:<BOS>one",
        "call:<BOS>two",
    )
    assert all(isinstance(prompt, TemplatedPrompt) for prompt in backend.prompts_seen)


def test_invalid_auto_chat_metadata_does_not_break_plain_generate(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": {"default": 7}}),
        encoding="utf-8",
    )
    backend = MockBackend()
    local_llm = LLM(model=str(tmp_path), backend=backend)

    outputs = local_llm.generate("completion remains valid")

    assert outputs[0].prompt == "completion remains valid"
    with pytest.raises(ValueError, match="chat metadata"):
        local_llm.chat([{"role": "user", "content": "hello"}])
    assert backend.prompts_seen == ("completion remains valid",)


@pytest.mark.parametrize(
    "carrier",
    [
        {"prompt_token_ids": [1, 2, 3]},
        {"input_audio": {"data": "AA=="}},
        {"multi_modal_data": {"image": "opaque"}},
    ],
)
def test_chat_rejects_nested_alternate_prompt_carriers_before_dispatch(carrier):
    backend = MockBackend()
    with pytest.warns(RuntimeWarning):
        local_llm = LLM(model="mock-model", backend=backend, legacy_chat=True)
    message = {"role": "user", "content": "hello", **carrier}

    with pytest.raises(ValueError, match="alternate prompt carriers"):
        local_llm.chat([message])

    assert backend.prompts_seen == ()


def test_chat_rejects_image_parts_instead_of_flattening_them_to_text():
    backend = MockBackend()
    with pytest.warns(RuntimeWarning):
        local_llm = LLM(model="mock-model", backend=backend, legacy_chat=True)

    with pytest.raises(ValueError, match="cannot execute"):
        local_llm.chat(
            [
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
        )

    assert backend.prompts_seen == ()


def test_default_backend_is_mock_when_vllm_missing():
    llm = LLM(model="mock-model")
    assert type(llm.backend).__name__ in {"MockBackend", "VLLMBackend"}


@pytest.mark.vllm
def test_signature_cross_check_against_real_vllm():
    import inspect

    import vllm

    theirs = set(inspect.signature(vllm.SamplingParams).parameters)
    ours = set(inspect.signature(SamplingParams).parameters)
    core = {"n", "temperature", "top_p", "top_k", "seed", "stop", "max_tokens", "min_tokens"}
    assert core <= theirs
    assert core <= ours

    constructor_core = {"model", "tokenizer", "chat_template"}
    assert constructor_core <= set(inspect.signature(vllm.LLM).parameters)
    assert constructor_core <= set(inspect.signature(LLM).parameters)

    chat_core = {
        "messages",
        "sampling_params",
        "chat_template",
        "add_generation_prompt",
        "tools",
        "chat_template_kwargs",
    }
    assert chat_core <= set(inspect.signature(vllm.LLM.chat).parameters)
    assert chat_core <= set(inspect.signature(LLM.chat).parameters)
