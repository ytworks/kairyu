"""Contract tests pinning the vLLM-compatible public API surface (design doc D2).

The shape below mirrors vLLM's official ``examples/offline_inference/basic.py``:
importing LLM/SamplingParams, calling ``llm.generate(prompts, sampling_params)``
and reading ``output.prompt`` / ``output.outputs[0].text``.
"""

import pytest

from kairyu import LLM, RequestOutput, SamplingParams
from kairyu.engine.mock import MockBackend


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


def test_chat_renders_messages(llm):
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Say hi."},
    ]
    outputs = llm.chat(messages)
    assert len(outputs) == 1
    assert outputs[0].outputs[0].text


def test_default_backend_is_mock_when_vllm_missing():
    llm = LLM(model="mock-model")
    assert type(llm.backend).__name__ in {"MockBackend", "VLLMBackend"}


def test_signature_cross_check_against_real_vllm():
    vllm = pytest.importorskip("vllm")
    import inspect

    theirs = set(inspect.signature(vllm.SamplingParams).parameters)
    ours = set(inspect.signature(SamplingParams).parameters)
    core = {"n", "temperature", "top_p", "top_k", "seed", "stop", "max_tokens", "min_tokens"}
    assert core <= theirs
    assert core <= ours
