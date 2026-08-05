import importlib.util
import sys
import types

import pytest

from kairyu import SamplingParams
from kairyu.engine import vllm_backend
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.prompt import (
    MultimodalItem,
    MultimodalPrompt,
    TemplatedPrompt,
    TextPrompt,
    TokensPrompt,
)
from kairyu.engine.vllm_backend import VLLMBackend, to_vllm_sampling_kwargs
from kairyu.sampling_params import GENERATION_CONFIG_SAMPLING_FIELDS


def test_module_imports_without_vllm():
    assert vllm_backend is not None  # import at top of file already proves this


def test_param_mapping_is_pure_and_complete():
    params = SamplingParams(
        n=2,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        seed=7,
        stop=("END",),
        max_tokens=128,
        min_tokens=2,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        repetition_penalty=1.1,
        ignore_eos=True,
    )
    kwargs = to_vllm_sampling_kwargs(params)
    assert kwargs == {
        "n": 2,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.0,
        "seed": 7,
        "stop": ["END"],
        "stop_token_ids": [],
        "max_tokens": 128,
        "min_tokens": 2,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "repetition_penalty": 1.1,
        "ignore_eos": True,
        "skip_special_tokens": True,
    }


def _install_fake_vllm(monkeypatch) -> dict:
    """Fake vllm module capturing AsyncEngineArgs kwargs (m5 D3 plumbing tests)."""
    captured: dict = {}

    class _FakeArgs:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class _FakeEngine:
        @staticmethod
        def from_engine_args(args) -> object:
            return object()

    fake_vllm = types.SimpleNamespace(
        AsyncEngineArgs=_FakeArgs, AsyncLLMEngine=_FakeEngine, SamplingParams=dict
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    return captured


class _CapturingEngine:
    def __init__(self, default_sampling: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.model_config = types.SimpleNamespace(
            get_diff_sampling_param=lambda: dict(default_sampling or {})
        )

    async def generate(self, prompt, params, request_id, *, priority, **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "params": params,
                "request_id": request_id,
                "priority": priority,
                **kwargs,
            }
        )
        yield types.SimpleNamespace(
            outputs=[
                types.SimpleNamespace(
                    index=0,
                    text="done",
                    token_ids=[7],
                    cumulative_logprob=0.0,
                    finish_reason="length",
                    stop_reason=None,
                )
            ],
            finished=True,
        )


def _capturing_backend(monkeypatch) -> tuple[VLLMBackend, _CapturingEngine]:
    _install_fake_vllm(monkeypatch)
    backend = VLLMBackend(model="m")
    engine = _CapturingEngine()
    backend._engine = engine
    return backend, engine


def test_tensor_parallel_size_reaches_vllm_engine_args(monkeypatch):
    captured = _install_fake_vllm(monkeypatch)
    VLLMBackend(model="m", tensor_parallel_size=4)
    assert captured["model"] == "m"
    assert captured["tensor_parallel_size"] == 4
    assert captured["scheduling_policy"] == "priority"


def test_vllm_backend_rejects_fcfs_that_would_ignore_priority(monkeypatch):
    _install_fake_vllm(monkeypatch)
    with pytest.raises(ValueError, match="requires scheduling_policy='priority'"):
        VLLMBackend(model="m", scheduling_policy="fcfs")


def test_llm_default_backend_forwards_tensor_parallel_size_to_vllm(monkeypatch):
    from kairyu.entrypoints import llm as llm_module

    captured = _install_fake_vllm(monkeypatch)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "vllm" else None
    )
    backend = llm_module._default_backend(
        "m",
        None,
        tensor_parallel_size=8,
        tokenizer="tokenizer-override",
    )
    assert isinstance(backend, VLLMBackend)
    assert captured["tensor_parallel_size"] == 8
    assert captured["tokenizer"] == "tokenizer-override"


def test_async_engine_default_backend_forwards_tensor_parallel_size_to_vllm(monkeypatch):
    from kairyu.entrypoints import async_engine

    captured = _install_fake_vllm(monkeypatch)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "vllm" else None
    )
    args = async_engine.AsyncEngineArgs(
        model="m",
        tokenizer="tokenizer-override",
        tensor_parallel_size=2,
    )
    backend = async_engine._default_backend(args)
    assert isinstance(backend, VLLMBackend)
    assert captured["tensor_parallel_size"] == 2
    assert captured["tokenizer"] == "tokenizer-override"


def test_instantiation_without_vllm_raises_clear_error(monkeypatch):
    real_import = vllm_backend.importlib.import_module

    def import_without_vllm(name):
        if name == "vllm":
            raise ImportError(name)
        return real_import(name)

    monkeypatch.setattr(vllm_backend.importlib, "import_module", import_without_vllm)
    with pytest.raises(RuntimeError, match="vllm"):
        VLLMBackend(model="meta-llama/Llama-3.1-8B")


@pytest.mark.parametrize(
    "prompt",
    [
        "legacy text",
        TextPrompt(prompt="legacy text"),
    ],
)
async def test_vllm_backend_forwards_text_variants_as_strings(monkeypatch, prompt):
    backend, engine = _capturing_backend(monkeypatch)
    await backend.generate(
        GenerationRequest(
            request_id="text",
            prompt=prompt,
            sampling_params=SamplingParams(max_tokens=1),
        )
    )
    assert engine.calls[0]["prompt"] == "legacy text"
    assert isinstance(engine.calls[0]["prompt"], str)
    assert "tokenization_kwargs" not in engine.calls[0]


async def test_vllm_backend_disables_special_tokens_only_for_templated_text(
    monkeypatch,
):
    backend, engine = _capturing_backend(monkeypatch)

    await backend.generate(
        GenerationRequest(
            request_id="templated",
            prompt=TemplatedPrompt("<BOS>user hello<ASSISTANT>"),
            sampling_params=SamplingParams(max_tokens=1),
        )
    )

    assert engine.calls[0]["prompt"] == "<BOS>user hello<ASSISTANT>"
    assert type(engine.calls[0]["prompt"]) is str
    assert engine.calls[0]["tokenization_kwargs"] == {
        "add_special_tokens": False,
    }


async def test_vllm_backend_forwards_token_ids_without_using_display_text(monkeypatch):
    backend, engine = _capturing_backend(monkeypatch)
    result = await backend.generate(
        GenerationRequest(
            request_id="tokens",
            prompt=TokensPrompt(
                prompt_token_ids=(11, 17, 23),
                prompt="display text must not be tokenized or forwarded",
            ),
            sampling_params=SamplingParams(max_tokens=1),
            priority=-4,
        )
    )
    assert engine.calls == [
        {
            "prompt": {"prompt_token_ids": [11, 17, 23]},
            "params": {
                "n": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "seed": None,
                "stop": [],
                "stop_token_ids": [],
                "max_tokens": 1,
                "min_tokens": 0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "ignore_eos": False,
                "skip_special_tokens": True,
            },
            "request_id": "tokens",
            "priority": -4,
        }
    ]
    assert result.prompt == TokensPrompt(
        prompt_token_ids=(11, 17, 23),
        prompt="display text must not be tokenized or forwarded",
    )
    assert result.prompt_token_ids == (11, 17, 23)


async def test_vllm_backend_resolves_omissions_but_preserves_explicit_neutrals(
    monkeypatch,
):
    _install_fake_vllm(monkeypatch)
    backend = VLLMBackend(model="m")
    engine = _CapturingEngine(
        {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
        }
    )
    backend._engine = engine
    omitted = SamplingParams(max_tokens=1).with_generation_config_omitted(
        GENERATION_CONFIG_SAMPLING_FIELDS
    )

    await backend.generate(
        GenerationRequest("omitted", "hello", omitted)
    )
    await backend.generate(
        GenerationRequest("explicit", "hello", SamplingParams(max_tokens=1))
    )

    assert backend.generation_defaults is not None
    assert backend.generation_defaults.source == "vllm-model-config"
    assert backend.generation_defaults.temperature == 0.6

    assert {
        field: engine.calls[0]["params"][field]
        for field in GENERATION_CONFIG_SAMPLING_FIELDS
    } == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
    }
    assert {
        field: engine.calls[1]["params"][field]
        for field in GENERATION_CONFIG_SAMPLING_FIELDS
    } == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }


async def test_vllm_backend_rejects_multimodal_before_engine_dispatch(monkeypatch):
    backend, engine = _capturing_backend(monkeypatch)
    request = GenerationRequest(
        request_id="mm",
        prompt=MultimodalPrompt(
            base=TextPrompt(prompt="describe this image"),
            items=(
                MultimodalItem(
                    modality="image",
                    encoding="uri",
                    data="https://example.test/image.png",
                ),
            ),
        ),
        sampling_params=SamplingParams(max_tokens=1),
    )
    with pytest.raises(ValueError, match="does not support multimodal"):
        await backend.generate(request)
    assert engine.calls == []


def test_vllm_backend_rejects_unrendered_tool_intent_for_token_prompt(monkeypatch):
    backend, _ = _capturing_backend(monkeypatch)
    request = GenerationRequest(
        request_id="tokens-tools",
        prompt=TokensPrompt(prompt_token_ids=(1, 2, 3)),
        sampling_params=SamplingParams(max_tokens=1),
        tools=(
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            },
        ),
    )
    with pytest.raises(ValueError, match="tool"):
        backend.validate_request(request)


async def test_vllm_backend_rejects_forced_tokens_before_engine_dispatch(monkeypatch):
    backend, engine = _capturing_backend(monkeypatch)
    request = GenerationRequest(
        request_id="forced",
        prompt="hello",
        sampling_params=SamplingParams(
            max_tokens=1,
            forced_token_ids=(7,),
            ignore_eos=True,
        ),
    )

    with pytest.raises(ValueError, match="forced_token_ids"):
        await backend.generate(request)

    assert engine.calls == []


@pytest.mark.vllm
@pytest.mark.hf_hub
async def test_generate_roundtrip_with_real_vllm():
    backend = VLLMBackend(model="facebook/opt-125m")

    result = await backend.generate(
        GenerationRequest(
            request_id="r1",
            prompt="Hello",
            sampling_params=SamplingParams(max_tokens=8),
        )
    )
    assert result.completions[0].text
    await backend.shutdown()
