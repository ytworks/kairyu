import importlib.util
import sys
import types

import pytest

from kairyu import SamplingParams
from kairyu.engine import vllm_backend
from kairyu.engine.vllm_backend import VLLMBackend, to_vllm_sampling_kwargs

VLLM_INSTALLED = importlib.util.find_spec("vllm") is not None


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


def test_tensor_parallel_size_reaches_vllm_engine_args(monkeypatch):
    captured = _install_fake_vllm(monkeypatch)
    VLLMBackend(model="m", tensor_parallel_size=4)
    assert captured["model"] == "m"
    assert captured["tensor_parallel_size"] == 4


def test_llm_default_backend_forwards_tensor_parallel_size_to_vllm(monkeypatch):
    from kairyu.entrypoints import llm as llm_module

    captured = _install_fake_vllm(monkeypatch)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "vllm" else None
    )
    backend = llm_module._default_backend("m", None, tensor_parallel_size=8)
    assert isinstance(backend, VLLMBackend)
    assert captured["tensor_parallel_size"] == 8


def test_async_engine_default_backend_forwards_tensor_parallel_size_to_vllm(monkeypatch):
    from kairyu.entrypoints import async_engine

    captured = _install_fake_vllm(monkeypatch)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "vllm" else None
    )
    args = async_engine.AsyncEngineArgs(model="m", tensor_parallel_size=2)
    backend = async_engine._default_backend(args)
    assert isinstance(backend, VLLMBackend)
    assert captured["tensor_parallel_size"] == 2


@pytest.mark.skipif(VLLM_INSTALLED, reason="vLLM installed; error path not applicable")
def test_instantiation_without_vllm_raises_clear_error():
    with pytest.raises(RuntimeError, match="vllm"):
        VLLMBackend(model="meta-llama/Llama-3.1-8B")


@pytest.mark.skipif(not VLLM_INSTALLED, reason="requires vLLM")
async def test_generate_roundtrip_with_real_vllm():
    backend = VLLMBackend(model="facebook/opt-125m")
    from kairyu.engine.backend import GenerationRequest

    result = await backend.generate(
        GenerationRequest(
            request_id="r1",
            prompt="Hello",
            sampling_params=SamplingParams(max_tokens=8),
        )
    )
    assert result.completions[0].text
    await backend.shutdown()
