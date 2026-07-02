import importlib.util

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
