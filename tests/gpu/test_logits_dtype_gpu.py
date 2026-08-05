"""Real-CUDA contracts for the opt-in FP32 output-head GEMM (#364)."""

from __future__ import annotations

import pytest
import torch

from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import EngineSampling
from kairyu.models.config import parse_model_config
from kairyu.models.llama import DenseDecoder
from kairyu.models.logits import project_logits

pytestmark = pytest.mark.gpu

_TINY = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "intermediate_size": 32,
    "vocab_size": 2,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
}


def _require_cuda() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU-only host
        pytest.skip("FP32-output final GEMM needs CUDA")


def _near_tie_decoder() -> DenseDecoder:
    """Two BF16 rows whose FP32 dot products differ below one BF16 output ULP."""

    decoder = DenseDecoder(
        parse_model_config(_TINY),
        logits_dtype="float32",
    ).to(device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        decoder.lm_head.weight.fill_(1.0)
        # BF16 represents this weight exactly.  Summing sixteen terms gives
        # 15.9921875 versus 16.0 in FP32, but both round to 16.0 as BF16 logits.
        decoder.lm_head.weight[0, 0] = 0.9921875
    return decoder


def test_fp32_output_resolves_a_tie_lost_by_bf16_and_is_not_posthoc_cast() -> None:
    _require_cuda()
    decoder = _near_tie_decoder()
    hidden = torch.ones(16, device="cuda:0", dtype=torch.bfloat16)

    model_logits = project_logits(decoder.lm_head, hidden, "model")
    fp32_logits = decoder.logits(hidden)
    oracle = torch.nn.functional.linear(
        hidden.float(), decoder.lm_head.weight.float()
    )

    assert model_logits.dtype is torch.bfloat16
    assert torch.equal(model_logits, torch.tensor([16.0, 16.0], device="cuda:0"))
    assert torch.argmax(model_logits).item() == 0
    assert fp32_logits.dtype is torch.float32
    assert torch.equal(fp32_logits, oracle)
    assert torch.argmax(fp32_logits).item() == 1
    assert not torch.equal(fp32_logits, model_logits.float())


def test_fp32_output_reaches_greedy_and_raw_logprob_reporting_before_selection() -> None:
    _require_cuda()
    decoder = _near_tie_decoder()
    hidden = torch.ones(16, device="cuda:0", dtype=torch.bfloat16)
    model_logits = project_logits(decoder.lm_head, hidden, "model")
    fp32_logits = decoder.logits(hidden)
    sampling = EngineSampling(temperature=0.0, logprobs=2)

    model_sample = Sampler().sample_device("model", sampling, 0, model_logits)
    fp32_sample = Sampler().sample_device("float32", sampling, 0, fp32_logits)
    expected_raw = torch.log_softmax(fp32_logits, dim=-1)

    assert int(model_sample.token_id.cpu()) == 0
    assert int(fp32_sample.token_id.cpu()) == 1
    assert fp32_sample.logprob is not None
    assert torch.equal(fp32_sample.logprob, expected_raw[1])
    assert fp32_sample.top_indices is not None
    assert int(fp32_sample.top_indices[0].cpu()) == 1


@pytest.mark.parametrize(
    ("sampling", "position"),
    (
        pytest.param(
            EngineSampling(temperature=0.01, seed=364),
            1,
            id="temperature",
        ),
        pytest.param(
            EngineSampling(temperature=1.0, min_p=0.995, seed=2),
            0,
            id="min-p",
        ),
        pytest.param(
            EngineSampling(temperature=1.0, top_k=1, seed=2),
            0,
            id="top-k",
        ),
        pytest.param(
            EngineSampling(temperature=1.0, top_p=0.501, seed=2),
            0,
            id="top-p",
        ),
    ),
)
def test_fp32_near_tie_is_not_downcast_before_sampling_boundaries(
    sampling,
    position,
) -> None:
    _require_cuda()
    decoder = _near_tie_decoder()
    hidden = torch.ones(16, device="cuda:0", dtype=torch.bfloat16)
    model_logits = project_logits(decoder.lm_head, hidden, "model")
    fp32_logits = decoder.logits(hidden)

    model_sample = Sampler().sample_device(
        "model",
        sampling,
        position,
        model_logits,
    )
    fp32_sample = Sampler().sample_device(
        "float32",
        sampling,
        position,
        fp32_logits,
    )

    # The exact seed/position fixtures make the rounded BF16 tie choose token
    # 0 while the preserved FP32 gap chooses token 1.  A hidden downcast before
    # temperature scaling or filtering would collapse both arms back to 0.
    assert int(model_sample.token_id.cpu()) == 0
    assert int(fp32_sample.token_id.cpu()) == 1


@pytest.mark.parametrize("leading_shape", [(), (4,), (2, 3)])
def test_fp32_output_preserves_all_leading_dimensions(leading_shape) -> None:
    _require_cuda()
    decoder = _near_tie_decoder()
    hidden = torch.randn(
        *leading_shape,
        _TINY["hidden_size"],
        device="cuda:0",
        dtype=torch.bfloat16,
    )

    actual = decoder.logits(hidden)
    expected = torch.nn.functional.linear(
        hidden.reshape(-1, _TINY["hidden_size"]).float(),
        decoder.lm_head.weight.float(),
    ).reshape(*leading_shape, _TINY["vocab_size"])

    assert actual.shape == (*leading_shape, _TINY["vocab_size"])
    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fp32_output_keeps_the_bf16_head_weight_and_state_dict_identity() -> None:
    _require_cuda()
    decoder = _near_tie_decoder()
    weight = decoder.lm_head.weight
    weight_pointer = weight.data_ptr()
    state_keys = tuple(decoder.state_dict())

    output = decoder.logits(
        torch.ones(8, _TINY["hidden_size"], device="cuda:0", dtype=torch.bfloat16)
    )

    assert output.dtype is torch.float32
    assert decoder.lm_head.weight is weight
    assert decoder.lm_head.weight.dtype is torch.bfloat16
    assert decoder.lm_head.weight.data_ptr() == weight_pointer
    assert tuple(decoder.state_dict()) == state_keys


def test_fp32_output_head_is_cuda_graph_capture_and_replay_safe() -> None:
    _require_cuda()
    decoder = _near_tie_decoder()
    hidden = torch.ones(4, _TINY["hidden_size"], device="cuda:0", dtype=torch.bfloat16)

    for _ in range(3):
        decoder.logits(hidden)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = decoder.logits(hidden)

    hidden.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.nn.functional.linear(
        hidden.float(), decoder.lm_head.weight.float()
    )

    assert captured.dtype is torch.float32
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
