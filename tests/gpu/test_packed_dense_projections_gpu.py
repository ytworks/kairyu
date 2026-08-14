"""Real-SM dense projection packing at Qwen3-32B TP4 decode shapes."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from evals.profiling import profile_scope
from kairyu.models.config import parse_model_config
from kairyu.models.llama import DenseDecoder

pytestmark = pytest.mark.gpu

_QWEN32B_TP4_LOCAL = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 5120,
    "num_hidden_layers": 1,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "intermediate_size": 6400,
    "vocab_size": 256,
    "max_position_embeddings": 40960,
    "rms_norm_eps": 1e-6,
}


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("packed dense projection gate needs CUDA")


def _fp32_error_metrics(
    actual: torch.Tensor,
    oracle: torch.Tensor,
) -> tuple[float, float, float]:
    error = actual.float() - oracle
    return (
        float(error.abs().max().detach()),
        float(error.square().mean().sqrt().detach()),
        float(
            (
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(oracle)
            ).detach()
        ),
    )


@pytest.fixture(scope="module")
def actual_shape_layer():
    _require_cuda()
    torch.manual_seed(156)
    model = DenseDecoder(parse_model_config(_QWEN32B_TP4_LOCAL)).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    return model.model.layers[0]


def test_qkv_is_one_linear_at_binding_decode_shape(actual_shape_layer):
    layer = actual_shape_layer
    torch.manual_seed(157)
    hidden = torch.randn(16, 5120, device="cuda", dtype=torch.bfloat16)

    # Warm dispatcher/cuBLAS selection before observing the structural call.
    layer.self_attn._project_qkv(hidden)
    torch.cuda.synchronize()
    with profile_scope(
        enabled=True,
        activities=("cpu", "cuda"),
        acc_events=False,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as qkv_profile:
        actual_qkv = layer.self_attn._project_qkv(hidden)
        torch.cuda.synchronize()

    qkv_linears = sum(
        event.count
        for event in qkv_profile.key_averages()
        if event.key == "aten::linear"
    )
    assert qkv_linears == 1
    assert not hasattr(layer.mlp, "_gate_up_pack")

    separate_qkv = (
        layer.self_attn.q_proj(hidden),
        layer.self_attn.k_proj(hidden),
        layer.self_attn.v_proj(hidden),
    )
    fp32_oracles = tuple(
        F.linear(hidden.float(), projection.weight.float(), None)
        for projection in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        )
    )
    for actual, separate, oracle in zip(
        actual_qkv,
        separate_qkv,
        fp32_oracles,
        strict=True,
    ):
        packed_max, packed_rmse, packed_relative_l2 = (
            _fp32_error_metrics(actual, oracle)
        )
        separate_max, separate_rmse, separate_relative_l2 = (
            _fp32_error_metrics(separate, oracle)
        )

        # Bind useful numerical quality, rather than accepting a broad
        # packed-vs-separate tolerance.  The absolute limits are conservative
        # against four measured seeds at this exact Qwen3-32B TP4 shape, while
        # the relative limits prevent packing from materially worsening the
        # established BF16 projection.
        assert packed_max <= 0.015625
        assert packed_rmse <= 0.002
        assert packed_relative_l2 <= 0.003
        assert packed_max <= max(separate_max * 1.25, 0.0078125)
        assert packed_rmse <= separate_rmse * 1.05
        assert packed_relative_l2 <= separate_relative_l2 * 1.05


def test_qkv_packed_execution_is_cuda_graph_capturable(actual_shape_layer):
    attention = actual_shape_layer.self_attn
    static_hidden = torch.randn(16, 5120, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            attention._project_qkv(static_hidden)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = attention._project_qkv(static_hidden)
    replacement = torch.randn_like(static_hidden)
    static_hidden.copy_(replacement)
    graph.replay()
    torch.cuda.synchronize()
    replayed = tuple(output.clone() for output in captured)
    expected = attention._project_qkv(replacement)
    for actual, reference in zip(replayed, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
