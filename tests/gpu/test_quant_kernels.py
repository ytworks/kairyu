"""Deploy-day mirror: GPU quant kernels vs the CPU dequant oracles (m14 D4)."""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def cuda():
    if not torch.cuda.is_available():  # pragma: no cover - deploy-day only
        pytest.skip("CUDA required")
    return "cuda"


def test_fp8_kernel_matches_cpu_reference(cuda):
    from kairyu.kernels.fp8_gemm_gpu import linear_forward
    from kairyu.quant.fp8 import quantize_fp8
    from kairyu.quant.linear import Fp8Linear

    torch.manual_seed(0)
    module = Fp8Linear(64, 32, bias=False)
    weight = torch.randn(32, 64)
    q, scale = quantize_fp8(weight)
    module.weight.copy_(q)
    module.weight_scale.copy_(scale)
    x = torch.randn(4, 64)
    cpu_out = module.forward(x)
    module = module.to(cuda)
    gpu_out = linear_forward(x.to(cuda), module).cpu()
    assert (gpu_out - cpu_out).abs().max().item() < 2e-2


def test_awq_kernel_matches_cpu_reference(cuda):
    from kairyu.kernels.awq_gemm_gpu import linear_forward
    from kairyu.quant.awq import quantize_awq
    from kairyu.quant.linear import AwqLinear

    torch.manual_seed(1)
    module = AwqLinear(64, 32, bias=False, group_size=16)
    qweight, qzeros, scales = quantize_awq(torch.randn(32, 64), group_size=16)
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    x = torch.randn(4, 64)
    cpu_out = module.forward(x)
    module = module.to(cuda)
    gpu_out = linear_forward(x.to(cuda), module).cpu()
    assert (gpu_out - cpu_out).abs().max().item() < 2e-2
