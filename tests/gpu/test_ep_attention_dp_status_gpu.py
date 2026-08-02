"""GPU coverage for attention-DP's deferred rank-status sidecar."""

from types import SimpleNamespace

import pytest
import torch

from kairyu.engine.core import worker as worker_module

pytestmark = pytest.mark.gpu


def test_attention_dp_status_failure_is_copied_before_host_validation(
    monkeypatch,
) -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("attention-DP status sidecar needs CUDA")

    gathered = torch.tensor(
        (
            (worker_module._EP_ATTENTION_DP_PACKET_OK, 11),
            (worker_module._EP_ATTENTION_DP_PACKET_OK, 12),
            (worker_module._EP_ATTENTION_DP_PACKET_FAILED, -1),
            (worker_module._EP_ATTENTION_DP_PACKET_OK, 14),
        ),
        dtype=torch.int64,
        device="cuda",
    ).reshape(-1)
    runner = SimpleNamespace(_output_copy_stream=torch.cuda.Stream())

    original_tolist = torch.Tensor.tolist
    original_item = torch.Tensor.item
    original_cpu = torch.Tensor.cpu

    def forbid_cuda_tolist(tensor, *args, **kwargs):
        if tensor.device.type == "cuda":
            raise AssertionError("CUDA Tensor.tolist() before deferred status copy")
        return original_tolist(tensor, *args, **kwargs)

    def forbid_cuda_item(tensor, *args, **kwargs):
        if tensor.device.type == "cuda":
            raise AssertionError("CUDA Tensor.item() before deferred status copy")
        return original_item(tensor, *args, **kwargs)

    def forbid_cuda_cpu(tensor, *args, **kwargs):
        if tensor.device.type == "cuda":
            raise AssertionError("CUDA Tensor.cpu() before deferred status copy")
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "tolist", forbid_cuda_tolist)
    monkeypatch.setattr(torch.Tensor, "item", forbid_cuda_item)
    monkeypatch.setattr(torch.Tensor, "cpu", forbid_cuda_cpu)

    output = worker_module._ep_attention_dp_deferred_status_output(
        gathered,
        chunk_count=1,
        world_size=4,
        local_runner=runner,
    )

    with pytest.raises(
        RuntimeError,
        match="rank 2: local execution or token-packet encoding failed",
    ):
        output.resolve_sidecars()
