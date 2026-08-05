"""Output-head projection policies for target-model logits.

The float32 policy is implemented at the GEMM boundary. Casting an already
rounded BF16 result would change only the tensor dtype, not recover precision.
"""

from __future__ import annotations

import torch
from torch import nn

from kairyu.engine.core.logits_dtype import validate_logits_dtype


def project_logits(
    head: nn.Module,
    hidden: torch.Tensor,
    logits_dtype: str,
) -> torch.Tensor:
    """Apply ``head`` using model precision or an FP32-output final GEMM."""

    selected = validate_logits_dtype(logits_dtype)
    if selected == "model":
        return head(hidden)
    if not isinstance(head, nn.Linear):
        raise RuntimeError(
            "logits_dtype='float32' requires a dense torch.nn.Linear output head"
        )
    if hidden.ndim < 1 or hidden.shape[-1] != head.in_features:
        raise ValueError(
            "output-head hidden dimension does not match lm_head.in_features"
        )
    weight = head.weight
    if hidden.device != weight.device:
        raise RuntimeError(
            "logits_dtype='float32' requires hidden states and lm_head weight "
            "on the same device"
        )
    flat = hidden.reshape(-1, hidden.shape[-1])
    if flat.device.type == "cuda":
        if flat.dtype in (torch.float16, torch.bfloat16):
            if weight.dtype is not flat.dtype:
                raise RuntimeError(
                    "logits_dtype='float32' requires lm_head weight and hidden "
                    "states to use the same model dtype"
                )
            output = torch.mm(flat, weight.t(), out_dtype=torch.float32)
        elif flat.dtype is torch.float32 and weight.dtype is torch.float32:
            output = torch.mm(flat, weight.t())
        else:
            raise RuntimeError(
                "logits_dtype='float32' CUDA output head requires matching "
                "float16, bfloat16, or float32 operands"
            )
    elif (
        flat.device.type == "cpu"
        and flat.dtype is torch.float32
        and weight.dtype is torch.float32
    ):
        output = torch.mm(flat, weight.t())
    else:
        raise RuntimeError(
            "logits_dtype='float32' supports CUDA float16/bfloat16 models and "
            "models already computing in float32"
        )
    if head.bias is not None:
        output = output + head.bias.to(torch.float32)
    return output.reshape(*hidden.shape[:-1], head.out_features)
