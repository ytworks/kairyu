#!/usr/bin/env python3
"""Pinned external-runtime kernel comparison for issue #205.

Run inside the official vLLM v0.26.0 CUDA 12.9 image.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import vllm
from vllm import _custom_ops as ops


def _measure(fn, warmup: int, repeats: int) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    samples = sorted(
        start.elapsed_time(end)
        for start, end in zip(starts, ends, strict=True)
    )
    p95_index = min(len(samples) - 1, math.ceil(len(samples) * 0.95) - 1)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": samples[p95_index],
        "min_ms": samples[0],
        "samples": repeats,
        "warmup": warmup,
    }


def _error_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float]:
    error = actual.float() - expected.float()
    expected_rms = expected.float().square().mean().sqrt().clamp(min=1e-12)
    return {
        "relative_rmse": (
            error.square().mean().sqrt() / expected_rms
        ).item(),
        "mean_abs_error": error.abs().mean().item(),
        "max_abs_error": error.abs().max().item(),
    }


def _fp8(weight: torch.Tensor):
    qweight, weight_scale = ops.scaled_fp8_quant(
        weight, use_per_token_if_dynamic=True
    )

    def build(x):
        def forward():
            qactivation, activation_scale = ops.scaled_fp8_quant(
                x, use_per_token_if_dynamic=True
            )
            return ops.cutlass_scaled_mm(
                qactivation,
                qweight.T,
                activation_scale,
                weight_scale.T,
                torch.bfloat16,
            )

        return forward

    return build


def _int8(weight: torch.Tensor):
    qweight, weight_scale, _ = ops.scaled_int8_quant(weight)

    def build(x):
        def forward():
            qactivation, activation_scale, _ = ops.scaled_int8_quant(x)
            return ops.cutlass_scaled_mm(
                qactivation,
                qweight.T,
                activation_scale,
                weight_scale.T,
                torch.bfloat16,
            )

        return forward

    return build


def _nvfp4(weight: torch.Tensor):
    weight_scale = (
        weight.float().abs().amax() / (6.0 * 448.0)
    ).clamp(min=1e-12)
    qweight, weight_block_scale = ops.scaled_fp4_quant(
        weight, weight_scale.reciprocal()
    )

    def build(x):
        def forward():
            activation_scale = (
                x.float().abs().amax() / (6.0 * 448.0)
            ).clamp(min=1e-12)
            qactivation, activation_block_scale = ops.scaled_fp4_quant(
                x, activation_scale.reciprocal()
            )
            alpha = activation_scale * weight_scale
            return ops.cutlass_scaled_fp4_mm(
                qactivation,
                qweight,
                activation_block_scale,
                weight_block_scale,
                alpha,
                torch.bfloat16,
            )

        return forward

    return build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(205)
    weight = torch.randn(
        args.n, args.k, device="cuda", dtype=torch.bfloat16
    )
    records = []
    for scheme, prepare in (
        ("fp8", _fp8),
        ("int8", _int8),
        ("nvfp4", _nvfp4),
    ):
        build = prepare(weight)
        for m_size in args.m:
            x = torch.randn(
                m_size, args.k, device="cuda", dtype=torch.bfloat16
            )
            try:
                forward = build(x)
                quantized_output = forward()
                unquantized_output = F.linear(x, weight)
                record = {
                    "status": "ok",
                    **_measure(forward, warmup=args.warmup, repeats=args.repeats),
                    "correctness_vs_unquantized_bf16": _error_metrics(
                        quantized_output, unquantized_output
                    ),
                }
            except RuntimeError as error:
                record = {"status": "unsupported", "error": str(error)}
            records.append(
                {
                    "scheme": scheme,
                    "shape": {"m": m_size, "n": args.n, "k": args.k},
                    **record,
                }
            )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime": "vLLM",
        "vllm": vllm.__version__,
        "image": "vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404",
        "image_digest": (
            "sha256:4d08193d2fd05aadb1b5678f93ae609efb2635df67da45f3efe781c368b34dc8"
        ),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "method": (
            "vLLM scaled activation quantization + CUTLASS scaled MM; "
            "CUDA events"
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
