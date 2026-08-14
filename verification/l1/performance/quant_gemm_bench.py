#!/usr/bin/env python3
"""Reproducible fused-quant GEMM benchmark for issue #205."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from kairyu.quant import awq, fp8, gptq, int8, nvfp4
from kairyu.quant.linear import (
    AwqLinear,
    Fp8Linear,
    GptqLinear,
    Int8Linear,
    NvFp4Linear,
)

SCHEMES = ("fp8", "int8", "awq", "gptq", "nvfp4")


def _module(scheme: str, weight: torch.Tensor):
    out_features, in_features = weight.shape
    if scheme == "fp8":
        module = Fp8Linear(in_features, out_features, bias=False)
        qweight, scale = fp8.quantize_fp8(weight)
        module.weight.copy_(qweight)
        module.weight_scale.copy_(scale)
        return module
    if scheme == "int8":
        module = Int8Linear(in_features, out_features, bias=False)
        qweight, scale = int8.quantize_int8_weight(weight)
        module.weight.copy_(qweight)
        module.weight_scale.copy_(scale)
        return module
    if scheme == "awq":
        module = AwqLinear(in_features, out_features, bias=False, group_size=128)
        qweight, qzeros, scales = awq.quantize_awq(weight, group_size=128)
        module.qweight.copy_(qweight)
        module.qzeros.copy_(qzeros)
        module.scales.copy_(scales)
        return module
    if scheme == "gptq":
        module = GptqLinear(in_features, out_features, bias=False, group_size=128)
        qweight, qzeros, scales, g_idx = gptq.quantize_gptq(
            weight, group_size=128
        )
        module.qweight.copy_(qweight)
        module.qzeros.copy_(qzeros)
        module.scales.copy_(scales)
        module.g_idx.copy_(g_idx)
        return module
    module = NvFp4Linear(in_features, out_features, bias=False)
    qweight, scale, global_scale = nvfp4.quantize_nvfp4(weight)
    module.weight.copy_(qweight)
    module.weight_scale.copy_(scale)
    module.weight_scale_2.copy_(global_scale)
    module.input_scale.copy_(torch.tensor(4.0 / (6.0 * 448.0)))
    return module


def _measure(fn, warmup: int, repeats: int) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
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
    peak_delta = torch.cuda.max_memory_allocated() - baseline
    p95_index = min(len(samples) - 1, math.ceil(len(samples) * 0.95) - 1)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": samples[p95_index],
        "min_ms": samples[0],
        "peak_temporary_bytes": peak_delta,
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


def _quantized_reference(module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(module, Int8Linear):
        # CUDA does not implement int32 GEMM. Float32 accumulation is an
        # independent benchmark oracle here; the GPU tests retain the exact
        # int32 CPU oracle on smaller matrices.
        x_q, x_scale = int8.quantize_int8_activation(x.float())
        accumulated = torch.matmul(
            x_q.float(), module.weight.float().T
        )
        return accumulated * x_scale * module.weight_scale.reshape(-1)
    return module.forward_reference(x.float())


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_dir = Path(".git")
        head = (git_dir / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            return (git_dir / head.removeprefix("ref: ")).read_text().strip()
        return head


def _git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Benchmarks are normally run from the issue worktree before commit.
        return True


def _driver_version() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "--id=0",
        ],
        text=True,
    ).splitlines()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.k % 128 or args.n % 8:
        raise ValueError(
            "K must be divisible by 128 and N by 8 for all schemes"
        )

    torch.manual_seed(205)
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    records = []
    for scheme in SCHEMES:
        torch.manual_seed(205)
        weight = torch.randn(args.n, args.k)
        module = _module(scheme, weight).to(device)
        dense_weight = weight.to(device=device, dtype=torch.bfloat16)
        for m_size in args.m:
            x = torch.randn(m_size, args.k, device=device, dtype=torch.bfloat16)
            fused_output = module(x)
            quantized_reference = _quantized_reference(module, x)
            unquantized_output = F.linear(x, dense_weight)

            fused = _measure(
                lambda module=module, x=x: module(x),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            dequantized = _measure(
                lambda module=module, x=x: F.linear(
                    x, module.dequantize().to(torch.bfloat16)
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            bf16 = _measure(
                lambda dense_weight=dense_weight, x=x: F.linear(x, dense_weight),
                warmup=args.warmup,
                repeats=args.repeats,
            )
            records.append(
                {
                    "scheme": scheme,
                    "shape": {"m": m_size, "n": args.n, "k": args.k},
                    "fused": fused,
                    "dequantize_then_bf16": dequantized,
                    "unquantized_bf16": bf16,
                    "correctness": {
                        "fused_vs_quantized_reference": _error_metrics(
                            fused_output, quantized_reference
                        ),
                        "quantized_reference_vs_unquantized_bf16": _error_metrics(
                            quantized_reference, unquantized_output
                        ),
                    },
                    "speedup_vs_dequantize": (
                        dequantized["median_ms"] / fused["median_ms"]
                    ),
                    "speedup_vs_bf16": bf16["median_ms"] / fused["median_ms"],
                }
            )
        torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "device": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
            "driver": _driver_version(),
        },
        "method": {
            "dtype": "bfloat16",
            "seed": 205,
            "timing": "CUDA events; one synchronize after all recorded samples",
            "paths": {
                "fused": (
                    "QuantizedLinear.forward -> selected native/fused GPU kernel"
                ),
                "dequantize_then_bf16": "full weight dequantize + torch F.linear",
                "unquantized_bf16": "torch F.linear over original BF16 weight",
            },
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
