#!/usr/bin/env python
"""Check declared test prerequisites without starting the test.

Exit status 77 means that the test was *not run* because a prerequisite is
absent.  It is intentionally non-zero so callers cannot mistake an unavailable
environment for passing product evidence.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

NOT_RUN_EXIT = 77


def _compute_capability(value: str) -> tuple[int, int]:
    try:
        major_text, minor_text = value.split(".", 1)
        capability = int(major_text), int(minor_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "compute capability must have the form MAJOR.MINOR"
        ) from error
    if capability[0] < 0 or capability[1] < 0:
        raise argparse.ArgumentTypeError("compute capability must be non-negative")
    return capability


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-gpus", type=int, default=0)
    parser.add_argument("--require-nccl", action="store_true")
    parser.add_argument("--require-peer-access", action="store_true")
    parser.add_argument(
        "--min-compute-capability",
        type=_compute_capability,
        metavar="MAJOR.MINOR",
    )
    parser.add_argument("--require-module", action="append", default=[])
    parser.add_argument("--require-executable", action="append", default=[])
    parser.add_argument("--require-path", action="append", type=Path, default=[])
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    missing: list[str] = []
    imported: dict[str, ModuleType] = {}

    required_modules = list(dict.fromkeys(args.require_module))
    if (
        args.min_gpus
        or args.require_nccl
        or args.require_peer_access
        or args.min_compute_capability is not None
    ):
        required_modules = list(dict.fromkeys(["torch", *required_modules]))
    for name in required_modules:
        try:
            imported[name] = importlib.import_module(name)
        except Exception as error:  # import/ABI failures are unavailable, not a test pass
            missing.append(f"python-module:{name}:{type(error).__name__}")

    executables: dict[str, str] = {}
    for name in dict.fromkeys(args.require_executable):
        resolved = shutil.which(name)
        if resolved is None:
            missing.append(f"executable:{name}")
        else:
            executables[name] = resolved

    paths: dict[str, str] = {}
    for path in dict.fromkeys(args.require_path):
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            missing.append(f"path:{path}")
        else:
            paths[str(path)] = str(resolved)

    gpu_count: int | None = None
    gpu_names: list[str] = []
    compute_capabilities: list[tuple[int, int]] = []
    torch = imported.get("torch")
    if torch is not None and (
        args.min_gpus
        or args.require_nccl
        or args.require_peer_access
        or args.min_compute_capability is not None
    ):
        try:
            cuda = torch.cuda
            gpu_count = int(cuda.device_count()) if cuda.is_available() else 0
            required_gpu_count = max(
                args.min_gpus,
                1 if args.min_compute_capability is not None else 0,
            )
            if gpu_count < required_gpu_count:
                missing.append(
                    f"cuda-devices:{required_gpu_count}-required:{gpu_count}-visible"
                )
            else:
                gpu_names = [
                    str(cuda.get_device_properties(index).name)
                    for index in range(gpu_count)
                ]
                compute_capabilities = [
                    tuple(int(part) for part in cuda.get_device_capability(index))
                    for index in range(gpu_count)
                ]
                if args.min_compute_capability is not None and any(
                    capability < args.min_compute_capability
                    for capability in compute_capabilities
                ):
                    required = ".".join(
                        str(part) for part in args.min_compute_capability
                    )
                    visible = ",".join(
                        ".".join(str(part) for part in capability)
                        for capability in compute_capabilities
                    )
                    missing.append(
                        f"cuda-compute-capability:{required}-required:{visible}-visible"
                    )
            if args.require_peer_access and (
                gpu_count < 2 or not bool(cuda.can_device_access_peer(0, 1))
            ):
                missing.append("cuda-peer-access:0-to-1")
        except Exception as error:
            missing.append(f"cuda-runtime:{type(error).__name__}")
        if args.require_nccl:
            try:
                if not bool(torch.distributed.is_nccl_available()):
                    missing.append("torch-distributed:nccl")
            except Exception as error:
                missing.append(f"torch-distributed:nccl:{type(error).__name__}")

    return {
        "schema_version": 1,
        "status": "not-run" if missing else "available",
        "missing": missing,
        "visible_gpu_count": gpu_count,
        "visible_gpu_names": gpu_names,
        "visible_compute_capabilities": compute_capabilities,
        "executables": executables,
        "paths": paths,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.min_gpus < 0:
        raise SystemExit("--min-gpus must be non-negative")
    result = evaluate(args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "available" else NOT_RUN_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
