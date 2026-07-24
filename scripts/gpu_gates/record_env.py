#!/usr/bin/env python
"""Write the runbook §0 hardware and library environment record."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from kairyu.engine.core.hw_profile import (
    EnvRecord,
    HardwareProfile,
    probe,
    write_env_record,
)

CommandRunner = Callable[[Sequence[str]], str]


def _run(command: Sequence[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def build_env_record(
    *,
    profile: HardwareProfile,
    record_date: str,
    run_command: CommandRunner = _run,
) -> EnvRecord:
    import torch

    driver = run_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    ).splitlines()[0]
    topology = run_command(["nvidia-smi", "topo", "-m"])
    inventory = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,pci.bus_id,memory.total,mig.mode.current,vbios_version",
            "--format=csv,noheader",
        ]
    )
    notes = (
        "GPU topology and inventory audit follows. Numeric bandwidth/P2P "
        "measurements are unmeasured and remain null.\n\n"
        f"topology:\n{topology}\n\ninventory:\n{inventory}"
    )
    return EnvRecord(
        date=record_date,
        profile=profile,
        driver=driver,
        cuda=torch.version.cuda,
        library_versions={
            name: _version(distribution)
            for name, distribution in (
                ("torch", "torch"),
                ("flashinfer", "flashinfer-python"),
                ("triton", "triton"),
                ("nixl", "nixl"),
                ("vllm", "vllm"),
                ("sglang", "sglang"),
            )
        },
        notes=notes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--results-dir", type=Path, default=Path("bench/results"))
    args = parser.parse_args(argv)

    profile = probe()
    if profile.arch != "cuda":
        parser.error("CUDA hardware is required")
    record = build_env_record(profile=profile, record_date=args.date)
    path = write_env_record(record, args.results_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
