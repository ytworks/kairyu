#!/usr/bin/env python
"""Write the runbook §0 hardware and library environment record."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import subprocess
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from kairyu.engine.core.gpu_topology import (
    interconnect_from_topology,
    measure_device_bandwidth,
    measure_p2p_matrix,
    parse_topology_matrix,
)
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
    # off by default: measuring allocates GB on every visible GPU, so callers opt
    # in. `main()` turns it on, which is where the hardware actually is.
    measure: bool = False,
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
    # probe() hardcodes "pcie" because it runs on every engine start and cannot
    # shell out; the topology matrix is the authority, and this is the one place
    # that pays for reading it (runbook §6.1)
    links = parse_topology_matrix(topology)
    profile = dataclasses.replace(
        profile, interconnect=interconnect_from_topology(links) or profile.interconnect
    )
    if measure:
        profile = dataclasses.replace(
            profile,
            measured_bandwidth_gbs=round(measure_device_bandwidth(torch), 2),
            p2p_matrix=measure_p2p_matrix(torch, profile.device_count),
        )
        measurement_note = (
            "Bandwidth and P2P are MEASURED (GB/s): p2p_matrix[i][j] is a "
            "device-to-device copy from i to j, the diagonal is a device-local "
            "copy counting read+write."
        )
    else:
        measurement_note = (
            "Numeric bandwidth/P2P measurements were skipped (--no-measure) and "
            "remain null."
        )
    notes = (
        f"{measurement_note}\n\nGPU topology and inventory audit follows.\n\n"
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
    parser.add_argument(
        "--no-measure",
        action="store_true",
        help="record topology only; leaves bandwidth/P2P null (fast, and safe to "
        "run while the GPUs are serving)",
    )
    args = parser.parse_args(argv)

    profile = probe()
    if profile.arch != "cuda":
        parser.error("CUDA hardware is required")
    record = build_env_record(
        profile=profile, record_date=args.date, measure=not args.no_measure
    )
    path = write_env_record(record, args.results_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
