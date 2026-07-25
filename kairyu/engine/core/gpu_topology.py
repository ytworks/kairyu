"""GPU interconnect topology and measured P2P bandwidth (runbook §6.1).

`HardwareProfile` carries `interconnect`, `measured_bandwidth_gbs` and
`p2p_matrix`, and `probe()` deliberately leaves the last two unset: it runs on
every engine start and must stay cheap, so it cannot copy hundreds of megabytes
around. The env script fills them in once, per the G2 §8 evidence rule that a
number is recorded next to the config that produced it.

On the PCIe fleet the matrix is not a formality. `nvidia-smi topo -m` on the
8x RTX PRO 6000 host reports pairs (0,1) (2,3) (4,5) (6,7) as NODE — same NUMA
node — and every other pair as SYS, across the socket. A TP group laid out
across SYS links moves each RowParallelLinear all_reduce over the slow path, so
the placement decision needs the measured numbers, not the nominal link rate.

The parsing half is pure and CPU-tested; only the measuring half needs hardware.
"""

from __future__ import annotations

import re

#: `nvidia-smi topo -m` link codes, fastest first. NV# is an NVLink bundle.
_LINK_ORDER = ("NV", "PIX", "PXB", "PHB", "NODE", "SYS")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_GPU_LABEL = re.compile(r"^GPU(\d+)$")


def parse_topology_matrix(text: str) -> dict[tuple[int, int], str]:
    """`nvidia-smi topo -m` -> {(row_gpu, col_gpu): link code}.

    NIC rows/columns and the affinity columns are dropped; the diagonal keeps
    its literal ``X``. Unparseable input yields an empty map rather than an
    exception — this feeds an evidence record, never a control decision.
    """
    clean = _ANSI.sub("", text)
    lines = [line for line in clean.splitlines() if line.strip()]
    header: list[int] | None = None
    matrix: dict[tuple[int, int], str] = {}
    for line in lines:
        fields = line.split()
        if header is None:
            # the header has no row label, so every field is a column name
            gpu_columns = [
                index for index, name in enumerate(fields) if _GPU_LABEL.match(name)
            ]
            if not gpu_columns:
                continue
            header = [int(_GPU_LABEL.match(fields[i]).group(1)) for i in gpu_columns]
            continue
        label = _GPU_LABEL.match(fields[0])
        if label is None:
            continue  # NIC rows, the legend
        row = int(label.group(1))
        for position, column in enumerate(header):
            value_index = position + 1  # +1 for the row label
            if value_index >= len(fields):
                break
            matrix[(row, column)] = fields[value_index]
    return matrix


def interconnect_from_topology(matrix: dict[tuple[int, int], str]) -> str:
    """"nvlink" when any GPU pair is bridged by NVLink, else "pcie"."""
    for (row, column), link in matrix.items():
        # NVLink bundles are NV1..NV18; NODE is a NUMA-local PCIe hop, not NVLink
        if row != column and link.startswith("NV"):
            return "nvlink"
    return "pcie" if matrix else "none"


def slowest_link(matrix: dict[tuple[int, int], str], ranks: tuple[int, ...]) -> str | None:
    """The worst link inside a candidate TP group — its all_reduce floor."""
    worst: str | None = None
    worst_rank = -1
    for row in ranks:
        for column in ranks:
            if row == column:
                continue
            link = matrix.get((row, column))
            if link is None:
                continue
            index = next(
                (i for i, prefix in enumerate(_LINK_ORDER) if link.startswith(prefix)),
                len(_LINK_ORDER),
            )
            if index > worst_rank:
                worst, worst_rank = link, index
    return worst


def measure_device_bandwidth(
    torch, device: int = 0, size_mb: int = 512, iters: int = 5
) -> float:
    """Device-local copy bandwidth in GB/s (read + write counted once each)."""
    import time

    elements = size_mb * 1024 * 1024 // 4
    with torch.cuda.device(device):
        source = torch.empty(elements, dtype=torch.float32, device=f"cuda:{device}")
        target = torch.empty_like(source)
        target.copy_(source)  # warm up allocator + kernels before timing
        torch.cuda.synchronize(device)
        best = float("inf")
        for _ in range(iters):
            start = time.perf_counter()
            target.copy_(source)
            torch.cuda.synchronize(device)
            best = min(best, time.perf_counter() - start)
    moved = source.numel() * source.element_size() * 2  # read + write
    return moved / best / 1e9


def measure_p2p_matrix(
    torch, device_count: int, size_mb: int = 256, iters: int = 3
) -> tuple[tuple[float, ...], ...]:
    """Ordered-pair GPU->GPU copy bandwidth in GB/s; diagonal is device-local.

    Deliberately measures the path as configured rather than after forcing peer
    access on: what a TP group will actually get is the number worth recording.
    """
    import time

    elements = size_mb * 1024 * 1024 // 4
    rows: list[tuple[float, ...]] = []
    for source_device in range(device_count):
        row: list[float] = []
        source = torch.empty(
            elements, dtype=torch.float32, device=f"cuda:{source_device}"
        )
        for target_device in range(device_count):
            if source_device == target_device:
                row.append(measure_device_bandwidth(torch, source_device, size_mb, iters))
                continue
            target = torch.empty(
                elements, dtype=torch.float32, device=f"cuda:{target_device}"
            )
            target.copy_(source)  # warm up the peer path before timing
            torch.cuda.synchronize(source_device)
            torch.cuda.synchronize(target_device)
            best = float("inf")
            for _ in range(iters):
                start = time.perf_counter()
                target.copy_(source)
                torch.cuda.synchronize(target_device)
                best = min(best, time.perf_counter() - start)
            moved = source.numel() * source.element_size()
            row.append(moved / best / 1e9)
            del target
            torch.cuda.empty_cache()
        rows.append(tuple(round(value, 2) for value in row))
        del source
        torch.cuda.empty_cache()
    return tuple(rows)
