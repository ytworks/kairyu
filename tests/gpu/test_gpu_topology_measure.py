"""The measuring half of the topology module needs real devices (runbook §6.1)."""

import pytest
import torch

pytestmark = pytest.mark.gpu

# small enough to stay quick, large enough that setup cost does not dominate
SIZE_MB = 64


def _require_devices(count: int) -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("interconnect measurement needs CUDA")
    if torch.cuda.device_count() < count:  # pragma: no cover - small box
        pytest.skip(f"need {count} CUDA devices, found {torch.cuda.device_count()}")


def test_device_bandwidth_is_plausible():
    from kairyu.engine.core.gpu_topology import measure_device_bandwidth

    _require_devices(1)
    gbs = measure_device_bandwidth(torch, size_mb=SIZE_MB, iters=3)
    # a GPU that cannot clear 50 GB/s device-local is not a GPU we can serve on;
    # the ceiling catches a timer that measured nothing at all
    assert 50.0 < gbs < 100_000.0, gbs


def test_p2p_matrix_is_square_and_diagonal_dominant():
    from kairyu.engine.core.gpu_topology import measure_p2p_matrix

    _require_devices(2)
    count = torch.cuda.device_count()
    matrix = measure_p2p_matrix(torch, count, size_mb=SIZE_MB, iters=2)

    assert len(matrix) == count
    assert all(len(row) == count for row in matrix)
    for index, row in enumerate(matrix):
        peers = [value for position, value in enumerate(row) if position != index]
        # device-local must beat every peer hop; if it does not, the copies were
        # not going where the indices claim
        assert row[index] > max(peers), f"row {index}: {row}"
        assert all(value > 0 for value in peers)


def test_probed_interconnect_matches_the_topology_report():
    import subprocess

    from kairyu.engine.core.gpu_topology import (
        interconnect_from_topology,
        parse_topology_matrix,
    )

    _require_devices(2)
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=True
    ).stdout
    matrix = parse_topology_matrix(topology)
    assert matrix, "nvidia-smi topo -m produced no parseable GPU rows"
    assert interconnect_from_topology(matrix) in ("pcie", "nvlink")
